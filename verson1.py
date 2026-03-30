from ultralytics import YOLO
import cv2
import numpy as np
import os
import yaml
import time
from datetime import datetime
import torch

roi_selecting = False
roi_complete = False
roi_start_point = (0, 0)
roi_end_point = (0, 0)
roi_rect = None
drawing = False

detection_status = "Safe"  
current_obj_count = 0  # 当前检测到的目标数量
last_frame_time = 0  # 上一帧的时间
current_fps = 0  # 当前帧率
ui_position = None  # UI位置mub

def mouse_callback(event, x, y, flags, param):
    global roi_selecting, roi_start_point, roi_end_point, drawing, roi_complete
    
    if roi_selecting:
        if event == cv2.EVENT_LBUTTONDOWN: 
            drawing = True
            roi_start_point = (x, y)
            roi_complete = False
            
        elif event == cv2.EVENT_MOUSEMOVE and drawing:  
            roi_end_point = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi_end_point = (x, y)
            roi_complete = True 

def get_roi(frame, window_name="Detection Result", instruction="draw the area,then press ENTER...", window_width=1280, window_height=720):
    global roi_selecting, roi_start_point, roi_end_point, drawing, roi_complete, roi_rect, ui_position
    
    clone = frame.copy()
    roi_selecting = True
    roi_complete = False
    
    height, width = frame.shape[:2]
    
    ui_x = 20
    ui_y = 20
    ui_width = 400
    ui_height = 60
    ui_position = (ui_x, ui_y, ui_width, ui_height)
    
    # 创建窗口并设置大小（使用最终要用的窗口名称和大小）
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        # 制作一个临时显示
        temp = clone.copy()
        cv2.putText(temp, instruction, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 如果正在绘制或已完成绘制，显示矩形
        if drawing or roi_complete:
            cv2.rectangle(temp, roi_start_point, roi_end_point, (0, 255, 0), 2)
        
        cv2.imshow(window_name, temp)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and roi_complete:  # Enter键
            # 坐标从左上到右下的格式
            x1 = min(roi_start_point[0], roi_end_point[0])
            y1 = min(roi_start_point[1], roi_end_point[1])
            x2 = max(roi_start_point[0], roi_end_point[0])
            y2 = max(roi_start_point[1], roi_end_point[1])
            
            # 保存ROI矩形
            roi_rect = (x1, y1, x2, y2)
            
            # 重置标志
            roi_selecting = False
            # 不再销毁窗口，保持窗口打开状态
            return roi_rect
        
        # 按Esc键取消选择
        elif key == 27: 
            roi_selecting = False
            cv2.destroyWindow(window_name)
            return None

def draw_status_ui(frame, status, fps, obj_count):
    global ui_position
    
    # 如果UI位置未设置，使用默认位置
    if ui_position is None:
        height, width = frame.shape[:2]
        ui_x = 20
        ui_y = 20
        ui_width = 400
        ui_height = 60
    else:
        ui_x, ui_y, ui_width, ui_height = ui_position
    
    # 绘制半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (ui_x, ui_y), (ui_x + ui_width, ui_y + ui_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # 绘制边框
    cv2.rectangle(frame, (ui_x, ui_y), (ui_x + ui_width, ui_y + ui_height), (255, 255, 255), 1)
    
    # 绘制垂直分隔线
    line_x = ui_x + 130
    cv2.line(frame, (line_x, ui_y), (line_x, ui_y + ui_height), (255, 255, 255), 1)
    
    if status == "Warning":
        status_color = (0, 0, 255)  # 红色表示Warning
    else:
        status_color = (0, 255, 0)  # 绿色表示Safe
    
    cv2.putText(frame, status, (ui_x + 30, ui_y + 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
    
    # 绘制FPS和目标计数
    cv2.putText(frame, f"FPS: {fps:.1f}", (line_x + 30, ui_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(frame, f"target: {obj_count}", (line_x + 30, ui_y + 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 如果是Warning状态，添加闪烁Warning指示器
    if status == "Warning" and int(time.time() * 2) % 2 == 0:
        indicator_x = ui_x + ui_width - 30
        indicator_y = ui_y + ui_height // 2
        cv2.circle(frame, (indicator_x, indicator_y), 10, (0, 0, 255), -1)

def process_frame(frame, model, roi=None, alarm_cooldown=0, device='cpu'):
    """对单帧进行处理，返回标注后的帧和是否发现目标的标志"""
    global detection_status, current_obj_count, last_frame_time, current_fps
    
    # 计算帧率
    current_time = time.time()
    if last_frame_time == 0:
        last_frame_time = current_time
        current_fps = 0
    else:
        time_diff = current_time - last_frame_time
        if time_diff > 0:
            current_fps = 0.9 * current_fps + 0.1 * (1.0 / time_diff)  # 平滑FPS计算
        last_frame_time = current_time
    
    # 原始帧的副本，用于显示
    display_frame = frame.copy()
    
    # 如果有ROI，绘制ROI矩形并只处理ROI区域内的图像
    if roi:
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        
        # 在显示帧上绘制ROI矩形
        cv2.rectangle(display_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
        
        # 从原始帧中提取ROI区域
        roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # 使用模型仅对ROI区域进行预测，禁止打印详细信息
        # results = model(roi_image, verbose=False)
        results = model(roi_image, verbose=False, device=device)
        alarm = False
        obj_count = 0
        
        # 在ROI上绘制检测结果，并调整坐标以匹配原始帧
        for result in results:
            boxes = result.boxes
            
            # 遍历所有检测框
            for box in boxes:
                # 获取坐标 (相对于ROI区域)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 调整坐标到原始帧
                x1 += roi_x1
                y1 += roi_y1
                x2 += roi_x1
                y2 += roi_y1

                confidence = float(box.conf[0])
                
                color = (0, 0, 255)
                obj_count += 1
                alarm = True
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2) 
                conf_text = f"{confidence:.2f}"
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(display_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                cv2.putText(display_frame, conf_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # 如果没有ROI，对整个帧进行处理
        # results = model(frame, verbose=False)
        results = model(frame, verbose=False, device=device)
        alarm = False
        obj_count = 0
        
        # 在原始帧上绘制检测结果
        for result in results:
            boxes = result.boxes
            
            # 遍历所有检测框
            for box in boxes:
                # 获取坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                confidence = float(box.conf[0])
                
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                obj_count += 1
                alarm = True
            
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                conf_text = f"{confidence:.2f}"
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(display_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                cv2.putText(display_frame, conf_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 更新系统状态
    current_obj_count = obj_count
    
    # 根据当前检测结果更新状态
    if obj_count > 0:
        detection_status = "Warning"
    else:
        detection_status = "Safe"
    
    # 绘制UI
    draw_status_ui(display_frame, detection_status, current_fps, obj_count)
    
    # 如果发现目标且不在冷却期间，返回时间戳用于警报
    if alarm and current_time > alarm_cooldown:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return display_frame, True, timestamp
    
    return display_frame, False, None

def process_image(model, image_path, window_width=1280, window_height=720, save_path=None, device='cpu'):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 直接使用最终窗口名称和大小来选择ROI
    roi = get_roi(image, window_name='Detection Result', window_width=window_width, window_height=window_height)
    if roi is None:
        print("未选择ROI，将在整个图像上进行检测")
    else:
        print(f"已选择ROI: {roi}")
    
    # 处理图像
    result_image, alarm, _ = process_frame(image, model, roi, device=device)
    
    # 窗口已经在get_roi中创建并设置好大小，这里直接显示
    cv2.imshow('Detection Result', result_image)

    if alarm:
        print("Warning! 检测到区域内有目标!")
    
    # 如果指定了保存路径
    if save_path:
        cv2.imwrite(save_path, result_image)
        print(f"结果已保存到: {save_path}")
    
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(model, video_path, window_width=1280, window_height=720, save_path=None, conf_threshold=0.25, alert_save_dir=None, device='cpu'):
    # 如果是0-9的数字，转换为整数以使用摄像头
    if video_path.isdigit():
        video_path = int(video_path)
        print(f"使用摄像头 ID: {video_path}")
    else:
        print(f"处理视频文件: {video_path}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频源: {video_path}")
        return
    
    # 获取视频参数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 读取第一帧用于ROI选择
    ret, first_frame = cap.read()
    if not ret:
        print("无法读取第一帧")
        return
    
    # 直接使用最终窗口名称和大小来选择ROI
    roi = get_roi(first_frame, window_name='Video Detection', window_width=window_width, window_height=window_height)
    if roi is None:
        print("未选择ROI，将在整个图像上进行检测")
    else:
        print(f"已选择ROI: {roi}")
    
    # 重置视频
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 创建视频写入器
    video_writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用 'XVID'
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
    
    # 创建警报保存目录
    if alert_save_dir and not os.path.exists(alert_save_dir):
        os.makedirs(alert_save_dir)
    
    # 窗口已经在get_roi中创建并设置好大小，不需要再次创建
    
    frame_count = 0
    alarm_cooldown = 0  # 警报冷却时间（秒）
    alarm_interval = 3  # 警报间隔（秒）
    
    print("正在使用GPU进行推理，速度提升")

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            # 如果是文件末尾就退出，如果是摄像头就继续
            if isinstance(video_path, int):
                continue
            else:
                break
        
        frame_count += 1
        current_time = time.time()
        
        # 处理帧
        result_frame, alarm, timestamp = process_frame(frame, model, roi, alarm_cooldown, device)
        
        # 如果触发警报且不在冷却期间
        if alarm and current_time > alarm_cooldown:
            print("\a")  # 使用系统蜂鸣声
            print(f"Warning! 在帧 {frame_count} 检测到区域内有目标!")
            alarm_cooldown = current_time + alarm_interval
            
            # 保存警报截图
            if alert_save_dir:
                alert_path = os.path.join(alert_save_dir, f"alert_{timestamp}.jpg")
                cv2.imwrite(alert_path, result_frame)
                print(f"警报截图已保存到: {alert_path}")
        
        # 显示帧数和时间
        cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Video Detection', result_frame)
        
        # 如果需要保存
        if video_writer:
            video_writer.write(result_frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 清理
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    if save_path and video_writer:
        print(f"视频已保存到: {save_path}")

def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"加载配置文件出错: {e}")
        return None

def main(config_path='config.yaml'):
    config = load_config(config_path)
    if not config:
        print("无法加载配置，程序退出")
        return
    
    # 从配置中获取参数
    model_path = config.get('model', 'weights/yolov8s/best.onnx')
    source = config.get('source')
    window_width = config.get('width', 1280)
    window_height = config.get('height', 720)
    save_path = config.get('save')
    conf_threshold = config.get('conf', 0.25)
    alert_save_dir = config.get('alert_save_dir', 'alerts')  # 新增警报截图保存目录
    
    # 检查必要参数
    if not source:
        print("错误: 必须在配置文件中指定'source'参数")
        return
    
    # 禁止YOLO输出检测日志
    import os
    os.environ["YOLO_VERBOSE"] = "False"
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path, task='detect')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    # 设置模型推理参数，禁止打印详细信息
    model.predict(verbose=False)
    
    # 判断是图像还是视频
    if source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        process_image(model, source, window_width, window_height, save_path, device)
    else:
        # 设置保存路径
        if save_path and not save_path.lower().endswith('.mp4'):
            save_path = os.path.splitext(save_path)[0] + '.mp4'
        
        process_video(model, source, window_width, window_height, save_path, conf_threshold, alert_save_dir, device)

if __name__ == "__main__":
    import sys
    
    # 如果提供了配置文件路径，使用提供的路径
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        main(config_file)
    else:
        # 否则使用默认的config.yaml
        main()