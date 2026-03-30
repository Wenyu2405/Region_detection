import threading
import time
import cv2
import numpy as np
import os
import yaml
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QTextCursor
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QTextEdit, QGridLayout

from ultralytics import YOLO
import torch

import sys

# 告诉Qt使用系统插件，而不是OpenCV中的插件
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
# 确保使用xcb平台
os.environ["QT_QPA_PLATFORM"] = "xcb"
# 禁止YOLO输出检测日志 - 添加GPU加速优化
os.environ["YOLO_VERBOSE"] = "False"

# 导入yaml配置
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

try:
    from dahua.dahua_camera import DahuaCamera
    DAHUA_AVAILABLE = True
except ImportError:
    print("提示:不选用大华相机模块")
    DAHUA_AVAILABLE = False

try:
    from hik.hik_camera import call_back_get_image, start_grab_and_get_data_size, close_and_destroy_device, set_Value, get_Value, image_control
    HIK_AVAILABLE = True
except ImportError:
    print("提示：不选用海康相机模块")
    HIK_AVAILABLE = False

# 全局变量
camera_image = None
first_frame = None  # 存储第一帧用于ROI选择
yolo_model = None
detection_status = "Safe"
current_obj_count = 0
last_frame_time = 0
current_fps = 0
roi_rect = None
detection_enabled = False
video_playing = False  # 视频播放状态
alarm_cooldown = 0
alarm_interval = 3

# 视频捕获对象
video_capture = None
source_type = None
source_path = None

# ROI选择相关变量
roi_selecting = False
roi_complete = False
roi_start_point = None
roi_end_point = None
drawing = False

def determine_source_type(source):
    """根据source参数判断输入源类型"""
    if source is None:
        return 'none'
    
    source_str = str(source).lower()
    
    # 检查是否是数字（摄像头ID）
    if source_str.isdigit():
        return 'camera'
    
    # 检查是否是图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    if source_str.endswith(image_extensions):
        return 'image'
    
    # 检查是否是视频文件
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    if source_str.endswith(video_extensions):
        return 'video'
    
    # 默认尝试作为视频处理
    return 'video'

def initialize_video_source(source):
    """初始化视频源并获取第一帧"""
    global video_capture, first_frame
    
    # 如果source是数字字符串，转换为整数
    if str(source).isdigit():
        source = int(source)
        print(f"初始化摄像头 ID: {source}")
    else:
        print(f"初始化视频文件: {source}")
    
    video_capture = cv2.VideoCapture(source)
    
    if not video_capture.isOpened():
        print(f"无法打开视频源: {source}")
        return False
    
    # 读取第一帧
    ret, frame = video_capture.read()
    if ret:
        first_frame = frame.copy()
        print("已获取视频第一帧")
        return True
    else:
        print("无法读取视频第一帧")
        return False

def video_capture_thread():
    """视频捕获线程 - 优化版本"""
    global camera_image, video_capture, video_playing
    
    if video_capture is None:
        return
    
    while video_playing:
        ret, frame = video_capture.read()
        if ret:
            camera_image = frame
            # 减少延迟，提高帧率 - 从0.033改为0.01
            time.sleep(0.01)  # 约100fps上限，让GPU处理成为瓶颈而不是捕获
        else:
            # 如果是视频文件结束，停止播放
            if not str(source_path).isdigit():
                print("视频播放完毕，自动停止")
                video_playing = False
                break
            else:
                print("摄像头读取失败")
                time.sleep(0.1)

def load_static_image(image_path):
    """加载静态图像"""
    global first_frame
    
    img = cv2.imread(image_path)
    if img is not None:
        first_frame = img
        print(f"已加载静态图像: {image_path}")
        return True
    else:
        print(f"无法加载图像: {image_path}")
        return False

def draw_status_ui(frame, status, fps, obj_count):
    """在画面上绘制状态UI"""
    height, width = frame.shape[:2]
    ui_x = 20
    ui_y = 20
    ui_width = 400
    ui_height = 60
    
    # 绘制半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (ui_x, ui_y), (ui_x + ui_width, ui_y + ui_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # 绘制边框
    cv2.rectangle(frame, (ui_x, ui_y), (ui_x + ui_width, ui_y + ui_height), (255, 255, 255), 1)
    
    # 绘制垂直分隔线
    line_x = ui_x + 130
    cv2.line(frame, (line_x, ui_y), (line_x, ui_y + ui_height), (255, 255, 255), 1)
    
    # 绘制状态文本
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

def process_frame(frame, model, roi=None, device='cpu', conf_threshold=0.25):
    """对单帧进行处理，返回标注后的帧和是否发现目标的标志 - GPU优化版本"""
    global detection_status, current_obj_count, last_frame_time, current_fps, alarm_cooldown
    
    # 计算帧率
    current_time = time.time()
    if last_frame_time == 0:
        last_frame_time = current_time
        current_fps = 0
    else:
        time_diff = current_time - last_frame_time
        if time_diff > 0:
            current_fps = 0.9 * current_fps + 0.1 * (1.0 / time_diff)
        last_frame_time = current_time
    
    display_frame = frame.copy()
    
    # 如果有ROI，绘制ROI矩形并只处理ROI区域内的图像
    if roi:
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        
        # 在显示帧上绘制ROI矩形
        cv2.rectangle(display_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
        
        # 从原始帧中提取ROI区域
        roi_image = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # 使用模型仅对ROI区域进行预测 - GPU加速优化
        results = model(roi_image, verbose=False, device=device, conf=conf_threshold)
        alarm = False
        obj_count = 0
        
        # 在ROI上绘制检测结果，并调整坐标以匹配原始帧
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
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
        # 如果没有ROI，对整个帧进行处理 - GPU加速优化
        results = model(frame, verbose=False, device=device, conf=conf_threshold)
        alarm = False
        obj_count = 0
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
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
    
    if obj_count > 0:
        detection_status = "Warning"
    else:
        detection_status = "Safe"
    
    # 绘制UI
    draw_status_ui(display_frame, detection_status, current_fps, obj_count)
    
    # 如果发现目标且不在冷却期间，返回时间戳用于警报
    if alarm and current_time > alarm_cooldown:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alarm_cooldown = current_time + alarm_interval
        return display_frame, True, timestamp
    
    return display_frame, False, None

class YOLOUI(QWidget):
    def __init__(self):
        super().__init__()
        self.capturing = False  # 默认不开始捕获
        self.detection_enabled = False
        self.roi_selecting = False
        self.roi_drawing = False
        self.roi_start = None
        self.roi_current = None
        # GPU优化 - 明确设备选择和加速提示
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alert_save_dir = config.get('alert_save_dir', 'alerts')
        self.conf_threshold = config.get('conf', 0.25)
        
        # 创建警报保存目录
        if not os.path.exists(self.alert_save_dir):
            os.makedirs(self.alert_save_dir)
            
        self.initUI()
        self.init_yolo_model()

    def initUI(self):
        # 图像显示区域
        self.image_label = QLabel(self)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 2px solid black;")
        self.image_label.mousePressEvent = self.image_mouse_press
        self.image_label.mouseMoveEvent = self.image_mouse_move
        self.image_label.mouseReleaseEvent = self.image_mouse_release
        
        # 显示初始提示
        self.show_initial_message()
        
        # 文本框
        self.text_area = QTextEdit(self)
        self.text_area.setFixedSize(300, 200)
        
        # 按钮
        self.btn_roi = QPushButton('选择检测区域', self)
        self.btn_roi.setFixedSize(300, 70)
        self.btn_roi.clicked.connect(self.select_roi)
        
        self.btn_detection = QPushButton('开始检测', self)
        self.btn_detection.setFixedSize(300, 70)
        self.btn_detection.clicked.connect(self.toggle_detection)
        self.btn_detection.setEnabled(False)  # 初始禁用
        
        self.btn_clear_roi = QPushButton('清除ROI', self)
        self.btn_clear_roi.setFixedSize(300, 70)
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        
        self.btn_save_image = QPushButton('保存当前图像', self)
        self.btn_save_image.setFixedSize(300, 70)
        self.btn_save_image.clicked.connect(self.save_current_image)
        
        # 设置按钮样式
        for btn in [self.btn_roi, self.btn_detection, self.btn_clear_roi, self.btn_save_image]:
            btn.setStyleSheet("QPushButton { font-size: 14px; }")
        
        # 按钮布局
        button_layout = QGridLayout()
        button_layout.addWidget(self.btn_roi, 0, 0)
        button_layout.addWidget(self.btn_detection, 0, 1)
        button_layout.addWidget(self.btn_clear_roi, 1, 0)
        button_layout.addWidget(self.btn_save_image, 1, 1)
        
        # 底部控制区域
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.addLayout(button_layout)
        bottom_layout.addWidget(self.text_area)
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(bottom_widget)
        
        self.setLayout(main_layout)
        self.setGeometry(100, 100, 1200, 900)
        self.setWindowTitle('YOLO目标检测系统 - GPU加速版')
        self.setMinimumSize(800, 700)
        
        # 相机定时器 - 优化更新频率
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera)
        # 不自动启动定时器
        
        self.show()

    def show_initial_message(self):
        """显示初始提示信息"""
        # 创建一个空白图像显示提示
        blank_image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(blank_image, 'Click "Select Detection Area"', (150, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(blank_image, 'to start', (250, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        rgb_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
        self.display_image(rgb_image)

    def init_yolo_model(self):
        """初始化YOLO模型 - GPU优化版本"""
        global yolo_model
        try:
            model_path = config.get('model', 'weights/yolov8s/best.onnx')
            yolo_model = YOLO(model_path, task='detect')
            
            # GPU加速优化 - 设置模型推理参数
            yolo_model.predict(verbose=False)
            
            self.append_text(f"YOLO模型已加载: {model_path}")
            self.append_text(f"使用设备: {self.device}")
            
            # GPU加速提示
            if self.device == 'cuda':
                self.append_text("正在使用GPU进行推理，速度提升")
            else:
                self.append_text("正在使用CPU进行推理")
                
            self.append_text(f"置信度阈值: {self.conf_threshold}")
            self.append_text("请先选择检测区域")
        except Exception as e:
            self.append_text(f"模型加载失败: {str(e)}")

    def update_camera(self):
        """更新相机画面 - GPU优化版本"""
        global camera_image, yolo_model, roi_rect, detection_enabled, video_playing, first_frame
        
        # 检查视频是否已停止播放（用于视频文件播放完毕的情况）
        if self.detection_enabled and not video_playing and source_type == 'video':
            self.stop_detection()
            self.append_text('视频播放完毕，检测已自动停止')
            return
        
        # 如果正在选择ROI，显示第一帧
        if self.roi_selecting and first_frame is not None:
            img = first_frame.copy()
            
            # 如果正在绘制ROI，添加选择框
            if self.roi_drawing and self.roi_start and self.roi_current:
                x1 = int(self.roi_start[0] * self.scale_x)
                y1 = int(self.roi_start[1] * self.scale_y)
                x2 = int(self.roi_current[0] * self.scale_x)
                y2 = int(self.roi_current[1] * self.scale_y)
                
                h, w = img.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 如果已经选择了ROI，显示ROI框
            if roi_rect:
                roi_x1, roi_y1, roi_x2, roi_y2 = roi_rect
                cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
            
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.display_image(rgb_image)
            return
        
        # 如果视频正在播放且有图像数据
        if video_playing and camera_image is not None and self.capturing:
            img = camera_image.copy()
            
            # 如果启用检测且模型已加载 - GPU加速处理
            if self.detection_enabled and yolo_model is not None:
                processed_frame, alarm, timestamp = process_frame(img, yolo_model, roi_rect, self.device, self.conf_threshold)
                
                # 如果检测到目标
                if alarm:
                    self.append_text(f"警告！检测到目标 - {timestamp}")
                    # 保存警报截图
                    alert_path = os.path.join(self.alert_save_dir, f"alert_{timestamp}.jpg")
                    cv2.imwrite(alert_path, processed_frame)
                    self.append_text(f"警报截图已保存: {alert_path}")
                
                img = processed_frame
            else:
                # 如果有ROI但未启用检测，只显示ROI框
                if roi_rect:
                    roi_x1, roi_y1, roi_x2, roi_y2 = roi_rect
                    cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
            
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.display_image(rgb_image)

    def display_image(self, rgb_image):
        """显示图像到界面"""
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # 获取标签尺寸并调整图像
        label_size = self.image_label.size()
        label_w, label_h = label_size.width(), label_size.height()
        
        # 计算缩放比例
        scale_w = label_w / w
        scale_h = label_h / h
        scale = min(scale_w, scale_h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized_image = cv2.resize(rgb_image, (new_w, new_h))
        
        # 保存缩放比例用于坐标转换
        self.scale_x = w / new_w
        self.scale_y = h / new_h
        self.original_size = (w, h)
        
        qt_image = QImage(resized_image.data, new_w, new_h, new_w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

    def image_mouse_press(self, event):
        """处理鼠标按下事件"""
        if self.roi_selecting and event.button() == Qt.LeftButton:
            self.roi_drawing = True
            self.roi_start = (event.pos().x(), event.pos().y())
            self.roi_current = self.roi_start

    def image_mouse_move(self, event):
        """处理鼠标移动事件"""
        if self.roi_selecting and self.roi_drawing:
            self.roi_current = (event.pos().x(), event.pos().y())

    def image_mouse_release(self, event):
        """处理鼠标释放事件"""
        global roi_rect
        
        if self.roi_selecting and self.roi_drawing and event.button() == Qt.LeftButton:
            self.roi_drawing = False
            
            if self.roi_start and self.roi_current:
                # 转换界面坐标到图像坐标
                x1 = int(self.roi_start[0] * self.scale_x)
                y1 = int(self.roi_start[1] * self.scale_y)
                x2 = int(self.roi_current[0] * self.scale_x)
                y2 = int(self.roi_current[1] * self.scale_y)
                
                # 确保坐标顺序正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # 确保坐标在图像范围内
                h, w = self.original_size[1], self.original_size[0]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                # 检查ROI区域是否足够大
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    roi_rect = (x1, y1, x2, y2)
                    self.append_text(f"ROI区域已选择: {roi_rect}")
                    
                    # 完成ROI选择，退出选择模式
                    self.roi_selecting = False
                    self.roi_start = None
                    self.roi_current = None
                    self.btn_roi.setText('重新选择区域')
                    self.btn_roi.setStyleSheet("QPushButton { font-size: 14px; }")
                    
                    # 启用开始检测按钮
                    self.btn_detection.setEnabled(True)
                    self.btn_detection.setStyleSheet("QPushButton { font-size: 14px; background-color: #66ff66; }")
                    
                    self.append_text('ROI选择完成，现在可以开始检测')
                else:
                    self.append_text('ROI区域太小，请重新选择')

    def select_roi(self):
        """选择ROI区域"""
        global first_frame, source_type, source_path
        
        if not self.roi_selecting:
            # 如果还没有第一帧，先初始化视频源
            if first_frame is None:
                if source_type == 'image':
                    success = load_static_image(source_path)
                elif source_type in ['video', 'camera']:
                    success = initialize_video_source(source_path)
                else:
                    success = False
                
                if not success:
                    self.append_text('错误: 无法初始化视频源')
                    return
            
            self.roi_selecting = True
            self.roi_drawing = False
            self.roi_start = None
            self.roi_current = None
            self.btn_roi.setText('拖拽选择区域...')
            self.btn_roi.setStyleSheet("QPushButton { font-size: 14px; background-color: #ffff66; }")
            self.append_text('请在图像上拖拽选择检测区域')
            
            # 启动定时器显示第一帧
            if not self.camera_timer.isActive():
                self.camera_timer.start()
        else:
            # 取消ROI选择
            self.roi_selecting = False
            self.roi_drawing = False
            self.roi_start = None
            self.roi_current = None
            self.btn_roi.setText('选择检测区域')
            self.btn_roi.setStyleSheet("QPushButton { font-size: 14px; }")
            self.append_text('已取消ROI选择')

    def toggle_detection(self):
        """开启/关闭检测（同时控制视频播放） - GPU优化版本"""
        global detection_enabled, video_playing
        
        if yolo_model is None:
            self.append_text('错误: YOLO模型未加载')
            return
        
        if roi_rect is None:
            self.append_text('错误: 请先选择检测区域')
            return
            
        self.detection_enabled = not self.detection_enabled
        detection_enabled = self.detection_enabled
        
        if self.detection_enabled:
            # 开始检测和视频播放
            video_playing = True
            self.capturing = True
            
            # 启动视频捕获线程
            if source_type in ['video', 'camera']:
                self.video_thread = threading.Thread(target=video_capture_thread, daemon=True)
                self.video_thread.start()
            
            self.btn_detection.setText('停止检测')
            self.btn_detection.setStyleSheet("QPushButton { font-size: 14px; background-color: #ff6666; }")
            
            # GPU加速提示
            if self.device == 'cuda':
                self.append_text('检测已开启，视频开始播放 - GPU加速模式')
            else:
                self.append_text('检测已开启，视频开始播放 - CPU模式')
            
            # 启动更新定时器 - 优化刷新率
            if not self.camera_timer.isActive():
                self.camera_timer.start(16)  # 约60fps - 从50ms改为16ms
        else:
            self.stop_detection()

    def stop_detection(self):
        """停止检测的统一方法"""
        global video_playing, detection_enabled
        
        # 停止检测和视频播放
        video_playing = False
        self.capturing = False
        self.detection_enabled = False
        detection_enabled = False
        
        self.btn_detection.setText('开始检测')
        self.btn_detection.setStyleSheet("QPushButton { font-size: 14px; background-color: #66ff66; }")
        self.append_text('检测已停止，视频已暂停')

    def clear_roi(self):
        """清除ROI区域"""
        global roi_rect
        roi_rect = None
        self.roi_selecting = False
        self.roi_drawing = False
        self.roi_start = None
        self.roi_current = None
        self.btn_roi.setText('选择检测区域')
        self.btn_roi.setStyleSheet("QPushButton { font-size: 14px; }")
        
        # 禁用开始检测按钮
        self.btn_detection.setEnabled(False)
        self.btn_detection.setStyleSheet("QPushButton { font-size: 14px; }")
        
        self.append_text('ROI区域已清除，请重新选择')
        
        # 显示初始提示
        self.show_initial_message()

    def save_current_image(self):
        """保存当前图像"""
        global camera_image, first_frame
        
        # 优先保存当前播放的图像，如果没有则保存第一帧
        image_to_save = camera_image if camera_image is not None else first_frame
        
        if image_to_save is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"saved_image_{timestamp}.jpg"
            cv2.imwrite(save_path, image_to_save)
            self.append_text(f"图像已保存: {save_path}")
        else:
            self.append_text('错误: 无可用图像')

    def append_text(self, text):
        """在文本区域追加文本"""
        current_text = self.text_area.toPlainText()
        self.text_area.setPlainText(current_text + '\n' + text)
        # 自动滚动到底部
        cursor = self.text_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_area.setTextCursor(cursor)

    def keyPressEvent(self, event):
        """按键事件"""
        if event.key() == Qt.Key_Escape:
            if self.roi_selecting:
                # 如果正在选择ROI，ESC键取消选择
                self.roi_selecting = False
                self.roi_drawing = False
                self.roi_start = None
                self.roi_current = None
                self.btn_roi.setText('选择检测区域')
                self.btn_roi.setStyleSheet("QPushButton { font-size: 14px; }")
                self.append_text('已取消ROI选择')
            else:
                self.close()

    def closeEvent(self, event):
        """窗口关闭事件"""
        global video_playing
        video_playing = False
        if hasattr(self, 'video_thread'):
            self.video_thread.join(timeout=1.0)
        if video_capture:
            video_capture.release()
        event.accept()

if __name__ == '__main__':
    # 获取配置参数
    source = config.get('source')
    source_type = determine_source_type(source)
    source_path = source
    
    print(f"配置的输入源: {source}")
    print(f"检测到的源类型: {source_type}")
    
    # GPU检测和提示
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"检测到设备: {device}")
    if device == 'cuda':
        print("GPU加速已启用，推理性能将显著提升")
    else:
        print("使用CPU推理，建议安装CUDA以获得更好性能")
    
    # 不在启动时初始化视频源，等待用户选择ROI时再初始化
    if source_type == 'none':
        print("警告: 未配置输入源")
    
    # 启动UI
    app = QApplication(sys.argv)
    ui = YOLOUI()
    sys.exit(app.exec_())