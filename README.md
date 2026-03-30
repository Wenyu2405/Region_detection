# YOLO 可视化目标实时检测系统

基于 YOLOv8 的目标检测系统，支持图像、视频文件及摄像头实时检测，
提供图形界面（GUI）与纯命令行两种运行方式，并内置 ROI 区域选择与
告警截图保存功能。

## 项目结构

├── main.py           # GUI 版本（PyQt5），支持 GPU 加速

├── CPUversion.py     # 命令行版本，强制使用 CPU 推理

├── trans.py          # 模型训练入口

├── config.yaml       # 运行时配置文件

├── configs/

│   └── fusion.yaml   # 训练数据集配置

├── weights/

│   └── yolov8s/

│       └── best.onnx # 训练好的模型权重

└── alerts/           # 告警截图自动保存目录


## 环境依赖

Python >= 3.8

pip install ultralytics opencv-python numpy PyQt5 torch onnxruntime


## 配置说明

编辑 `config.yaml`：

model: weights/yolov8s/best.onnx  # 模型路径（支持 .pt 和 .onnx）
source: person_test1.mp4          # 输入源：视频路径 / 图片路径 / 摄像头ID（如 0）
width: 1280                        # 显示窗口宽度
height: 720                        # 显示窗口高度
save: null                         # 结果保存路径，不保存填 null
conf: 0.25                         # 置信度阈值
alert_save_dir: alerts             # 告警截图保存目录


## 使用方法

### 方式一：GUI 版本（推荐，支持 GPU）

python main.py

操作步骤：
1. 启动后点击「选择检测区域」，在画面上拖拽框选 ROI
2. ROI 选择完成后，点击「开始检测」
3. 检测到目标时自动保存告警截图到 alerts/ 目录
4. 点击「停止检测」或按 ESC 退出

### 方式二：命令行版本（纯 CPU，适合无 GPU 环境）

python CPUversion.py

操作步骤：
1. 弹出第一帧画面后，用鼠标拖拽选择 ROI，按 Enter 确认
2. 按 q 键退出检测

### 方式三：模型训练

python trans.py

训练参数在 `configs/fusion.yaml` 和 `trans.py` 中配置。


## 支持的输入源

| 类型     | 示例值                  |
|----------|-------------------------|
| 摄像头   | `0`（摄像头 ID）        |
| 视频文件 | `video.mp4`             |
| 图片文件 | `image.jpg`             |


## 告警机制

- 检测到目标时触发告警，默认**冷却时间 3 秒**（避免重复告警）
- 告警截图自动保存至 `alert_save_dir` 指定目录
- GUI 版本同时在日志区域显示告警信息


## 注意事项

- GPU 版本（`main.py`）需要安装支持 CUDA 的 PyTorch
- CPU 版本（`CPUversion.py`）在启动时通过环境变量完全禁用 CUDA
- 使用 ONNX 模型时，即使强制 CPU 模式，终端仍可能打印 CUDA
  相关提示信息，属于 ONNX Runtime 本身行为，不影响运行结果
- 若需彻底避免 CUDA 日志，建议改用 `.pt` 格式的 PyTorch 模型

很早的产物
