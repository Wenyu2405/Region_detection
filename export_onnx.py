from ultralytics import YOLO

model = YOLO("/home/wenyu/final/weights/best.pt")

path = model.export(
    format="onnx",
    opset=12,        
    simplify=True,   # 简化ONNX图结构
    dynamic=False,   # 固定输入尺寸
)
print(f"模型已导出到: {path}")