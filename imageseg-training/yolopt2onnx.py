from ultralytics import YOLO
# 加载.pt模型
model = YOLO(r"C:\Users\phant\workspace\gitlink\4CImageSeg\4CImageSeg.Training\best.pt")
# 导出为ONNX格式
model.export(format="onnx", imgsz=640, simplify=True, dynamic=False)