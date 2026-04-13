from multiprocessing import freeze_support
from pathlib import Path

import onnx
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_CONFIG_PATH = PROJECT_DIR / "himars.yaml"
BASE_MODEL_PATH = "yolo26s.pt"
REQUIRED_OUTPUT_NAMES = ("logits", "pred_boxes")

# 没有 cuda 环境则使用： batch=4,device="cpu",workers=0
def main() -> None:
    model = YOLO(BASE_MODEL_PATH)

    model.train(
        data=str(DATASET_CONFIG_PATH),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        cache=False,
        amp=True,
        project=str(PROJECT_DIR / "runs"),
    )

    exported_model_path = model.export(format="onnx", simplify=True)

if __name__ == "__main__":
    freeze_support()
    main()
