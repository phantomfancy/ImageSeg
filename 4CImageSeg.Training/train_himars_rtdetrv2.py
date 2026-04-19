from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import json
import numpy as np
import os
from multiprocessing import freeze_support
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_CONFIG_PATH = PROJECT_DIR / "data.yaml"
OUTPUT_DIR = PROJECT_DIR / "runs" / "rtdetrv2-equipment"
METADATA_CACHE_PATH = PROJECT_DIR / ".cache" / "train_equipment_metadata.json"
MODEL_NAME = "PekingU/rtdetr_v2_r18vd"
IMAGE_SIZE = 640
EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
TRAIN_BATCH_SIZE = 8 if torch.cuda.is_available() else 2
EVAL_BATCH_SIZE = 8 if torch.cuda.is_available() else 2
GRADIENT_ACCUMULATION_STEPS = 1
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2
SEED = 42
EXPORT_ONNX = True
EXPORT_ONNX_FP16 = True
EXPORT_ONNX_INT8 = True
OVERWRITE_OUTPUT = True
RESUME_FROM_CHECKPOINT: str | None = "auto"
GRADIENT_CHECKPOINTING = False
COMPILE_MODEL = False
COMPILE_BACKEND = "inductor"
ALLOW_TF32 = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUMMARY_FILE_NAME = "training-summary.json"
ONNX_FP32_DIR_NAME = "onnx-fp32"
ONNX_FP16_DIR_NAME = "onnx-fp16"
ONNX_INT8_DIR_NAME = "onnx-int8"
ONNX_FP32_FILE_NAME = "model.onnx"
ONNX_FP16_FILE_NAME = "model_fp16.onnx"
ONNX_INT8_FILE_NAME = "model_int8_static.onnx"
ONNX_INT8_DYNAMIC_FILE_NAME = "model_int8_dynamic.onnx"
INT8_CALIBRATION_IMAGE_LIMIT = 128


@dataclass(frozen=True)
class SplitSummary:
    name: str
    image_count: int
    labeled_image_count: int
    empty_image_count: int
    missing_label_file_count: int
    annotation_count: int
    invalid_annotation_count: int
    skipped_image_count: int


def detect_default_worker_count() -> int:
    if os.name == "nt":
        return 1

    cpu_count = os.cpu_count() or 2
    return min(4, max(1, cpu_count // 2))


DATALOADER_NUM_WORKERS = detect_default_worker_count()
DATALOADER_PREFETCH_FACTOR = 1


def resolve_dataset_path(config_path: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate

    config_parent_candidate = (config_path.parent / candidate).resolve()
    if config_parent_candidate.exists():
        return config_parent_candidate

    return (PROJECT_DIR / candidate).resolve()


def infer_labels_dir(dataset_root: Path, images_dir: Path) -> Path:
    relative_images_dir = images_dir.relative_to(dataset_root)
    parts = list(relative_images_dir.parts)
    if parts and parts[0].lower() == "images":
        parts[0] = "labels"
    else:
        parts.insert(0, "labels")
    return (dataset_root / Path(*parts)).resolve()


def sort_class_name_items(raw_names: dict[Any, Any]) -> list[str]:
    def key_selector(item: tuple[Any, Any]) -> tuple[int, str]:
        key = str(item[0])
        return (int(key), key) if key.isdigit() else (10**9, key)

    return [str(value) for _, value in sorted(raw_names.items(), key=key_selector)]


def load_dataset_config(config_path: Path) -> tuple[Path, Path, Path, Path, list[str]]:
    if not config_path.is_file():
        raise FileNotFoundError(f"数据集配置不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    required_keys = {"train", "val", "names"}
    missing_keys = sorted(required_keys.difference(raw))
    if missing_keys:
        raise ValueError(f"数据集配置缺少必要字段：{', '.join(missing_keys)}")

    raw_names = raw["names"]
    if isinstance(raw_names, dict):
        class_names = sort_class_name_items(raw_names)
    elif isinstance(raw_names, list):
        class_names = [str(name) for name in raw_names]
    else:
        raise TypeError("数据集配置中的 names 字段必须为映射或列表。")

    dataset_root = resolve_dataset_path(config_path, str(raw["path"]))
    train_images_dir = (dataset_root / str(raw["train"])).resolve()
    val_images_dir = (dataset_root / str(raw["val"])).resolve()
    train_labels_dir = infer_labels_dir(dataset_root, train_images_dir)
    val_labels_dir = infer_labels_dir(dataset_root, val_images_dir)

    for candidate in (train_images_dir, val_images_dir):
        if not candidate.is_dir():
            raise FileNotFoundError(f"图片目录不存在：{candidate}")

    for candidate in (train_labels_dir, val_labels_dir):
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)

    if not class_names:
        raise ValueError("数据集配置中没有可用类别。")

    return train_images_dir, val_images_dir, train_labels_dir, val_labels_dir, class_names


def load_metadata_cache(cache_path: Path) -> dict[str, dict[str, int]]:
    if not cache_path.is_file():
        return {}

    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, dict[str, int]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue

        if not {"mtime_ns", "size", "width", "height"}.issubset(value):
            continue

        normalized[key] = {
            "mtime_ns": int(value["mtime_ns"]),
            "size": int(value["size"]),
            "width": int(value["width"]),
            "height": int(value["height"]),
        }

    return normalized


def save_metadata_cache(cache_path: Path, metadata_cache: dict[str, dict[str, int]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata_cache, handle, ensure_ascii=False, indent=2, sort_keys=True)


def get_image_dimensions(image_path: Path, metadata_cache: dict[str, dict[str, int]]) -> tuple[int, int]:
    stat = image_path.stat()
    key = str(image_path.resolve())
    cached = metadata_cache.get(key)
    if cached is not None:
        if cached["mtime_ns"] == stat.st_mtime_ns and cached["size"] == stat.st_size:
            return cached["width"], cached["height"]

    with Image.open(image_path) as image:
        width, height = image.size

    metadata_cache[key] = {
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "width": width,
        "height": height,
    }
    return width, height


def clip_bbox(left: float, top: float, width: float, height: float, image_width: int, image_height: int) -> list[float] | None:
    right = left + width
    bottom = top + height

    clipped_left = min(max(left, 0.0), float(image_width))
    clipped_top = min(max(top, 0.0), float(image_height))
    clipped_right = min(max(right, 0.0), float(image_width))
    clipped_bottom = min(max(bottom, 0.0), float(image_height))

    clipped_width = clipped_right - clipped_left
    clipped_height = clipped_bottom - clipped_top
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    return [clipped_left, clipped_top, clipped_width, clipped_height]


def empty_objects() -> dict[str, list[Any]]:
    return {
        "id": [],
        "area": [],
        "bbox": [],
        "category": [],
    }


def load_yolo_annotations(
    label_path: Path,
    image_width: int,
    image_height: int,
    class_count: int,
) -> tuple[dict[str, list[Any]], int]:
    objects = empty_objects()
    invalid_annotation_count = 0

    if not label_path.exists():
        return objects, invalid_annotation_count

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) != 5:
                invalid_annotation_count += 1
                continue

            try:
                category_id = int(float(parts[0]))
                center_x, center_y, width, height = map(float, parts[1:])
            except ValueError:
                invalid_annotation_count += 1
                continue

            if category_id < 0 or category_id >= class_count or width <= 0 or height <= 0:
                invalid_annotation_count += 1
                continue

            box_width = width * image_width
            box_height = height * image_height
            left = (center_x * image_width) - (box_width / 2)
            top = (center_y * image_height) - (box_height / 2)
            bbox = clip_bbox(left, top, box_width, box_height, image_width, image_height)
            if bbox is None:
                invalid_annotation_count += 1
                continue

            objects["id"].append(len(objects["id"]))
            objects["area"].append(bbox[2] * bbox[3])
            objects["bbox"].append(bbox)
            objects["category"].append(category_id)

    return objects, invalid_annotation_count


def build_split_samples(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    class_count: int,
    metadata_cache: dict[str, dict[str, int]],
) -> tuple[list[dict[str, Any]], SplitSummary]:
    samples: list[dict[str, Any]] = []
    labeled_image_count = 0
    missing_label_file_count = 0
    annotation_count = 0
    invalid_annotation_count = 0
    skipped_image_count = 0

    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_id, image_path in enumerate(image_paths):
        try:
            width, height = get_image_dimensions(image_path, metadata_cache)
        except (OSError, UnidentifiedImageError):
            skipped_image_count += 1
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        objects, invalid_count = load_yolo_annotations(label_path, width, height, class_count)
        invalid_annotation_count += invalid_count

        if objects["id"]:
            labeled_image_count += 1

        if not label_path.exists():
            missing_label_file_count += 1

        annotation_count += len(objects["id"])
        samples.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "width": width,
                "height": height,
                "objects": objects,
            }
        )

    summary = SplitSummary(
        name=split_name,
        image_count=len(samples),
        labeled_image_count=labeled_image_count,
        empty_image_count=len(samples) - labeled_image_count,
        missing_label_file_count=missing_label_file_count,
        annotation_count=annotation_count,
        invalid_annotation_count=invalid_annotation_count,
        skipped_image_count=skipped_image_count,
    )
    return samples, summary


def format_annotations_as_coco(sample: dict[str, Any]) -> dict[str, Any]:
    annotations: list[dict[str, Any]] = []

    for category_id, area, bbox in zip(
        sample["objects"]["category"],
        sample["objects"]["area"],
        sample["objects"]["bbox"],
    ):
        annotations.append(
            {
                "image_id": sample["image_id"],
                "category_id": int(category_id),
                "iscrowd": 0,
                "area": float(area),
                "bbox": [float(value) for value in bbox],
            }
        )

    return {
        "image_id": sample["image_id"],
        "annotations": annotations,
    }


class YoloDetectionDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        with Image.open(sample["image_path"]) as image:
            image = np.asarray(image.convert("RGB"), dtype=np.uint8)

        return {
            "image": image,
            "annotations": format_annotations_as_coco(sample),
        }


class DetectionBatchCollator:
    def __init__(self, image_processor: RTDetrImageProcessor) -> None:
        self.image_processor = image_processor

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = [example["image"] for example in examples]
        annotations = [example["annotations"] for example in examples]
        encoded = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        batch = {key: value for key, value in encoded.items() if key != "labels"}
        batch["labels"] = encoded["labels"]
        return batch


def print_split_summary(summary: SplitSummary) -> None:
    print(
        f"[{summary.name}] "
        f"images={summary.image_count}, "
        f"labeled={summary.labeled_image_count}, "
        f"empty={summary.empty_image_count}, "
        f"missing_labels={summary.missing_label_file_count}, "
        f"annotations={summary.annotation_count}, "
        f"invalid_annotations={summary.invalid_annotation_count}, "
        f"skipped_images={summary.skipped_image_count}"
    )


def configure_torch_runtime() -> dict[str, Any]:
    has_cuda = torch.cuda.is_available()
    has_bf16 = has_cuda and torch.cuda.is_bf16_supported()

    if ALLOW_TF32 and has_cuda:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

    return {
        "device": "cuda" if has_cuda else "cpu",
        "cuda_available": has_cuda,
        "bf16_available": has_bf16,
        "fp16_enabled": has_cuda and not has_bf16,
        "bf16_enabled": has_bf16,
    }


def build_training_arguments(runtime: dict[str, Any]) -> TrainingArguments:
    supported_parameters = inspect.signature(TrainingArguments.__init__).parameters
    common_kwargs: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR),
        "overwrite_output_dir": OVERWRITE_OUTPUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": MAX_GRAD_NORM,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": EPOCHS,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": LOGGING_STEPS,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "remove_unused_columns": False,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "dataloader_pin_memory": runtime["cuda_available"],
        "dataloader_persistent_workers": DATALOADER_NUM_WORKERS > 0,
        "fp16": runtime["fp16_enabled"],
        "bf16": runtime["bf16_enabled"],
        "prediction_loss_only": True,
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
        "save_safetensors": True,
    }

    if runtime["cuda_available"] and "optim" in supported_parameters:
        common_kwargs["optim"] = "adamw_torch_fused"

    if DATALOADER_NUM_WORKERS > 0 and "dataloader_prefetch_factor" in supported_parameters:
        common_kwargs["dataloader_prefetch_factor"] = DATALOADER_PREFETCH_FACTOR

    if "eval_strategy" in supported_parameters:
        common_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in supported_parameters:
        common_kwargs["evaluation_strategy"] = "epoch"

    filtered_kwargs = {
        key: value
        for key, value in common_kwargs.items()
        if key in supported_parameters
    }
    return TrainingArguments(**filtered_kwargs)


def build_trainer(
    model: RTDetrV2ForObjectDetection,
    training_args: TrainingArguments,
    image_processor: RTDetrImageProcessor,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Trainer:
    supported_parameters = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "data_collator": DetectionBatchCollator(image_processor),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }

    if "processing_class" in supported_parameters:
        trainer_kwargs["processing_class"] = image_processor
    elif "tokenizer" in supported_parameters:
        trainer_kwargs["tokenizer"] = image_processor

    return Trainer(**trainer_kwargs)


def resolve_resume_checkpoint() -> str | None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if RESUME_FROM_CHECKPOINT is None:
        return None

    if RESUME_FROM_CHECKPOINT == "auto":
        last_checkpoint = get_last_checkpoint(str(OUTPUT_DIR))
        if last_checkpoint is not None:
            print(f"检测到可恢复 checkpoint：{last_checkpoint}")
            return last_checkpoint

        if any(OUTPUT_DIR.iterdir()) and not OVERWRITE_OUTPUT:
            raise RuntimeError(
                "输出目录非空且未发现 checkpoint。"
                "为避免覆盖现有产物，请修改 OUTPUT_DIR 或显式开启 OVERWRITE_OUTPUT。"
            )
        return None

    checkpoint_path = Path(RESUME_FROM_CHECKPOINT).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"指定的 checkpoint 不存在：{checkpoint_path}")
    return str(checkpoint_path)


def write_training_summary(
    training_settings: dict[str, Any],
    class_names: list[str],
    split_summaries: list[SplitSummary],
    runtime: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": training_settings,
        "class_names": class_names,
        "runtime": runtime,
        "splits": [summary.__dict__ for summary in split_summaries],
    }

    with (output_dir / SUMMARY_FILE_NAME).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


class RTDetrCalibrationDataReader:
    def __init__(self, image_processor: RTDetrImageProcessor, image_paths: list[Path]) -> None:
        self.image_processor = image_processor
        self.image_paths = image_paths
        self.index = 0

    def get_next(self) -> dict[str, Any] | None:
        if self.index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            encoded = self.image_processor(images=image, return_tensors="np")

        return {"pixel_values": encoded["pixel_values"]}

    def rewind(self) -> None:
        self.index = 0


def find_exported_onnx_file(output_dir: Path, preferred_file_name: str) -> Path:
    preferred_path = output_dir / preferred_file_name
    if preferred_path.is_file():
        return preferred_path

    candidates = sorted(output_dir.glob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"未在导出目录中找到 ONNX 文件：{output_dir}")
    return candidates[0]


def copy_onnx_metadata_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in ("config.json", "preprocessor_config.json"):
        source_path = source_dir / file_name
        if source_path.is_file():
            shutil.copy2(source_path, target_dir / file_name)


class WorkspaceTemporaryDirectory:
    def __init__(
        self,
        base_dir: Path,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
        ignore_cleanup_errors: bool = False,
    ) -> None:
        del ignore_cleanup_errors
        target_base_dir = Path(dir) if dir is not None else base_dir
        name = f"{prefix or 'tmp.'}{uuid4().hex}{suffix or ''}"
        self.path = target_base_dir / name
        self.name = str(self.path)

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.name

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


@contextmanager
def patched_temp_workspace(base_dir: Path) -> Any:
    import tempfile

    base_dir.mkdir(parents=True, exist_ok=True)

    old_temp_dir_class = tempfile.TemporaryDirectory
    old_tempdir = tempfile.tempdir
    old_temp = os.environ.get("TEMP")
    old_tmp = os.environ.get("TMP")

    class QuantTemporaryDirectory(WorkspaceTemporaryDirectory):
        def __init__(
            self,
            suffix: str | None = None,
            prefix: str | None = None,
            dir: str | None = None,
            ignore_cleanup_errors: bool = False,
        ) -> None:
            super().__init__(
                base_dir=base_dir,
                suffix=suffix,
                prefix=prefix,
                dir=dir,
                ignore_cleanup_errors=ignore_cleanup_errors,
            )

    try:
        tempfile.TemporaryDirectory = QuantTemporaryDirectory
        tempfile.tempdir = str(base_dir)
        os.environ["TEMP"] = str(base_dir)
        os.environ["TMP"] = str(base_dir)
        yield
    finally:
        tempfile.TemporaryDirectory = old_temp_dir_class
        tempfile.tempdir = old_tempdir

        if old_temp is None:
            os.environ.pop("TEMP", None)
        else:
            os.environ["TEMP"] = old_temp

        if old_tmp is None:
            os.environ.pop("TMP", None)
        else:
            os.environ["TMP"] = old_tmp


def topologically_sort_onnx_graph(model: Any) -> Any:
    graph = model.graph
    available = {value.name for value in graph.input}
    available.update(initializer.name for initializer in graph.initializer)
    available.update(initializer.values.name for initializer in graph.sparse_initializer)

    remaining_nodes = list(graph.node)
    ordered_nodes = []

    while remaining_nodes:
        progress = False
        next_remaining_nodes = []

        for node in remaining_nodes:
            input_names = [name for name in node.input if name]
            if all(name in available for name in input_names):
                ordered_nodes.append(node)
                for output_name in node.output:
                    if output_name:
                        available.add(output_name)
                progress = True
            else:
                next_remaining_nodes.append(node)

        if not progress:
            missing_names = []
            for node in next_remaining_nodes[:3]:
                missing_names.extend(name for name in node.input if name and name not in available)
            raise RuntimeError(
                "无法对 ONNX 图进行拓扑排序，缺失输入："
                f"{', '.join(missing_names[:10])}"
            )

        remaining_nodes = next_remaining_nodes

    del graph.node[:]
    graph.node.extend(ordered_nodes)
    return model


def validate_onnx_model(model_path: Path) -> None:
    import onnx

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    print(f"ONNX 校验通过：{model_path}")


def collect_calibration_image_paths(
    val_samples: list[dict[str, Any]],
    train_samples: list[dict[str, Any]],
    limit: int,
) -> list[Path]:
    selected_paths: list[Path] = []
    seen_paths: set[Path] = set()

    for samples in (val_samples, train_samples):
        for sample in samples:
            image_path = sample["image_path"]
            if image_path in seen_paths:
                continue

            seen_paths.add(image_path)
            selected_paths.append(image_path)
            if len(selected_paths) >= limit:
                return selected_paths

    return selected_paths


def export_onnx_fp32(model_dir: Path, output_dir: Path) -> Path:
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as error:
        raise RuntimeError(
            "缺少 optimum 导出依赖。请先安装 optimum[exporters]。"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    main_export(
        model_name_or_path=str(model_dir),
        output=output_dir,
        task="object-detection",
        device="cpu",
    )
    model_path = find_exported_onnx_file(output_dir, ONNX_FP32_FILE_NAME)
    validate_onnx_model(model_path)
    return model_path


def export_onnx_fp16(fp32_model_path: Path, fp32_dir: Path, output_dir: Path) -> Path:
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as error:
        raise RuntimeError(
            "缺少 FP16 转换依赖。请确认已安装 onnx 与 onnxruntime。"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_onnx_metadata_files(fp32_dir, output_dir)

    fp16_model_path = output_dir / ONNX_FP16_FILE_NAME
    model = onnx.load(str(fp32_model_path))
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    model_fp16 = topologically_sort_onnx_graph(model_fp16)
    onnx.save(model_fp16, str(fp16_model_path))
    validate_onnx_model(fp16_model_path)
    return fp16_model_path


def export_onnx_int8_dynamic(fp32_model_path: Path, fp32_dir: Path, output_dir: Path) -> Path:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as error:
        raise RuntimeError(
            "缺少 ONNX Runtime 量化依赖。请确认已安装 onnxruntime。"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_onnx_metadata_files(fp32_dir, output_dir)

    int8_model_path = output_dir / ONNX_INT8_DYNAMIC_FILE_NAME
    with patched_temp_workspace(output_dir / ".ort-quant-tmp"):
        quantize_dynamic(
            model_input=str(fp32_model_path),
            model_output=str(int8_model_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        )
    validate_onnx_model(int8_model_path)
    return int8_model_path


def export_onnx_int8_static(
    fp32_model_path: Path,
    fp32_dir: Path,
    output_dir: Path,
    image_processor: RTDetrImageProcessor,
    calibration_image_paths: list[Path],
) -> Path:
    try:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    except ImportError as error:
        raise RuntimeError(
            "缺少 ONNX Runtime 量化依赖。请确认已安装 onnxruntime。"
        ) from error

    if not calibration_image_paths:
        raise RuntimeError("未找到可用于 INT8 校准的图片。")

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_onnx_metadata_files(fp32_dir, output_dir)

    int8_model_path = output_dir / ONNX_INT8_FILE_NAME
    calibration_reader = RTDetrCalibrationDataReader(image_processor, calibration_image_paths)
    try:
        with patched_temp_workspace(output_dir / ".ort-quant-tmp"):
            quantize_static(
                model_input=str(fp32_model_path),
                model_output=str(int8_model_path),
                calibration_data_reader=calibration_reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            )
    except Exception as error:
        print(f"INT8 static 量化失败，回退到 dynamic 量化：{error}")
        return export_onnx_int8_dynamic(fp32_model_path, fp32_dir, output_dir)

    validate_onnx_model(int8_model_path)
    return int8_model_path


def main() -> None:
    runtime = configure_torch_runtime()
    set_seed(SEED)

    print(
        "训练配置："
        f" device={runtime['device']},"
        f" model={MODEL_NAME},"
        f" image_size={IMAGE_SIZE},"
        f" epochs={EPOCHS},"
        f" train_batch={TRAIN_BATCH_SIZE},"
        f" eval_batch={EVAL_BATCH_SIZE},"
        f" grad_accum={GRADIENT_ACCUMULATION_STEPS},"
        f" workers={DATALOADER_NUM_WORKERS},"
        f" export_onnx={EXPORT_ONNX},"
        f" export_fp16={EXPORT_ONNX_FP16},"
        f" export_int8={EXPORT_ONNX_INT8}"
    )

    (
        train_images_dir,
        val_images_dir,
        train_labels_dir,
        val_labels_dir,
        class_names,
    ) = load_dataset_config(DATASET_CONFIG_PATH)

    metadata_cache = load_metadata_cache(METADATA_CACHE_PATH)
    train_samples, train_summary = build_split_samples(
        "train",
        train_images_dir,
        train_labels_dir,
        len(class_names),
        metadata_cache,
    )
    val_samples, val_summary = build_split_samples(
        "val",
        val_images_dir,
        val_labels_dir,
        len(class_names),
        metadata_cache,
    )
    save_metadata_cache(METADATA_CACHE_PATH, metadata_cache)

    if not train_samples:
        raise RuntimeError("训练集为空，无法开始训练。")
    if not val_samples:
        raise RuntimeError("验证集为空，无法开始训练。")

    print_split_summary(train_summary)
    print_split_summary(val_summary)

    id2label = {index: label for index, label in enumerate(class_names)}
    label2id = {label: index for index, label in id2label.items()}

    image_processor = RTDetrImageProcessor.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        do_resize=True,
        size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        do_pad=True,
        pad_size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
    )

    train_dataset = YoloDetectionDataset(train_samples)
    eval_dataset = YoloDetectionDataset(val_samples)

    model = RTDetrV2ForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    if COMPILE_MODEL:
        if not hasattr(torch, "compile"):
            raise RuntimeError("当前 PyTorch 版本不支持 torch.compile。")
        model = torch.compile(model, backend=COMPILE_BACKEND)

    training_args = build_training_arguments(runtime)
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        image_processor=image_processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    resume_checkpoint = resolve_resume_checkpoint()
    write_training_summary(
        training_settings={
            "dataset_config_path": str(DATASET_CONFIG_PATH),
            "output_dir": str(OUTPUT_DIR),
            "metadata_cache_path": str(METADATA_CACHE_PATH),
            "model_name": MODEL_NAME,
            "image_size": IMAGE_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "dataloader_num_workers": DATALOADER_NUM_WORKERS,
            "logging_steps": LOGGING_STEPS,
            "save_total_limit": SAVE_TOTAL_LIMIT,
            "seed": SEED,
            "export_onnx": EXPORT_ONNX,
            "export_onnx_fp16": EXPORT_ONNX_FP16,
            "export_onnx_int8": EXPORT_ONNX_INT8,
            "overwrite_output": OVERWRITE_OUTPUT,
            "resume_from_checkpoint": RESUME_FROM_CHECKPOINT,
            "resolved_resume_checkpoint": resume_checkpoint,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "compile_model": COMPILE_MODEL,
            "compile_backend": COMPILE_BACKEND,
            "allow_tf32": ALLOW_TF32,
            "int8_calibration_image_limit": INT8_CALIBRATION_IMAGE_LIMIT,
        },
        class_names=class_names,
        split_summaries=[train_summary, val_summary],
        runtime=runtime,
        output_dir=OUTPUT_DIR,
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_model_dir = OUTPUT_DIR / "final-model"
    trainer.save_model(str(final_model_dir))
    image_processor.save_pretrained(str(final_model_dir))

    if EXPORT_ONNX:
        fp32_output_dir = OUTPUT_DIR / ONNX_FP32_DIR_NAME
        fp32_model_path = export_onnx_fp32(final_model_dir, fp32_output_dir)

        if EXPORT_ONNX_FP16:
            export_onnx_fp16(fp32_model_path, fp32_output_dir, OUTPUT_DIR / ONNX_FP16_DIR_NAME)

        if EXPORT_ONNX_INT8:
            calibration_image_paths = collect_calibration_image_paths(
                val_samples,
                train_samples,
                INT8_CALIBRATION_IMAGE_LIMIT,
            )
            print(f"INT8 校准图片数：{len(calibration_image_paths)}")
            export_onnx_int8_static(
                fp32_model_path,
                fp32_output_dir,
                OUTPUT_DIR / ONNX_INT8_DIR_NAME,
                image_processor,
                calibration_image_paths,
            )


if __name__ == "__main__":
    freeze_support()
    main()
