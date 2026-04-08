import { AutoModel, AutoProcessor, RawImage } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const modelConfig = {
  modelId: "onnx-community/yolo26s-ONNX",
  modelVersion: "onnx-community/yolo26s-ONNX",
  scoreThreshold: 0.35,
  maxDetections: 20
};

let modelPromise;
let processorPromise;

export async function runSingleImageDetection(imageDataUrl, inputSource, mode) {
  const [model, processor] = await Promise.all([getModel(), getProcessor()]);
  const image = await loadImage(imageDataUrl);
  const rawImage = RawImage.fromCanvas(image.canvas);
  const inputs = await processor(rawImage);
  const output = await model(inputs);
  const detections = extractDetections(output, model.config.id2label, image.width, image.height);
  const annotatedImageDataUrl = drawDetections(image.canvas, detections);

  return {
    annotatedImageDataUrl,
    result: {
      inputSource,
      mode,
      modelVersion: modelConfig.modelVersion,
      detectedAtUtc: new Date().toISOString(),
      detections: detections.map((item) => ({
        label: item.label,
        confidence: item.confidence,
        box: {
          x: item.box.x,
          y: item.box.y,
          width: item.box.width,
          height: item.box.height
        }
      }))
    }
  };
}

async function getModel() {
  if (!modelPromise) {
    modelPromise = AutoModel.from_pretrained(modelConfig.modelId, createModelOptions()).catch((error) => {
      modelPromise = undefined;
      throw error;
    });
  }

  return modelPromise;
}

async function getProcessor() {
  if (!processorPromise) {
    processorPromise = AutoProcessor.from_pretrained(modelConfig.modelId).catch((error) => {
      processorPromise = undefined;
      throw error;
    });
  }

  return processorPromise;
}

function createModelOptions() {
  if (navigator.gpu) {
    return {
      device: "webgpu",
      dtype: "fp32"
    };
  }

  // 在非 WebGPU 浏览器中，Transformers.js 默认走 CPU/WASM。
  return {
    dtype: "q8"
  };
}

function extractDetections(output, id2label, imageWidth, imageHeight) {
  const scores = output.logits.sigmoid().data;
  const boxes = output.pred_boxes.data;
  const detections = [];

  for (let index = 0; index < 300; index += 1) {
    let bestScore = 0;
    let bestClassId = 0;

    for (let classIndex = 0; classIndex < 80; classIndex += 1) {
      const score = scores[index * 80 + classIndex];
      if (score > bestScore) {
        bestScore = score;
        bestClassId = classIndex;
      }
    }

    if (bestScore < modelConfig.scoreThreshold) {
      continue;
    }

    const cx = boxes[index * 4];
    const cy = boxes[index * 4 + 1];
    const width = boxes[index * 4 + 2];
    const height = boxes[index * 4 + 3];

    const left = clamp((cx - width / 2) * imageWidth, 0, imageWidth);
    const top = clamp((cy - height / 2) * imageHeight, 0, imageHeight);
    const right = clamp((cx + width / 2) * imageWidth, 0, imageWidth);
    const bottom = clamp((cy + height / 2) * imageHeight, 0, imageHeight);

    detections.push({
      label: id2label?.[bestClassId] ?? `Class ${bestClassId}`,
      confidence: bestScore,
      box: {
        x: left,
        y: top,
        width: Math.max(0, right - left),
        height: Math.max(0, bottom - top)
      }
    });
  }

  return detections
    .sort((left, right) => right.confidence - left.confidence)
    .slice(0, modelConfig.maxDetections);
}

function drawDetections(sourceCanvas, detections) {
  const canvas = document.createElement("canvas");
  canvas.width = sourceCanvas.width;
  canvas.height = sourceCanvas.height;

  const context = canvas.getContext("2d");
  context.drawImage(sourceCanvas, 0, 0);
  context.lineWidth = 3;
  context.font = "16px 'Segoe UI', sans-serif";
  context.textBaseline = "top";

  detections.forEach((item, index) => {
    const color = pickColor(index);
    const label = `${item.label} ${(item.confidence * 100).toFixed(1)}%`;
    const textMetrics = context.measureText(label);
    const textWidth = textMetrics.width + 12;
    const textHeight = 26;

    context.strokeStyle = color;
    context.fillStyle = color;
    context.strokeRect(item.box.x, item.box.y, item.box.width, item.box.height);

    const textY = item.box.y > textHeight ? item.box.y - textHeight : item.box.y;
    context.fillRect(item.box.x, textY, textWidth, textHeight);
    context.fillStyle = "#fff";
    context.fillText(label, item.box.x + 6, textY + 5);
  });

  return canvas.toDataURL("image/png");
}

function pickColor(index) {
  const palette = ["#f25f5c", "#247ba0", "#70c1b3", "#ff9f1c", "#6a4c93", "#0081a7"];
  return palette[index % palette.length];
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

async function loadImage(dataUrl) {
  const image = new Image();
  image.decoding = "async";
  image.src = dataUrl;
  if (typeof image.decode === "function") {
    await image.decode();
  } else {
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("无法加载待识别图片。"));
    });
  }

  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;

  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0);

  return {
    canvas,
    width: canvas.width,
    height: canvas.height
  };
}
