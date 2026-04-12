import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.webgpu.min.mjs";

const runtimeConfig = {
  ortVersion: "1.24.3",
  inputWidth: 640,
  inputHeight: 640,
  scoreThreshold: 0.35,
  maxDetections: 20
};

const defaultModelConfig = {
  cacheKey: "builtin:yolo26s:model_int8",
  sourceType: "default",
  modelUrl: "/models/yolo26s/model_int8.onnx",
  modelVersion: "builtin:/models/yolo26s/model_int8.onnx",
  displayName: "内置模型 yolo26s / model_int8.onnx",
  modelBytes: undefined
};

const cocoLabels = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush"
];

const ortDistBaseUrl = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${runtimeConfig.ortVersion}/dist/`;
ort.env.wasm.wasmPaths = ortDistBaseUrl;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

let sessionPromise;
let sessionCacheKey;

export async function runSingleImageDetection(imageDataUrl, inputSource, mode, modelOptions) {
  const resolvedModel = resolveModelOptions(modelOptions);
  const session = await getSession(resolvedModel);
  const image = await loadImage(imageDataUrl);
  const inputTensor = createInputTensor(image.canvas);
  const inputName = session.inputNames[0];
  const outputMap = await session.run({
    [inputName]: inputTensor
  });

  const logits = getOutputTensor(outputMap, session.outputNames, "logits");
  const predBoxes = getOutputTensor(outputMap, session.outputNames, "pred_boxes");
  const detections = extractDetections(logits, predBoxes, image.width, image.height);
  const annotatedImageDataUrl = drawDetections(image.canvas, detections);

  return {
    annotatedImageDataUrl,
    result: {
      inputSource,
      mode,
      modelVersion: resolvedModel.modelVersion,
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

async function getSession(modelOptions) {
  if (!sessionPromise || sessionCacheKey !== modelOptions.cacheKey) {
    sessionCacheKey = modelOptions.cacheKey;
    sessionPromise = createSessionWithFallback(modelOptions).catch((error) => {
      sessionPromise = undefined;
      sessionCacheKey = undefined;
      throw error;
    });
  }

  return sessionPromise;
}

async function createSessionWithFallback(modelOptions) {
  const errors = [];
  const modelSource = await getModelSource(modelOptions);

  for (const provider of getExecutionProviders()) {
    try {
      return await ort.InferenceSession.create(modelSource, {
        executionProviders: [provider]
      });
    } catch (error) {
      errors.push(`${provider}：${formatErrorMessage(error)}`);
    }
  }

  throw new Error(`无法通过 ONNX Runtime Web 加载模型 ${modelOptions.modelVersion}。${errors.join("；")}`);
}

function resolveModelOptions(modelOptions) {
  const sourceType = typeof modelOptions?.sourceType === "string"
    ? modelOptions.sourceType.trim().toLowerCase()
    : defaultModelConfig.sourceType;

  if (sourceType === "url") {
    const modelUrl = typeof modelOptions?.modelUrl === "string" ? modelOptions.modelUrl.trim() : "";
    return {
      cacheKey: normalizeCacheKey(modelOptions?.cacheKey, `url:${modelUrl}`),
      sourceType,
      modelUrl,
      modelVersion: normalizeModelVersion(modelOptions?.modelVersion, modelUrl),
      displayName: normalizeDisplayName(modelOptions?.displayName, modelUrl),
      modelBytes: undefined
    };
  }

  if (sourceType === "file") {
    const modelBytes = normalizeModelBytes(modelOptions?.modelBytes);
    return {
      cacheKey: normalizeCacheKey(modelOptions?.cacheKey, `file:${modelOptions?.displayName ?? modelBytes.length}`),
      sourceType,
      modelUrl: undefined,
      modelVersion: normalizeModelVersion(modelOptions?.modelVersion, modelOptions?.displayName ?? "本地文件"),
      displayName: normalizeDisplayName(modelOptions?.displayName, "本地文件"),
      modelBytes
    };
  }

  return {
    ...defaultModelConfig
  };
}

function normalizeCacheKey(value, fallbackValue) {
  return typeof value === "string" && value.length > 0 ? value : fallbackValue;
}

function normalizeModelVersion(value, fallbackValue) {
  return typeof value === "string" && value.length > 0 ? value : fallbackValue;
}

function normalizeDisplayName(value, fallbackValue) {
  return typeof value === "string" && value.length > 0 ? value : fallbackValue;
}

function normalizeModelBytes(modelBytes) {
  if (!modelBytes) {
    return new Uint8Array();
  }

  if (modelBytes instanceof Uint8Array) {
    return modelBytes;
  }

  if (modelBytes instanceof ArrayBuffer) {
    return new Uint8Array(modelBytes);
  }

  if (ArrayBuffer.isView(modelBytes)) {
    return new Uint8Array(modelBytes.buffer.slice(
      modelBytes.byteOffset,
      modelBytes.byteOffset + modelBytes.byteLength
    ));
  }

  if (Array.isArray(modelBytes)) {
    return Uint8Array.from(modelBytes);
  }

  if (typeof modelBytes.length === "number") {
    return Uint8Array.from(modelBytes);
  }

  throw new Error("无法解析本地模型文件数据。");
}

async function getModelSource(modelOptions) {
  if (modelOptions.sourceType === "file") {
    if (!(modelOptions.modelBytes instanceof Uint8Array) || modelOptions.modelBytes.length === 0) {
      throw new Error("本地模型文件为空。");
    }

    return modelOptions.modelBytes;
  }

  if (typeof modelOptions.modelUrl !== "string" || modelOptions.modelUrl.length === 0) {
    throw new Error("模型 URL 为空。");
  }

  return await fetchModelBytes(modelOptions.modelUrl);
}

async function fetchModelBytes(modelUrl) {
  let response;

  try {
    response = await fetch(modelUrl, {
      method: "GET",
      cache: "no-store"
    });
  } catch (error) {
    throw new Error(`浏览器无法直接下载模型。请检查 URL 是否可访问且已允许跨域访问。${formatErrorMessage(error)}`);
  }

  if (!response.ok) {
    throw new Error(`模型下载失败：HTTP ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  if (!buffer || buffer.byteLength === 0) {
    throw new Error("模型下载结果为空。");
  }

  return new Uint8Array(buffer);
}

function getExecutionProviders() {
  const providers = [];

  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    providers.push("webgpu");
  }

  providers.push("wasm");
  return providers;
}

function getOutputTensor(outputMap, outputNames, expectedName) {
  const exactName = outputNames.find((item) => item.toLowerCase() === expectedName.toLowerCase());
  if (exactName && outputMap[exactName]) {
    return outputMap[exactName];
  }

  const partialName = outputNames.find((item) => item.toLowerCase().includes(expectedName.toLowerCase()));
  if (partialName && outputMap[partialName]) {
    return outputMap[partialName];
  }

  throw new Error(`模型输出中未找到 ${expectedName}。`);
}

function createInputTensor(sourceCanvas) {
  const canvas = document.createElement("canvas");
  canvas.width = runtimeConfig.inputWidth;
  canvas.height = runtimeConfig.inputHeight;

  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("浏览器不支持 Canvas 2D。");
  }

  context.drawImage(sourceCanvas, 0, 0, runtimeConfig.inputWidth, runtimeConfig.inputHeight);
  const { data } = context.getImageData(0, 0, runtimeConfig.inputWidth, runtimeConfig.inputHeight);
  const tensorData = new Float32Array(1 * 3 * runtimeConfig.inputHeight * runtimeConfig.inputWidth);
  const channelSize = runtimeConfig.inputHeight * runtimeConfig.inputWidth;

  for (let index = 0; index < channelSize; index += 1) {
    const pixelOffset = index * 4;
    tensorData[index] = data[pixelOffset] / 255;
    tensorData[channelSize + index] = data[pixelOffset + 1] / 255;
    tensorData[channelSize * 2 + index] = data[pixelOffset + 2] / 255;
  }

  return new ort.Tensor("float32", tensorData, [1, 3, runtimeConfig.inputHeight, runtimeConfig.inputWidth]);
}

function extractDetections(logits, predBoxes, imageWidth, imageHeight) {
  validateTensor(logits, "logits");
  validateTensor(predBoxes, "pred_boxes");
  validateBoxDimensions(predBoxes.dims);

  const scores = logits.data;
  const boxes = predBoxes.data;
  const queryCount = resolveQueryCount(logits.dims, predBoxes.dims, boxes.length);
  const classCountWithNoObject = resolveClassCount(logits.dims, scores.length, queryCount);
  const classCount = Math.max(1, classCountWithNoObject - 1);
  const detections = [];

  for (let queryIndex = 0; queryIndex < queryCount; queryIndex += 1) {
    let maxLogit = Number.NEGATIVE_INFINITY;
    for (let classIndex = 0; classIndex < classCountWithNoObject; classIndex += 1) {
      const current = scores[queryIndex * classCountWithNoObject + classIndex];
      if (current > maxLogit) {
        maxLogit = current;
      }
    }

    let sum = 0;
    const normalizedScores = new Float32Array(classCountWithNoObject);
    for (let classIndex = 0; classIndex < classCountWithNoObject; classIndex += 1) {
      const value = Math.exp(scores[queryIndex * classCountWithNoObject + classIndex] - maxLogit);
      normalizedScores[classIndex] = value;
      sum += value;
    }

    let bestClassId = 0;
    let bestScore = 0;
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      const probability = normalizedScores[classIndex] / sum;
      if (probability > bestScore) {
        bestScore = probability;
        bestClassId = classIndex;
      }
    }

    if (bestScore < runtimeConfig.scoreThreshold) {
      continue;
    }

    const cx = boxes[queryIndex * 4];
    const cy = boxes[queryIndex * 4 + 1];
    const width = boxes[queryIndex * 4 + 2];
    const height = boxes[queryIndex * 4 + 3];

    const left = clamp((cx - width / 2) * imageWidth, 0, imageWidth);
    const top = clamp((cy - height / 2) * imageHeight, 0, imageHeight);
    const right = clamp((cx + width / 2) * imageWidth, 0, imageWidth);
    const bottom = clamp((cy + height / 2) * imageHeight, 0, imageHeight);

    detections.push({
      label: cocoLabels[bestClassId] ?? `class-${bestClassId}`,
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
    .slice(0, runtimeConfig.maxDetections);
}

function validateTensor(tensor, name) {
  if (!tensor?.data || !Array.isArray(tensor?.dims)) {
    throw new Error(`模型输出缺少 ${name}。`);
  }
}

function validateBoxDimensions(boxDims) {
  const boxSize = boxDims.length > 0 ? boxDims[boxDims.length - 1] : undefined;
  if (boxSize !== undefined && boxSize !== 4) {
    throw new Error(`pred_boxes 维度异常：${boxDims.join("x")}。`);
  }
}

function resolveQueryCount(scoreDims, boxDims, boxLength) {
  const queryCountFromScores = scoreDims.length >= 2 ? scoreDims[scoreDims.length - 2] : undefined;
  const queryCountFromBoxes = boxDims.length >= 2 ? boxDims[boxDims.length - 2] : undefined;

  if (
    Number.isInteger(queryCountFromScores) &&
    Number.isInteger(queryCountFromBoxes) &&
    queryCountFromScores !== queryCountFromBoxes
  ) {
    throw new Error(`模型输出维度不一致：logits=${scoreDims.join("x")}，pred_boxes=${boxDims.join("x")}。`);
  }

  const queryCount = queryCountFromBoxes ?? queryCountFromScores ?? Math.floor(boxLength / 4);
  if (!Number.isInteger(queryCount) || queryCount <= 0) {
    throw new Error("无法从模型输出中解析候选框数量。");
  }

  return queryCount;
}

function resolveClassCount(scoreDims, scoreLength, queryCount) {
  const classCountFromDims = scoreDims.length > 0 ? scoreDims[scoreDims.length - 1] : undefined;
  const classCount = classCountFromDims ?? Math.floor(scoreLength / queryCount);

  if (!Number.isInteger(classCount) || classCount <= 0) {
    throw new Error("无法从模型输出中解析类别数量。");
  }

  return classCount;
}

function drawDetections(sourceCanvas, detections) {
  const canvas = document.createElement("canvas");
  canvas.width = sourceCanvas.width;
  canvas.height = sourceCanvas.height;

  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("浏览器不支持 Canvas 2D。");
  }

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

function formatErrorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

async function loadImage(dataUrl) {
  const image = new Image();
  image.decoding = "async";

  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error("无法加载待识别图片。"));
    image.src = dataUrl;
  });

  if (typeof image.decode === "function") {
    try {
      await image.decode();
    } catch {
      // 某些浏览器在图片已可用时会抛出解码异常，此时直接使用已加载图像即可。
    }
  }

  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;

  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("浏览器不支持 Canvas 2D。");
  }

  context.drawImage(image, 0, 0);

  return {
    canvas,
    width: canvas.width,
    height: canvas.height
  };
}
