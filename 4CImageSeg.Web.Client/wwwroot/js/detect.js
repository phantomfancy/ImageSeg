import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.webgpu.min.mjs";

const runtimeConfig = {
  ortVersion: "1.24.3",
  inputWidth: 640,
  inputHeight: 640,
  scoreThreshold: 0.35,
  defaultMaxDetections: 5
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
ort.env.wasm.numThreads = 0;
ort.env.wasm.proxy = false;

if (ort.env.webgpu) {
  ort.env.webgpu.powerPreference = "high-performance";
}

let sessionStatePromise;
let sessionCacheKey;
let sessionModelOptions;
let realtimeState;

export async function listVideoDevices() {
  const devices = await navigator.mediaDevices.enumerateDevices();

  return devices
    .filter((device) => device.kind === "videoinput")
    .map((device, index) => ({
      deviceId: device.deviceId,
      label: device.label || `摄像头 ${index + 1}`
    }));
}

export async function runSingleImageDetection(imageDataUrl, inputSource, mode, modelOptions, runtimeOptions) {
  const resolvedModel = resolveModelOptions(modelOptions);
  const resolvedRuntime = resolveRuntimeOptions(runtimeOptions);
  const sessionState = await getSessionState(resolvedModel);
  const image = await loadImage(imageDataUrl);
  const owner = {
    sessionState,
    runtimeOptions: resolvedRuntime,
    inferenceBuffers: createInferenceBuffers()
  };
  const detections = await runDetectionOnCanvas(owner, image.canvas, image.width, image.height, resolvedRuntime);
  const annotatedImageDataUrl = drawDetections(image.canvas, detections);

  return {
    annotatedImageDataUrl,
    executionProvider: owner.sessionState.providerName,
    result: buildRecognitionResult(inputSource, mode, resolvedModel.modelVersion, detections)
  };
}

export async function startRealtimeDetection(videoElement, previewCanvas, request) {
  await disposeRealtimeDetection(videoElement, previewCanvas);

  const sourceKind = normalizeSourceKind(request?.sourceKind);
  const resolvedModel = resolveModelOptions(request?.modelOptions);
  const resolvedRuntime = resolveRuntimeOptions(request?.runtimeOptions);
  const sessionState = await getSessionState(resolvedModel);

  const state = {
    sourceKind,
    inputSource: normalizeDisplayName(request?.inputSource, sourceKind === "camera" ? "摄像头视频流" : "视频流"),
    mode: typeof request?.mode === "string" && request.mode.length > 0 ? request.mode : "local",
    modelVersion: resolvedModel.modelVersion,
    sessionState,
    runtimeOptions: resolvedRuntime,
    videoElement,
    previewCanvas,
    stream: null,
    objectUrl: null,
    stopRequested: false,
    completionPromise: null,
    resolveCompletion: null,
    inferenceBuffers: createInferenceBuffers(),
    lastFrameCanvas: document.createElement("canvas"),
    lastDetections: [],
    lastResult: null,
    lastAnnotatedImageDataUrl: null,
    lastFrameWidth: 0,
    lastFrameHeight: 0,
    hasPresentedFrame: false,
    failure: null
  };

  state.completionPromise = new Promise((resolve) => {
    state.resolveCompletion = resolve;
  });

  realtimeState = state;

  try {
    await prepareRealtimeSource(state, request);
    void pumpRealtimeDetection(state);
    return {
      providerName: state.sessionState.providerName
    };
  } catch (error) {
    state.failure = error;
    await finalizeRealtimeState(state);
    throw error;
  }
}

export async function stopRealtimeDetection() {
  if (!realtimeState) {
    return null;
  }

  const state = realtimeState;
  state.stopRequested = true;
  await state.completionPromise;

  if (state.failure) {
    throw state.failure;
  }

  return state.lastResult && state.lastAnnotatedImageDataUrl
    ? {
        annotatedImageDataUrl: state.lastAnnotatedImageDataUrl,
        result: state.lastResult
      }
    : null;
}

export async function disposeRealtimeDetection(videoElement, previewCanvas) {
  if (!realtimeState) {
    clearCanvas(previewCanvas);
    resetVideoElement(videoElement);
    return;
  }

  const state = realtimeState;
  state.stopRequested = true;
  await state.completionPromise;
}

async function pumpRealtimeDetection(state) {
  try {
    while (!state.stopRequested) {
      if (!isVideoReady(state.videoElement)) {
        await waitForNextVisualFrame();
        continue;
      }

      const frameWidth = state.videoElement.videoWidth;
      const frameHeight = state.videoElement.videoHeight;
      if (!frameWidth || !frameHeight) {
        await waitForNextVisualFrame();
        continue;
      }

      syncCanvasSize(state.lastFrameCanvas, frameWidth, frameHeight);
      syncCanvasSize(state.previewCanvas, frameWidth, frameHeight);

      const lastFrameContext = get2dContext(state.lastFrameCanvas, "无法获取视频帧绘制上下文。");
      lastFrameContext.drawImage(state.videoElement, 0, 0, frameWidth, frameHeight);
      if (!state.hasPresentedFrame) {
        drawPreviewFrame(state.previewCanvas, state.lastFrameCanvas, []);
      }

      const detections = await runDetectionOnCanvas(state, state.lastFrameCanvas, frameWidth, frameHeight, state.runtimeOptions);

      drawPreviewFrame(state.previewCanvas, state.lastFrameCanvas, detections);
      state.hasPresentedFrame = true;
      state.lastDetections = detections;
      state.lastFrameWidth = frameWidth;
      state.lastFrameHeight = frameHeight;
      state.lastResult = buildRecognitionResult(
        state.inputSource,
        state.mode,
        state.modelVersion,
        detections);

      await waitForNextVideoFrame(state.videoElement);
    }
  } catch (error) {
    state.failure = error;
    await finalizeRealtimeState(state);
    return;
  }

  await finalizeRealtimeState(state);
}

async function finalizeRealtimeState(state) {
  if (realtimeState === state && state.lastFrameWidth > 0 && state.lastFrameHeight > 0) {
    state.lastAnnotatedImageDataUrl = drawDetections(state.lastFrameCanvas, state.lastDetections);
  }

  cleanupRealtimeSource(state);

  if (realtimeState === state) {
    realtimeState = null;
  }

  if (typeof state.resolveCompletion === "function") {
    state.resolveCompletion();
    state.resolveCompletion = null;
  }
}

function cleanupRealtimeSource(state) {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }

  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
  }

  resetVideoElement(state.videoElement);
}

async function prepareRealtimeSource(state, request) {
  if (state.sourceKind === "camera") {
    await prepareCameraStream(state, request?.cameraDeviceId);
    return;
  }

  await prepareVideoFile(state, request?.videoBytes, request?.videoContentType);
}

async function prepareCameraStream(state, cameraDeviceId) {
  const constraints = cameraDeviceId
    ? { video: { deviceId: { exact: cameraDeviceId } }, audio: false }
    : { video: true, audio: false };

  state.stream = await navigator.mediaDevices.getUserMedia(constraints);
  state.videoElement.src = "";
  state.videoElement.srcObject = state.stream;
  state.videoElement.muted = true;
  state.videoElement.playsInline = true;
  await waitForVideoReady(state.videoElement);
  await state.videoElement.play();
}

async function prepareVideoFile(state, videoBytes, videoContentType) {
  const bytes = normalizeBinaryBytes(videoBytes, "无法解析本地视频文件数据。");
  if (bytes.length === 0) {
    throw new Error("本地视频文件为空。");
  }

  const blob = new Blob([bytes], {
    type: typeof videoContentType === "string" && videoContentType.length > 0
      ? videoContentType
      : "video/mp4"
  });

  state.objectUrl = URL.createObjectURL(blob);
  state.videoElement.srcObject = null;
  state.videoElement.src = state.objectUrl;
  state.videoElement.muted = true;
  state.videoElement.playsInline = true;
  state.videoElement.loop = true;
  await waitForVideoReady(state.videoElement);
  await state.videoElement.play();
}

async function waitForVideoReady(videoElement) {
  if (isVideoReady(videoElement)) {
    return;
  }

  await new Promise((resolve, reject) => {
    const handleCanPlay = () => {
      cleanup();
      resolve();
    };

    const handleLoadedMetadata = () => {
      if (isVideoReady(videoElement)) {
        cleanup();
        resolve();
      }
    };

    const handleError = () => {
      cleanup();
      reject(new Error("无法加载待识别视频流。"));
    };

    const cleanup = () => {
      videoElement.removeEventListener("canplay", handleCanPlay);
      videoElement.removeEventListener("loadedmetadata", handleLoadedMetadata);
      videoElement.removeEventListener("error", handleError);
    };

    videoElement.addEventListener("canplay", handleCanPlay, { once: true });
    videoElement.addEventListener("loadedmetadata", handleLoadedMetadata, { once: true });
    videoElement.addEventListener("error", handleError, { once: true });
  });
}

function isVideoReady(videoElement) {
  return !!videoElement &&
    videoElement.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    videoElement.videoWidth > 0 &&
    videoElement.videoHeight > 0;
}

function resetVideoElement(videoElement) {
  if (!videoElement) {
    return;
  }

  videoElement.pause();
  videoElement.srcObject = null;
  videoElement.removeAttribute("src");
  videoElement.load();
}

async function waitForNextVideoFrame(videoElement) {
  if (typeof videoElement?.requestVideoFrameCallback === "function") {
    await new Promise((resolve) => {
      videoElement.requestVideoFrameCallback(() => resolve());
    });
    return;
  }

  await waitForNextVisualFrame();
}

async function waitForNextVisualFrame() {
  await new Promise((resolve) => window.requestAnimationFrame(() => resolve()));
}

async function runDetectionOnCanvas(owner, sourceCanvas, width, height, runtimeOptions) {
  const inputTensor = createInputTensor(sourceCanvas, owner.inferenceBuffers);

  try {
    const outputMap = await owner.sessionState.session.run({
      [owner.sessionState.session.inputNames[0]]: inputTensor
    });

    const logits = getOutputTensor(outputMap, owner.sessionState.session.outputNames, "logits");
    const predBoxes = getOutputTensor(outputMap, owner.sessionState.session.outputNames, "pred_boxes");
    return extractDetections(logits, predBoxes, width, height, runtimeOptions);
  } catch (error) {
    if (!shouldRetryWithoutGraphCapture(owner.sessionState, error)) {
      throw error;
    }

    owner.sessionState = await switchSessionToNonCapturedWebGpu();
    const outputMap = await owner.sessionState.session.run({
      [owner.sessionState.session.inputNames[0]]: inputTensor
    });

    const logits = getOutputTensor(outputMap, owner.sessionState.session.outputNames, "logits");
    const predBoxes = getOutputTensor(outputMap, owner.sessionState.session.outputNames, "pred_boxes");
    return extractDetections(logits, predBoxes, width, height, runtimeOptions);
  }
}

function buildRecognitionResult(inputSource, mode, modelVersion, detections) {
  return {
    inputSource,
    mode,
    modelVersion,
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
  };
}

async function getSessionState(modelOptions) {
  if (!sessionStatePromise || sessionCacheKey !== modelOptions.cacheKey) {
    sessionCacheKey = modelOptions.cacheKey;
    sessionModelOptions = modelOptions;
    sessionStatePromise = createSessionWithFallback(modelOptions).catch((error) => {
      sessionStatePromise = undefined;
      sessionCacheKey = undefined;
      sessionModelOptions = undefined;
      throw error;
    });
  }

  return sessionStatePromise;
}

async function createSessionWithFallback(modelOptions, forceDisableGraphCapture = false) {
  const errors = [];
  const modelSource = await getModelSource(modelOptions);

  for (const plan of resolveExecutionPlans(forceDisableGraphCapture)) {
    try {
      const session = await ort.InferenceSession.create(
        modelSource,
        plan.sessionOptions);

      return {
        session,
        providerName: plan.providerName,
        graphCaptureEnabled: plan.graphCaptureEnabled === true
      };
    } catch (error) {
      errors.push(`${plan.label}：${formatErrorMessage(error)}`);
    }
  }

  throw new Error(`无法通过 ONNX Runtime Web 加载模型 ${modelOptions.modelVersion}。${errors.join("；")}`);
}

function resolveExecutionPlans(forceDisableGraphCapture = false) {
  const plans = [];

  if (typeof navigator !== "undefined" && "gpu" in navigator && !forceDisableGraphCapture) {
    plans.push({
      providerName: "webgpu",
      label: "webgpu(graph-capture)",
      graphCaptureEnabled: true,
      sessionOptions: {
        executionProviders: ["webgpu"],
        enableGraphCapture: true,
        freeDimensionOverrides: {
          batch: 1,
          height: runtimeConfig.inputHeight,
          width: runtimeConfig.inputWidth
        }
      }
    });

  }

  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    plans.push({
      providerName: "webgpu",
      label: "webgpu",
      graphCaptureEnabled: false,
      sessionOptions: {
        executionProviders: ["webgpu"]
      }
    });
  }

  plans.push({
    providerName: "wasm",
    label: "wasm",
    graphCaptureEnabled: false,
    sessionOptions: {
      executionProviders: ["wasm"]
    }
  });

  return plans;
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
    const modelBytes = normalizeBinaryBytes(modelOptions?.modelBytes, "无法解析本地模型文件数据。");
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

function resolveRuntimeOptions(runtimeOptions) {
  const maxDetections = Number.isFinite(runtimeOptions?.maxDetections)
    ? runtimeOptions.maxDetections
    : runtimeConfig.defaultMaxDetections;

  return {
    maxDetections: clamp(Math.trunc(maxDetections), 1, 50)
  };
}

function normalizeSourceKind(value) {
  return typeof value === "string" && value.trim().toLowerCase() === "camera"
    ? "camera"
    : "video";
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

function normalizeBinaryBytes(value, errorMessage) {
  if (!value) {
    return new Uint8Array();
  }

  if (value instanceof Uint8Array) {
    return value;
  }

  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }

  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer.slice(
      value.byteOffset,
      value.byteOffset + value.byteLength
    ));
  }

  if (Array.isArray(value)) {
    return Uint8Array.from(value);
  }

  if (typeof value.length === "number") {
    return Uint8Array.from(value);
  }

  throw new Error(errorMessage);
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

function createInferenceBuffers() {
  return {
    preprocessCanvas: document.createElement("canvas"),
    preprocessContext: null,
    tensorData: null,
    inputTensor: null
  };
}

function createInputTensor(sourceCanvas, inferenceBuffers) {
  syncCanvasSize(inferenceBuffers.preprocessCanvas, runtimeConfig.inputWidth, runtimeConfig.inputHeight);

  if (!inferenceBuffers.preprocessContext) {
    inferenceBuffers.preprocessContext = get2dContext(inferenceBuffers.preprocessCanvas, "浏览器不支持 Canvas 2D。");
  }

  inferenceBuffers.preprocessContext.drawImage(
    sourceCanvas,
    0,
    0,
    runtimeConfig.inputWidth,
    runtimeConfig.inputHeight);

  const { data } = inferenceBuffers.preprocessContext.getImageData(
    0,
    0,
    runtimeConfig.inputWidth,
    runtimeConfig.inputHeight);

  const tensorLength = 3 * runtimeConfig.inputHeight * runtimeConfig.inputWidth;
  if (!(inferenceBuffers.tensorData instanceof Float32Array) || inferenceBuffers.tensorData.length !== tensorLength) {
    inferenceBuffers.tensorData = new Float32Array(tensorLength);
    inferenceBuffers.inputTensor = new ort.Tensor(
      "float32",
      inferenceBuffers.tensorData,
      [1, 3, runtimeConfig.inputHeight, runtimeConfig.inputWidth]);
  }

  const channelSize = runtimeConfig.inputHeight * runtimeConfig.inputWidth;
  for (let index = 0; index < channelSize; index += 1) {
    const pixelOffset = index * 4;
    inferenceBuffers.tensorData[index] = data[pixelOffset] / 255;
    inferenceBuffers.tensorData[channelSize + index] = data[pixelOffset + 1] / 255;
    inferenceBuffers.tensorData[channelSize * 2 + index] = data[pixelOffset + 2] / 255;
  }

  return inferenceBuffers.inputTensor;
}

function extractDetections(logits, predBoxes, imageWidth, imageHeight, runtimeOptions) {
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
    for (let classIndex = 0; classIndex < classCountWithNoObject; classIndex += 1) {
      sum += Math.exp(scores[queryIndex * classCountWithNoObject + classIndex] - maxLogit);
    }

    let bestClassId = 0;
    let bestScore = 0;
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      const probability = Math.exp(scores[queryIndex * classCountWithNoObject + classIndex] - maxLogit) / sum;
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
    .slice(0, runtimeOptions.maxDetections);
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

  const context = get2dContext(canvas, "浏览器不支持 Canvas 2D。");
  context.drawImage(sourceCanvas, 0, 0);
  renderDetectionBoxes(context, detections);
  return canvas.toDataURL("image/png");
}

function drawPreviewFrame(previewCanvas, frameCanvas, detections) {
  if (!previewCanvas) {
    return;
  }

  syncCanvasSize(previewCanvas, frameCanvas.width, frameCanvas.height);
  const context = get2dContext(previewCanvas, "浏览器不支持 Canvas 2D。");
  context.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  context.drawImage(frameCanvas, 0, 0);
  renderDetectionBoxes(context, detections);
}

function renderDetectionBoxes(context, detections) {
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
}

function pickColor(index) {
  const palette = ["#f25f5c", "#247ba0", "#70c1b3", "#ff9f1c", "#6a4c93", "#0081a7"];
  return palette[index % palette.length];
}

function clearCanvas(canvas) {
  if (!canvas) {
    return;
  }

  const context = canvas.getContext("2d");
  if (context) {
    context.clearRect(0, 0, canvas.width, canvas.height);
  }
}

function syncCanvasSize(canvas, width, height) {
  if (!canvas || canvas.width === width && canvas.height === height) {
    return;
  }

  canvas.width = width;
  canvas.height = height;
}

function get2dContext(canvas, errorMessage) {
  const context = canvas?.getContext("2d");
  if (!context) {
    throw new Error(errorMessage);
  }

  return context;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function formatErrorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function shouldRetryWithoutGraphCapture(sessionState, error) {
  const message = formatErrorMessage(error);
  return sessionState?.providerName === "webgpu" &&
    sessionState?.graphCaptureEnabled === true &&
    message.includes("External buffer must be provided");
}

async function switchSessionToNonCapturedWebGpu() {
  if (!sessionModelOptions) {
    throw new Error("当前会话缺少模型配置，无法切换 WebGPU 图捕获模式。");
  }

  const fallbackPromise = createSessionWithFallback(sessionModelOptions, true);
  sessionStatePromise = fallbackPromise;
  return await fallbackPromise;
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

  const context = get2dContext(canvas, "浏览器不支持 Canvas 2D。");
  context.drawImage(image, 0, 0);

  return {
    canvas,
    width: canvas.width,
    height: canvas.height
  };
}
