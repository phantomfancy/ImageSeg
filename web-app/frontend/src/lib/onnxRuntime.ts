import * as ort from 'onnxruntime-web/webgpu'
import {
  decodeDetections,
  normalizeScore,
  sigmoid,
  type PreprocessedImageGeometry,
  type RecognitionResult,
  type ResolvedModelContract,
  type TensorLike,
} from '../../../contracts/src/index.ts'
import {
  finalizeModelImport,
  type FinalizeModelImportOptions,
  type InspectedModelPackage,
} from './modelPackage.ts'

ort.env.wasm.proxy = false
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = {
  mjs: new URL('/ort/ort-wasm-simd-threaded.asyncify.mjs', window.location.href),
  wasm: new URL('/ort/ort-wasm-simd-threaded.asyncify.wasm', window.location.href),
}

if (ort.env.webgpu) {
  ort.env.webgpu.powerPreference = 'high-performance'
}

const WEBGPU_UNSUPPORTED_MESSAGE = '当前浏览器不支持 WebGPU，无法执行识别。'

type SessionState = {
  session: ort.InferenceSession
  providerName: 'webgpu'
}

const sessionCache = new Map<string, Promise<SessionState>>()

export interface WebGpuSupportState {
  supported: boolean
  message?: string
}

export interface ImportedModel {
  key: string
  fileName: string
  bytes: Uint8Array
  contract: InspectedModelPackage['contract']
  providerName: 'webgpu'
  sidecars: InspectedModelPackage['sidecars']
  webGpuCompatibility: InspectedModelPackage['webGpuCompatibility']
}

export interface DetectionRun {
  providerName: 'webgpu'
  annotatedCanvas: HTMLCanvasElement
  recognitionResult: RecognitionResult
  runtimeMessage?: string
}

export interface RunDetectionOptions {
  scoreThresholdOverride?: number
  maxDetectionsOverride?: number
}

type PreparedInput = {
  tensor: ort.Tensor
  geometry: PreprocessedImageGeometry
}

export function getWebGpuSupportState(): WebGpuSupportState {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    return {
      supported: false,
      message: WEBGPU_UNSUPPORTED_MESSAGE,
    }
  }

  return { supported: true }
}

export async function inspectImportedModel(options: FinalizeModelImportOptions): Promise<ImportedModel> {
  const inspectedPackage = await finalizeModelImport(options)
  const {
    key,
    fileName,
    bytes,
    contract,
    sidecars,
  } = inspectedPackage

  return {
    key,
    fileName,
    bytes,
    contract,
    providerName: 'webgpu',
    sidecars,
    webGpuCompatibility: inspectedPackage.webGpuCompatibility,
  }
}

export async function runSingleImageDetection(
  model: ImportedModel,
  imageFile: File,
  options?: RunDetectionOptions,
): Promise<DetectionRun> {
  const sourceCanvas = await loadImageCanvas(imageFile)
  return runDetectionOnCanvas(model, sourceCanvas, imageFile.name, options)
}

export async function runDetectionOnCanvas(
  model: ImportedModel,
  sourceCanvas: HTMLCanvasElement,
  inputSource: string,
  options?: RunDetectionOptions,
): Promise<DetectionRun> {
  assertModelWebGpuCompatible(model)
  const sessionState = await getSessionState(model.key, model.bytes)
  const detectionContract = createDetectionContract(model.contract, options)
  const preparedInput = createPreparedInput(sourceCanvas, detectionContract)
  const outputMap = await sessionState.session.run({
    [detectionContract.preprocess.inputTensorName]: preparedInput.tensor,
  })
  const normalizedOutputs = await normalizeOutputs(outputMap)
  const detections = decodeDetections(
    detectionContract,
    normalizedOutputs,
    preparedInput.geometry,
  )

  return {
    providerName: sessionState.providerName,
    annotatedCanvas: drawDetections(sourceCanvas, detections),
    recognitionResult: {
      inputSource,
      modelVersion: model.fileName,
      detectedAtUtc: new Date().toISOString(),
      detections,
    },
    runtimeMessage: detections.length === 0
      ? buildNoDetectionsRuntimeMessage(detectionContract, normalizedOutputs)
      : undefined,
  }
}

async function getSessionState(key: string, bytes: Uint8Array): Promise<SessionState> {
  assertWebGpuSupported()
  const existing = sessionCache.get(key)
  if (existing) {
    return existing
  }

  const promise = createWebGpuSession(bytes).catch((error: unknown) => {
    sessionCache.delete(key)
    throw error
  })

  sessionCache.set(key, promise)
  return promise
}

async function createWebGpuSession(bytes: Uint8Array): Promise<SessionState> {
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['webgpu'],
  })

  return {
    session,
    providerName: 'webgpu',
  }
}

async function normalizeOutputs(
  outputMap: ort.InferenceSession.ReturnType,
): Promise<Record<string, TensorLike>> {
  const entries = await Promise.all(
    Object.entries(outputMap).map(async ([name, value]) => {
      const tensor = value as ort.Tensor
      const rawData = typeof tensor.getData === 'function'
        ? await tensor.getData()
        : tensor.data
      const data = ensureNumericTensorData(rawData, name)

      return [name, {
        dims: [...tensor.dims],
        data,
      } satisfies TensorLike] as const
    }),
  )

  return Object.fromEntries(entries)
}

async function loadImageCanvas(file: File): Promise<HTMLCanvasElement> {
  if (typeof createImageBitmap === 'function') {
    const bitmap = await createImageBitmap(file)
    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const context = get2dContext(canvas)
    context.drawImage(bitmap, 0, 0)
    bitmap.close()
    return canvas
  }

  const objectUrl = URL.createObjectURL(file)
  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const element = new Image()
      element.onload = () => resolve(element)
      element.onerror = () => reject(new Error('无法加载待识别图片。'))
      element.src = objectUrl
    })

    const canvas = document.createElement('canvas')
    canvas.width = image.naturalWidth || image.width
    canvas.height = image.naturalHeight || image.height
    get2dContext(canvas).drawImage(image, 0, 0)
    return canvas
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}

export function drawSourceToCanvas(
  source: CanvasImageSource,
  targetCanvas?: HTMLCanvasElement,
): HTMLCanvasElement {
  const { width, height } = resolveSourceDimensions(source)
  const canvas = targetCanvas ?? document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = get2dContext(canvas)
  context.drawImage(source, 0, 0, width, height)
  return canvas
}

function createPreparedInput(
  sourceCanvas: HTMLCanvasElement,
  contract: ResolvedModelContract,
): PreparedInput {
  const targetWidth = contract.preprocess.imageWidth
  const targetHeight = contract.preprocess.imageHeight
  const preprocessCanvas = document.createElement('canvas')
  preprocessCanvas.width = targetWidth
  preprocessCanvas.height = targetHeight

  const context = get2dContext(preprocessCanvas)
  const geometry = drawPreprocessedImage(context, sourceCanvas, contract)

  const imageData = context.getImageData(0, 0, targetWidth, targetHeight)
  const tensorData = new Float32Array(3 * targetWidth * targetHeight)
  const channelSize = targetWidth * targetHeight

  for (let index = 0; index < channelSize; index += 1) {
    const pixelOffset = index * 4
    tensorData[index] = imageData.data[pixelOffset] / 255
    tensorData[channelSize + index] = imageData.data[pixelOffset + 1] / 255
    tensorData[channelSize * 2 + index] = imageData.data[pixelOffset + 2] / 255
  }

  return {
    tensor: new ort.Tensor('float32', tensorData, [1, 3, targetHeight, targetWidth]),
    geometry,
  }
}

function ensureNumericTensorData(data: ort.Tensor['data'], outputName: string): ArrayLike<number> {
  if (Array.isArray(data)) {
    throw new Error(`输出 ${outputName} 不是数值 tensor。`)
  }

  return data as ArrayLike<number>
}

function drawDetections(
  sourceCanvas: HTMLCanvasElement,
  detections: RecognitionResult['detections'],
  targetCanvas?: HTMLCanvasElement,
): HTMLCanvasElement {
  const canvas = targetCanvas ?? document.createElement('canvas')
  canvas.width = sourceCanvas.width
  canvas.height = sourceCanvas.height
  const context = get2dContext(canvas)
  context.drawImage(sourceCanvas, 0, 0)
  context.lineWidth = 3
  context.font = "15px 'Cascadia Code', 'Consolas', monospace"
  context.textBaseline = 'top'

  detections.forEach((item, index) => {
    const color = pickColor(index)
    const label = `${item.label} ${(item.confidence * 100).toFixed(1)}%`
    const metrics = context.measureText(label)
    const textWidth = metrics.width + 14
    const textHeight = 26
    const textY = item.box.y > textHeight ? item.box.y - textHeight : item.box.y

    context.strokeStyle = color
    context.fillStyle = color
    context.strokeRect(item.box.x, item.box.y, item.box.width, item.box.height)
    context.fillRect(item.box.x, textY, textWidth, textHeight)
    context.fillStyle = '#f8f5ed'
    context.fillText(label, item.box.x + 7, textY + 5)
  })

  return canvas
}

function get2dContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error('浏览器不支持 Canvas 2D。')
  }

  return context
}

function drawPreprocessedImage(
  context: CanvasRenderingContext2D,
  sourceCanvas: HTMLCanvasElement,
  contract: ResolvedModelContract,
): PreprocessedImageGeometry {
  const targetWidth = contract.preprocess.imageWidth
  const targetHeight = contract.preprocess.imageHeight
  const sourceWidth = sourceCanvas.width
  const sourceHeight = sourceCanvas.height
  context.clearRect(0, 0, targetWidth, targetHeight)
  context.fillStyle = '#000'
  context.fillRect(0, 0, targetWidth, targetHeight)

  if (contract.preprocess.resizeMode === 'pad') {
    const scale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight)
    const contentWidth = sourceWidth * scale
    const contentHeight = sourceHeight * scale
    const padLeft = (targetWidth - contentWidth) / 2
    const padTop = (targetHeight - contentHeight) / 2

    context.drawImage(sourceCanvas, padLeft, padTop, contentWidth, contentHeight)

    return {
      sourceWidth,
      sourceHeight,
      targetWidth,
      targetHeight,
      resizeMode: 'pad',
      scaleX: contentWidth / sourceWidth,
      scaleY: contentHeight / sourceHeight,
      padLeft,
      padTop,
      contentWidth,
      contentHeight,
    }
  }

  context.drawImage(sourceCanvas, 0, 0, targetWidth, targetHeight)

  return {
    sourceWidth,
    sourceHeight,
    targetWidth,
    targetHeight,
    resizeMode: 'stretch',
    scaleX: targetWidth / sourceWidth,
    scaleY: targetHeight / sourceHeight,
    padLeft: 0,
    padTop: 0,
    contentWidth: targetWidth,
    contentHeight: targetHeight,
  }
}

function resolveSourceDimensions(source: CanvasImageSource): { width: number; height: number } {
  if (source instanceof HTMLVideoElement) {
    return {
      width: source.videoWidth || source.clientWidth,
      height: source.videoHeight || source.clientHeight,
    }
  }

  if (source instanceof HTMLCanvasElement) {
    return { width: source.width, height: source.height }
  }

  if (source instanceof ImageBitmap) {
    return { width: source.width, height: source.height }
  }

  if (source instanceof HTMLImageElement) {
    return {
      width: source.naturalWidth || source.width,
      height: source.naturalHeight || source.height,
    }
  }

  if (typeof VideoFrame !== 'undefined' && source instanceof VideoFrame) {
    return {
      width: source.displayWidth || source.codedWidth,
      height: source.displayHeight || source.codedHeight,
    }
  }

  throw new Error('无法解析当前帧源的尺寸。')
}

function pickColor(index: number): string {
  const palette = ['#d94f30', '#11698e', '#0f9d58', '#d68910', '#6a4c93', '#2f4858']
  return palette[index % palette.length]
}

function assertWebGpuSupported() {
  const supportState = getWebGpuSupportState()
  if (!supportState.supported) {
    throw new Error(supportState.message ?? WEBGPU_UNSUPPORTED_MESSAGE)
  }
}

function assertModelWebGpuCompatible(model: ImportedModel) {
  if (model.webGpuCompatibility.supported) {
    return
  }

  const issueMessage = model.webGpuCompatibility.issues
    .filter((item) => item.severity === 'error')
    .map((item) => item.message)
    .join(' ')

  throw new Error(issueMessage || '当前模型不兼容 WebGPU，无法执行识别。')
}

function createDetectionContract(
  contract: ResolvedModelContract,
  options?: RunDetectionOptions,
): ResolvedModelContract {
  const hasScoreThresholdOverride = Number.isFinite(options?.scoreThresholdOverride)
  const hasMaxDetectionsOverride = Number.isFinite(options?.maxDetectionsOverride)

  if (!hasScoreThresholdOverride && !hasMaxDetectionsOverride) {
    return contract
  }

  return {
    ...contract,
    decoder: {
      ...contract.decoder,
      scoreThreshold: hasScoreThresholdOverride
        ? clampScoreThreshold(options!.scoreThresholdOverride!)
        : contract.decoder.scoreThreshold,
      maxDetections: hasMaxDetectionsOverride
        ? clampMaxDetections(options!.maxDetectionsOverride!)
        : contract.decoder.maxDetections,
    },
  }
}

function buildNoDetectionsRuntimeMessage(
  contract: ResolvedModelContract,
  outputs: Record<string, TensorLike>,
): string {
  const highestScore = estimateHighestCandidateScore(contract, outputs)
  if (highestScore === null) {
    return '推理已执行完成，但当前没有达到阈值的候选目标。'
  }

  return `推理已执行完成，当前最高候选分数 ${(highestScore * 100).toFixed(1)}%，低于阈值 ${(contract.decoder.scoreThreshold * 100).toFixed(1)}%。`
}

function clampScoreThreshold(value: number): number {
  return Math.min(Math.max(value, 0), 1)
}

function clampMaxDetections(value: number): number {
  if (!Number.isFinite(value)) {
    return 0
  }

  return Math.max(0, Math.trunc(value))
}

function estimateHighestCandidateScore(
  contract: ResolvedModelContract,
  outputs: Record<string, TensorLike>,
): number | null {
  switch (contract.decoder.layoutKind) {
    case 'logits-and-pred-boxes':
      return estimateHighestHfScore(contract, outputs)
    case 'ultralytics-anchors':
      return estimateHighestAnchorScore(contract, outputs)
    case 'ultralytics-queries':
      return estimateHighestQueryScore(contract, outputs)
    default:
      return null
  }
}

function estimateHighestHfScore(
  contract: ResolvedModelContract,
  outputs: Record<string, TensorLike>,
): number | null {
  const logits = resolveTensor(outputs, contract.decoder.outputTensorNames, 'logits')
  const queryCount = logits.dims.at(-2) ?? 0
  const classCount = logits.dims.at(-1) ?? 0
  let bestScore = Number.NEGATIVE_INFINITY

  for (let queryIndex = 0; queryIndex < queryCount; queryIndex += 1) {
    if (classCount === 1) {
      bestScore = Math.max(bestScore, sigmoid(readLogit(logits, queryIndex, 0)))
      continue
    }

    const probabilities = softmax(logits, queryIndex, classCount)
    for (const probability of probabilities) {
      bestScore = Math.max(bestScore, probability)
    }
  }

  return Number.isFinite(bestScore) ? bestScore : null
}

function estimateHighestAnchorScore(
  contract: ResolvedModelContract,
  outputs: Record<string, TensorLike>,
): number | null {
  const output = resolveTensor(outputs, contract.decoder.outputTensorNames, 'output0')
  const channelCount = output.dims[1] ?? 0
  const anchorCount = output.dims[2] ?? 0
  const classCount = Math.max(1, channelCount - 4)
  let bestScore = Number.NEGATIVE_INFINITY

  for (let anchorIndex = 0; anchorIndex < anchorCount; anchorIndex += 1) {
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      bestScore = Math.max(bestScore, normalizeScore(readAnchor(output, classIndex + 4, anchorIndex)))
    }
  }

  return Number.isFinite(bestScore) ? bestScore : null
}

function estimateHighestQueryScore(
  contract: ResolvedModelContract,
  outputs: Record<string, TensorLike>,
): number | null {
  const output = resolveTensor(outputs, contract.decoder.outputTensorNames, 'output0')
  const queryCount = output.dims[1] ?? 0
  const vectorLength = output.dims[2] ?? 0
  const classCount = Math.max(1, vectorLength - 4)
  let bestScore = Number.NEGATIVE_INFINITY

  for (let queryIndex = 0; queryIndex < queryCount; queryIndex += 1) {
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      bestScore = Math.max(bestScore, normalizeScore(readQuery(output, queryIndex, classIndex + 4)))
    }
  }

  return Number.isFinite(bestScore) ? bestScore : null
}

function resolveTensor(
  outputs: Record<string, TensorLike>,
  preferredNames: string[],
  expectedName: string,
): TensorLike {
  const normalizedExpectedName = expectedName.toLowerCase()
  const exactPreferredNames = preferredNames.filter((name) => name.toLowerCase() === normalizedExpectedName)

  for (const name of exactPreferredNames) {
    const tensor = outputs[name]
    if (tensor) {
      return tensor
    }
  }

  const exactKey = Object.keys(outputs).find((item) => item.toLowerCase() === normalizedExpectedName)
  if (exactKey) {
    return outputs[exactKey]
  }

  const partialPreferredName = preferredNames.find((name) => name.toLowerCase().includes(normalizedExpectedName))
  if (partialPreferredName && outputs[partialPreferredName]) {
    return outputs[partialPreferredName]
  }

  const partialKey = Object.keys(outputs).find((item) => item.toLowerCase().includes(normalizedExpectedName))
  if (partialKey) {
    return outputs[partialKey]
  }

  if (preferredNames.length === 1 && outputs[preferredNames[0]]) {
    return outputs[preferredNames[0]]
  }

  throw new Error(`模型输出中未找到 ${expectedName}`)
}

function readLogit(tensor: TensorLike, queryIndex: number, classIndex: number): number {
  const classCount = tensor.dims.at(-1) ?? 0
  return Number(tensor.data[queryIndex * classCount + classIndex] ?? 0)
}

function readAnchor(tensor: TensorLike, channelIndex: number, anchorIndex: number): number {
  const anchorCount = tensor.dims[2] ?? 0
  return Number(tensor.data[channelIndex * anchorCount + anchorIndex] ?? 0)
}

function readQuery(tensor: TensorLike, queryIndex: number, itemIndex: number): number {
  const vectorLength = tensor.dims[2] ?? 0
  return Number(tensor.data[queryIndex * vectorLength + itemIndex] ?? 0)
}

function softmax(tensor: TensorLike, queryIndex: number, classCount: number): number[] {
  const values = new Array<number>(classCount).fill(0)
  let maxValue = Number.NEGATIVE_INFINITY

  for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
    const current = readLogit(tensor, queryIndex, classIndex)
    if (current > maxValue) {
      maxValue = current
    }
  }

  let sum = 0
  for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
    values[classIndex] = Math.exp(readLogit(tensor, queryIndex, classIndex) - maxValue)
    sum += values[classIndex]
  }

  if (sum <= 0) {
    return values
  }

  return values.map((item) => item / sum)
}
