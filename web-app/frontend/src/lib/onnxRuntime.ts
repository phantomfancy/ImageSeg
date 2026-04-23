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

const WEBGPU_UNSUPPORTED_MESSAGE = '当前浏览器不支持 WebGPU，无法执行识别。'

type SessionState = {
  client: RuntimeWorkerClient
  providerName: 'webgpu'
  inputName: string
  outputNames: readonly string[]
  preprocessCanvas: HTMLCanvasElement
  preprocessContext: CanvasRenderingContext2D
  inputData: Float32Array
}

export type RuntimeWorkerInitOptions = {
  modelKey: string
  bytes: Uint8Array
  preferredInputName: string
  inputDims: readonly [number, number, number, number]
  enableGraphCapture: true
}

export type RuntimeWorkerSessionMetadata = {
  inputName: string
  outputNames: readonly string[]
}

export interface RuntimeWorkerClient {
  init(options: RuntimeWorkerInitOptions): Promise<RuntimeWorkerSessionMetadata>
  run(inputData: Float32Array, fetchNames: readonly string[]): Promise<Record<string, TensorLike>>
  release(): Promise<void>
  terminate(): void
}

export interface WebGpuRuntimeHooks {
  isWebGpuSupported?: () => boolean
  createCanvas?: () => HTMLCanvasElement
  createWorkerClient?: () => RuntimeWorkerClient
}

let runtimeHooks: WebGpuRuntimeHooks | null = null
const sessionCache = new Map<string, SessionState>()
let sessionOperationQueue: Promise<void> = Promise.resolve()

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

export function setWebGpuRuntimeHooks(hooks: WebGpuRuntimeHooks | null) {
  runtimeHooks = hooks
}

export async function prepareImportedModelSession(model: ImportedModel): Promise<void> {
  await enqueueSessionOperation(async () => {
    await getSessionState(model)
  })
}

export function getWebGpuSupportState(): WebGpuSupportState {
  if (runtimeHooks?.isWebGpuSupported) {
    if (!runtimeHooks.isWebGpuSupported()) {
      return {
        supported: false,
        message: WEBGPU_UNSUPPORTED_MESSAGE,
      }
    }

    return { supported: true }
  }

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

export async function releaseCachedModelSessions(retainModelKey?: string): Promise<void> {
  await enqueueSessionOperation(async () => {
    const releaseStates = [...sessionCache.entries()]
      .filter(([modelKey]) => modelKey !== retainModelKey)
      .map(([modelKey, sessionState]) => {
        sessionCache.delete(modelKey)
        return sessionState
      })

    for (const sessionState of releaseStates) {
      await releaseSessionState(sessionState)
    }
  })
}

export async function runSingleImageDetection(
  model: ImportedModel,
  imageFile: File,
  options?: RunDetectionOptions,
): Promise<DetectionRun> {
  const sourceCanvas = await loadImageCanvas(imageFile)
  return runDetectionOnSource(model, sourceCanvas, imageFile.name, options)
}

export async function runDetectionOnCanvas(
  model: ImportedModel,
  sourceCanvas: HTMLCanvasElement,
  inputSource: string,
  options?: RunDetectionOptions,
  targetCanvas?: HTMLCanvasElement,
): Promise<DetectionRun> {
  return runDetectionOnSource(model, sourceCanvas, inputSource, options, targetCanvas)
}

export async function runDetectionOnSource(
  model: ImportedModel,
  source: CanvasImageSource,
  inputSource: string,
  options?: RunDetectionOptions,
  targetCanvas?: HTMLCanvasElement,
): Promise<DetectionRun> {
  return enqueueSessionOperation(async () => {
    const sessionState = await getSessionState(model)
    const detectionContract = createDetectionContract(model.contract, options)
    const annotatedCanvas = captureSourceFrame(source, targetCanvas)
    const geometry = prepareInput(sessionState, source, detectionContract)
    const fetchNames = resolveRequestedOutputNames(sessionState.outputNames, detectionContract.decoder.outputTensorNames)
    const normalizedOutputs = await sessionState.client.run(sessionState.inputData, fetchNames)
    const detections = decodeDetections(
      detectionContract,
      normalizedOutputs,
      geometry,
    )

    drawDetections(annotatedCanvas, detections)

    return {
      providerName: sessionState.providerName,
      annotatedCanvas,
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
  })
}

async function getSessionState(model: ImportedModel): Promise<SessionState> {
  assertWebGpuSupported()
  const existing = sessionCache.get(model.key)
  if (existing) {
    return existing
  }

  await releaseStaleSessionStates(model.key)
  const sessionState = await createSessionState(model)
  sessionCache.set(model.key, sessionState)
  return sessionState
}

function enqueueSessionOperation<T>(operation: () => Promise<T>): Promise<T> {
  const result = sessionOperationQueue.then(operation, operation)
  sessionOperationQueue = result.then(
    () => undefined,
    () => undefined,
  )
  return result
}

async function releaseStaleSessionStates(retainModelKey: string) {
  const staleSessionStates = [...sessionCache.entries()]
    .filter(([modelKey]) => modelKey !== retainModelKey)
    .map(([modelKey, sessionState]) => {
      sessionCache.delete(modelKey)
      return sessionState
    })

  for (const sessionState of staleSessionStates) {
    await releaseSessionState(sessionState)
  }
}

async function createSessionState(model: ImportedModel): Promise<SessionState> {
  const inputWidth = model.contract.preprocess.imageWidth
  const inputHeight = model.contract.preprocess.imageHeight
  const inputData = new Float32Array(3 * inputWidth * inputHeight)
  const inputDims = [1, 3, inputHeight, inputWidth] as const
  const preprocessCanvas = createCanvas()
  preprocessCanvas.width = inputWidth
  preprocessCanvas.height = inputHeight
  const preprocessContext = get2dContext(preprocessCanvas, { willReadFrequently: true })
  const client = await createRuntimeWorkerClient()

  try {
    const metadata = await client.init({
      modelKey: model.key,
      bytes: model.bytes,
      preferredInputName: model.contract.preprocess.inputTensorName,
      inputDims,
      enableGraphCapture: true,
    })

    return {
      client,
      providerName: 'webgpu',
      inputName: metadata.inputName,
      outputNames: metadata.outputNames,
      preprocessCanvas,
      preprocessContext,
      inputData,
    }
  } catch (error) {
    client.terminate()
    throw error
  }
}

async function releaseSessionState(sessionState: SessionState) {
  try {
    await sessionState.client.release()
  } finally {
    sessionState.client.terminate()
  }
}

function createCanvas(): HTMLCanvasElement {
  if (runtimeHooks?.createCanvas) {
    return runtimeHooks.createCanvas()
  }

  if (typeof document === 'undefined' || typeof document.createElement !== 'function') {
    throw new Error('当前环境不支持创建 Canvas。')
  }

  return document.createElement('canvas')
}

async function createRuntimeWorkerClient(): Promise<RuntimeWorkerClient> {
  if (runtimeHooks?.createWorkerClient) {
    return runtimeHooks.createWorkerClient()
  }

  const { default: OnnxRuntimeWorker } = await import('./onnxRuntimeWorkerFactory.ts')
  return new BrowserRuntimeWorkerClient(new OnnxRuntimeWorker())
}

type WorkerRequest =
  | (RuntimeWorkerInitOptions & { type: 'init'; requestId: number })
  | { type: 'run'; requestId: number; inputData: Float32Array; fetchNames: readonly string[] }
  | { type: 'release'; requestId: number }

type WorkerResponse =
  | { type: 'success'; requestId: number; result: unknown }
  | { type: 'error'; requestId: number; message: string }

class BrowserRuntimeWorkerClient implements RuntimeWorkerClient {
  private readonly worker: Worker
  private readonly pendingRequests = new Map<number, {
    resolve: (value: unknown) => void
    reject: (error: Error) => void
  }>()
  private nextRequestId = 1
  private terminated = false

  constructor(worker: Worker) {
    this.worker = worker

    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const pendingRequest = this.pendingRequests.get(event.data.requestId)
      if (!pendingRequest) {
        return
      }

      this.pendingRequests.delete(event.data.requestId)
      if (event.data.type === 'error') {
        pendingRequest.reject(new Error(event.data.message))
        return
      }

      pendingRequest.resolve(event.data.result)
    }

    this.worker.onerror = (event: ErrorEvent) => {
      this.rejectAll(new Error(event.message || 'ONNX Runtime Worker 执行失败。'))
    }

    this.worker.onmessageerror = () => {
      this.rejectAll(new Error('ONNX Runtime Worker 消息传递失败。'))
    }
  }

  init(options: RuntimeWorkerInitOptions): Promise<RuntimeWorkerSessionMetadata> {
    const bytes = options.bytes.slice()
    return this.post<RuntimeWorkerSessionMetadata>({
      type: 'init',
      requestId: this.nextRequestId++,
      ...options,
      bytes,
    }, [bytes.buffer])
  }

  run(inputData: Float32Array, fetchNames: readonly string[]): Promise<Record<string, TensorLike>> {
    const inputDataCopy = inputData.slice()
    return this.post<Record<string, TensorLike>>({
      type: 'run',
      requestId: this.nextRequestId++,
      inputData: inputDataCopy,
      fetchNames: [...fetchNames],
    }, [inputDataCopy.buffer])
  }

  release(): Promise<void> {
    if (this.terminated) {
      return Promise.resolve()
    }

    return this.post<void>({
      type: 'release',
      requestId: this.nextRequestId++,
    }, [])
  }

  terminate() {
    if (this.terminated) {
      return
    }

    this.terminated = true
    this.worker.terminate()
    this.rejectAll(new Error('ONNX Runtime Worker 已终止。'))
  }

  private post<T>(request: WorkerRequest, transfers: Transferable[]): Promise<T> {
    if (this.terminated) {
      return Promise.reject(new Error('ONNX Runtime Worker 已终止。'))
    }

    return new Promise<T>((resolve, reject) => {
      this.pendingRequests.set(request.requestId, {
        resolve: (value) => resolve(value as T),
        reject,
      })
      try {
        this.worker.postMessage(request, transfers)
      } catch (error) {
        this.pendingRequests.delete(request.requestId)
        reject(error instanceof Error ? error : new Error(String(error)))
      }
    })
  }

  private rejectAll(error: Error) {
    for (const pendingRequest of this.pendingRequests.values()) {
      pendingRequest.reject(error)
    }
    this.pendingRequests.clear()
  }
}

function resolveRequestedOutputNames(
  outputNames: readonly string[],
  preferredNames: readonly string[],
): string[] {
  if (preferredNames.length === 0) {
    return [...outputNames]
  }

  const resolvedNames: string[] = []
  for (const preferredName of preferredNames) {
    const resolvedName = resolveTensorName(outputNames, preferredName, '输出')
    if (!resolvedNames.includes(resolvedName)) {
      resolvedNames.push(resolvedName)
    }
  }

  return resolvedNames
}

function resolveTensorName(
  actualNames: readonly string[],
  preferredName: string,
  kind: '输入' | '输出',
): string {
  if (preferredName) {
    const normalizedPreferredName = preferredName.toLowerCase()
    const exactMatch = actualNames.find((name) => name.toLowerCase() === normalizedPreferredName)
    if (exactMatch) {
      return exactMatch
    }

    const partialMatch = actualNames.find((name) => name.toLowerCase().includes(normalizedPreferredName))
    if (partialMatch) {
      return partialMatch
    }
  }

  if (actualNames.length === 1) {
    return actualNames[0]
  }

  throw new Error(`无法在模型${kind}中匹配 ${preferredName || '目标名称'}。`)
}

function prepareInput(
  sessionState: SessionState,
  source: CanvasImageSource,
  contract: ResolvedModelContract,
): PreprocessedImageGeometry {
  const geometry = drawPreprocessedImage(sessionState.preprocessContext, source, contract)
  const targetWidth = contract.preprocess.imageWidth
  const targetHeight = contract.preprocess.imageHeight
  const imageData = sessionState.preprocessContext.getImageData(0, 0, targetWidth, targetHeight)
  writeTensorData(sessionState.inputData, imageData.data, targetWidth * targetHeight)

  return geometry
}

function writeTensorData(
  target: Float32Array,
  source: Uint8ClampedArray,
  channelSize: number,
) {
  for (let index = 0; index < channelSize; index += 1) {
    const pixelOffset = index * 4
    target[index] = source[pixelOffset] / 255
    target[channelSize + index] = source[pixelOffset + 1] / 255
    target[channelSize * 2 + index] = source[pixelOffset + 2] / 255
  }
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
  return captureSourceFrame(source, targetCanvas)
}

function captureSourceFrame(
  source: CanvasImageSource,
  targetCanvas?: HTMLCanvasElement,
): HTMLCanvasElement {
  const { width, height } = resolveSourceDimensions(source)
  const canvas = targetCanvas ?? document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = get2dContext(canvas)

  if (typeof HTMLCanvasElement !== 'undefined' && source instanceof HTMLCanvasElement && source === canvas) {
    return canvas
  }

  context.clearRect(0, 0, width, height)
  context.drawImage(source, 0, 0, width, height)
  return canvas
}

function drawDetections(
  canvas: HTMLCanvasElement,
  detections: RecognitionResult['detections'],
) {
  const context = get2dContext(canvas)
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
}

function get2dContext(
  canvas: HTMLCanvasElement,
  options?: CanvasRenderingContext2DSettings,
): CanvasRenderingContext2D {
  const context = canvas.getContext('2d', options)
  if (!context) {
    throw new Error('浏览器不支持 Canvas 2D。')
  }

  return context
}

function drawPreprocessedImage(
  context: CanvasRenderingContext2D,
  source: CanvasImageSource,
  contract: ResolvedModelContract,
): PreprocessedImageGeometry {
  const targetWidth = contract.preprocess.imageWidth
  const targetHeight = contract.preprocess.imageHeight
  const { width: sourceWidth, height: sourceHeight } = resolveSourceDimensions(source)
  context.clearRect(0, 0, targetWidth, targetHeight)
  context.fillStyle = '#000'
  context.fillRect(0, 0, targetWidth, targetHeight)

  if (contract.preprocess.resizeMode === 'pad') {
    const scale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight)
    const contentWidth = sourceWidth * scale
    const contentHeight = sourceHeight * scale
    const padLeft = (targetWidth - contentWidth) / 2
    const padTop = (targetHeight - contentHeight) / 2

    context.drawImage(source, padLeft, padTop, contentWidth, contentHeight)

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

  context.drawImage(source, 0, 0, targetWidth, targetHeight)

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
  if (typeof HTMLVideoElement !== 'undefined' && source instanceof HTMLVideoElement) {
    return {
      width: source.videoWidth || source.clientWidth,
      height: source.videoHeight || source.clientHeight,
    }
  }

  if (typeof HTMLCanvasElement !== 'undefined' && source instanceof HTMLCanvasElement) {
    return { width: source.width, height: source.height }
  }

  if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) {
    return { width: source.width, height: source.height }
  }

  if (typeof HTMLImageElement !== 'undefined' && source instanceof HTMLImageElement) {
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
