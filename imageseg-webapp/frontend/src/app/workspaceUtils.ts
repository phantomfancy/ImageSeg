import {
  CONFIG_FILE_NAME,
  PREPROCESSOR_CONFIG_FILE_NAME,
} from '../lib/modelPackage'
import type { DetectionRun, ImportedModel } from '../lib/onnxRuntime'
import type {
  CameraDeviceOption,
  DetectionItem,
  InputMode,
  PreviewViewerOffset,
  PreviewZoomTarget,
  ResolvedTheme,
  ThemeMode,
  StreamState,
} from './types'

export const DEFAULT_DETECTION_THRESHOLD = 0.8
export const DEFAULT_MAX_DETECTIONS = 0
export const FPS_WINDOW_MS = 1000
export const PREVIEW_VIEWER_DEFAULT_SCALE = 1
export const PREVIEW_VIEWER_MIN_SCALE = 0.25
export const PREVIEW_VIEWER_MAX_SCALE = 4
export const PREVIEW_VIEWER_WHEEL_STEP = 0.2
export const PREVIEW_VIEWER_DEFAULT_OFFSET: PreviewViewerOffset = { x: 0, y: 0 }

export function formatModelReadyMessage(
  model: ImportedModel,
  webGpuSupportState: { supported: boolean; message?: string },
): string {
  const sidecars = [
    model.sidecars.configFileName,
    model.sidecars.preprocessorConfigFileName,
  ].filter(Boolean)
  const runtimeNote = !webGpuSupportState.supported && webGpuSupportState.message
    ? ` ${webGpuSupportState.message}`
    : ''

  return sidecars.length > 0
    ? `模型解析成功：${model.contract.family}，已使用 ${sidecars.join('、')}，当前执行提供器为 ${model.providerName}。${runtimeNote}`
    : `模型解析成功：${model.contract.family}，当前执行提供器为 ${model.providerName}。${runtimeNote}`
}

export function formatAutoDiscoveredMessage(
  model: ImportedModel,
  discovered: { configFile?: File; preprocessorConfigFile?: File },
  webGpuSupportState: { supported: boolean; message?: string },
): string {
  const autoFiles = [
    discovered.configFile?.name,
    discovered.preprocessorConfigFile?.name,
  ].filter(Boolean)

  return autoFiles.length > 0
    ? `已自动找到 ${autoFiles.join('、')}，${formatModelReadyMessage(model, webGpuSupportState)}`
    : formatModelReadyMessage(model, webGpuSupportState)
}

export function buildPendingHfMessage(
  configFile: File | null,
  preprocessorConfigFile: File | null,
  supportsDirectoryPicker: boolean,
): string {
  if (!configFile && !preprocessorConfigFile) {
    return supportsDirectoryPicker
      ? `已识别为 Hugging Face 风格模型，请导入 ${CONFIG_FILE_NAME}；${PREPROCESSOR_CONFIG_FILE_NAME} 可选，或使用“自动查找同目录配置”。`
      : `已识别为 Hugging Face 风格模型，请导入 ${CONFIG_FILE_NAME}；${PREPROCESSOR_CONFIG_FILE_NAME} 可选。`
  }

  if (!configFile && preprocessorConfigFile) {
    return `已导入 ${PREPROCESSOR_CONFIG_FILE_NAME}，但仍缺少必选 ${CONFIG_FILE_NAME}。`
  }

  return `已导入 ${CONFIG_FILE_NAME}，${PREPROCESSOR_CONFIG_FILE_NAME} 可继续补充。`
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === 'AbortError'
    : Boolean(
      error &&
      typeof error === 'object' &&
      'name' in error &&
      error.name === 'AbortError',
    )
}

export function formatDims(dims: Array<number | string | null>): string {
  return `[${dims.map((item) => item ?? '?').join(', ')}]`
}

export function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export function toDetectionItems(run: DetectionRun): DetectionItem[] {
  return run.recognitionResult.detections.map((item) => ({
    label: item.label,
    confidence: item.confidence,
    boxSummary: `${item.box.x.toFixed(1)}, ${item.box.y.toFixed(1)}, ${item.box.width.toFixed(1)}, ${item.box.height.toFixed(1)}`,
  }))
}

export function clearCanvas(canvas: HTMLCanvasElement | null) {
  if (!canvas) {
    return
  }

  const context = canvas.getContext('2d')
  if (!context) {
    return
  }

  context.clearRect(0, 0, canvas.width, canvas.height)
  canvas.width = 0
  canvas.height = 0
}

export function syncCanvas(source: HTMLCanvasElement, target: HTMLCanvasElement | null) {
  if (!target || source === target) {
    return
  }

  target.width = source.width
  target.height = source.height
  const context = target.getContext('2d')
  if (!context) {
    return
  }

  context.drawImage(source, 0, 0)
}

export function drawRuntimeOverlay(canvas: HTMLCanvasElement | null, fps: number | null) {
  if (!canvas || fps === null) {
    return
  }

  const context = canvas.getContext('2d')
  if (!context) {
    return
  }

  const label = `FPS ${fps.toFixed(1)}`
  context.save()
  context.font = "15px 'Cascadia Code', 'Consolas', monospace"
  context.textBaseline = 'top'
  const metrics = context.measureText(label)
  const textWidth = metrics.width + 16
  const textHeight = 28

  context.fillStyle = 'rgba(20, 32, 42, 0.78)'
  context.fillRect(14, 14, textWidth, textHeight)
  context.fillStyle = '#f8f5ed'
  context.fillText(label, 22, 20)
  context.restore()
}

export function calculateNextFps(frameTimestamps: number[], timestamp: number): number {
  frameTimestamps.push(timestamp)
  while (frameTimestamps.length > 0 && timestamp - frameTimestamps[0] > FPS_WINDOW_MS) {
    frameTimestamps.shift()
  }

  if (frameTimestamps.length <= 1) {
    return 0
  }

  const elapsedMs = frameTimestamps[frameTimestamps.length - 1] - frameTimestamps[0]
  if (elapsedMs <= 0) {
    return 0
  }

  return Number((((frameTimestamps.length - 1) * 1000) / elapsedMs).toFixed(1))
}

export function clampDetectionThreshold(value: number): number {
  if (!Number.isFinite(value)) {
    return DEFAULT_DETECTION_THRESHOLD
  }

  return Math.min(Math.max(value, 0), 1)
}

export function normalizeMaxDetectionsValue(value: number | string): number {
  const parsedValue = typeof value === 'string'
    ? Number(value.trim())
    : value

  if (!Number.isFinite(parsedValue)) {
    return DEFAULT_MAX_DETECTIONS
  }

  return Math.max(0, Math.trunc(parsedValue))
}

export function resolvePreviewZoomTarget(input: {
  hasRenderedResult: boolean
  imagePreviewUrl: string
  inputMode: InputMode
  streamState: StreamState
  videoPreviewUrl: string
}): PreviewZoomTarget | null {
  if (input.hasRenderedResult) {
    return 'result-canvas'
  }

  if (input.inputMode === 'image' && input.imagePreviewUrl) {
    return 'image'
  }

  if (input.inputMode === 'video' && input.videoPreviewUrl) {
    return 'video'
  }

  if (input.inputMode === 'camera' && input.streamState === 'running') {
    return 'camera'
  }

  return null
}

export function readThemeMode(): ThemeMode {
  if (typeof window === 'undefined') {
    return 'system'
  }

  const storedValue = window.localStorage.getItem('4cimageseg-theme-mode')
  return storedValue === 'light' || storedValue === 'dark' || storedValue === 'system'
    ? storedValue
    : 'system'
}

export function getSystemTheme(): ResolvedTheme {
  if (typeof window === 'undefined') {
    return 'light'
  }

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function resolveThemeMode(
  themeMode: ThemeMode,
  systemTheme: ResolvedTheme,
): ResolvedTheme {
  return themeMode === 'system' ? systemTheme : themeMode
}

export function isPreviewScalableTarget(target: PreviewZoomTarget): boolean {
  return target === 'image' || target === 'result-canvas'
}

export function clampPreviewZoomScale(value: number): number {
  return Math.min(Math.max(value, PREVIEW_VIEWER_MIN_SCALE), PREVIEW_VIEWER_MAX_SCALE)
}

export function clampPreviewViewerOffset(
  value: PreviewViewerOffset,
  scale: number,
  viewport: HTMLDivElement | null,
  media: HTMLDivElement | null,
): PreviewViewerOffset {
  if (!viewport || !media || scale <= PREVIEW_VIEWER_DEFAULT_SCALE) {
    return PREVIEW_VIEWER_DEFAULT_OFFSET
  }

  const scaledWidth = media.offsetWidth * scale
  const scaledHeight = media.offsetHeight * scale
  const maxOffsetX = Math.max(0, (scaledWidth - viewport.clientWidth) / 2)
  const maxOffsetY = Math.max(0, (scaledHeight - viewport.clientHeight) / 2)

  return {
    x: Math.min(Math.max(value.x, -maxOffsetX), maxOffsetX),
    y: Math.min(Math.max(value.y, -maxOffsetY), maxOffsetY),
  }
}

export async function ensureVideoReady(videoElement: HTMLVideoElement): Promise<void> {
  if (videoElement.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && videoElement.videoWidth > 0) {
    return
  }

  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      videoElement.removeEventListener('loadeddata', handleLoadedData)
      videoElement.removeEventListener('error', handleError)
    }

    const handleLoadedData = () => {
      cleanup()
      resolve()
    }

    const handleError = () => {
      cleanup()
      reject(videoElement.error ?? new Error('无法加载视频。'))
    }

    videoElement.addEventListener('loadeddata', handleLoadedData, { once: true })
    videoElement.addEventListener('error', handleError, { once: true })
    videoElement.load()
  })
}

export function requestNextVideoFrame(
  videoElement: HTMLVideoElement,
  callback: () => void,
) {
  if ('requestVideoFrameCallback' in videoElement) {
    videoElement.requestVideoFrameCallback(() => {
      callback()
    })
    return
  }

  window.requestAnimationFrame(() => {
    callback()
  })
}

export async function downloadCanvasAsPng(
  canvas: HTMLCanvasElement,
  fileName: string,
): Promise<void> {
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((item) => {
      if (!item) {
        reject(new Error('无法生成 PNG 数据。'))
        return
      }
      resolve(item)
    }, 'image/png')
  })

  const downloadUrl = URL.createObjectURL(blob)
  triggerDownload(downloadUrl, fileName)
  window.setTimeout(() => {
    URL.revokeObjectURL(downloadUrl)
  }, 0)
}

function triggerDownload(url: string, fileName: string) {
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = fileName
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
}

export function buildImageExportFileName(fileName: string): string {
  return `${stripExtension(fileName)}-detected.png`
}

export function buildVideoExportFileName(fileName: string): string {
  return `${stripExtension(fileName)}-detected.webm`
}

function stripExtension(fileName: string): string {
  return fileName.replace(/\.[^.]+$/, '')
}

export function buildFrameSourceLabel(
  sourceKind: 'video' | 'camera',
  currentTime: number,
  videoFileName: string | undefined,
  selectedCameraId: string,
  cameraDevices: CameraDeviceOption[],
): string {
  if (sourceKind === 'video') {
    return `${videoFileName ?? 'video'}@${currentTime.toFixed(2)}s`
  }

  const cameraLabel = cameraDevices.find((item) => item.deviceId === selectedCameraId)?.label ?? 'camera'
  return `${cameraLabel}@${new Date().toISOString()}`
}

export function getPreferredVideoMimeType(): string | null {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') {
    return null
  }

  const candidates = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ]

  return candidates.find((item) => MediaRecorder.isTypeSupported(item)) ?? null
}
