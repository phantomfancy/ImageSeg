import { startTransition, useCallback, useEffect, useRef, useState } from 'react'
import type { DetectionRun, ImportedModel, RunDetectionOptions } from './lib/onnxRuntime'
import {
  drawSourceToCanvas,
  getWebGpuSupportState,
  inspectImportedModel,
  runDetectionOnCanvas,
  runSingleImageDetection,
} from './lib/onnxRuntime'
import { deriveModelImportControls } from './lib/modelImportState'
import {
  CONFIG_FILE_NAME,
  PREPROCESSOR_CONFIG_FILE_NAME,
  discoverHfSidecarsFromDirectory,
  inspectOnnxModelFile,
  validateSidecarFile,
  type InspectedOnnxModel,
} from './lib/modelPackage'
import './App.css'

type InputMode = 'image' | 'video' | 'camera'
type StreamState = 'idle' | 'running' | 'stopping' | 'exporting'

type DetectionItem = {
  label: string
  confidence: number
  boxSummary: string
}

type CameraDeviceOption = {
  deviceId: string
  label: string
}

const DEFAULT_DETECTION_THRESHOLD = 0.35
const DEFAULT_MAX_DETECTIONS = 0
const FPS_WINDOW_MS = 1000
const THRESHOLD_COMMIT_KEYS = new Set([
  'ArrowLeft',
  'ArrowRight',
  'ArrowUp',
  'ArrowDown',
  'Home',
  'End',
  'PageUp',
  'PageDown',
])

function App() {
  const [inputMode, setInputMode] = useState<InputMode>('image')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreviewUrl, setImagePreviewUrl] = useState('')
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoPreviewUrl, setVideoPreviewUrl] = useState('')
  const [onnxModelDraft, setOnnxModelDraft] = useState<InspectedOnnxModel | null>(null)
  const [configFile, setConfigFile] = useState<File | null>(null)
  const [preprocessorConfigFile, setPreprocessorConfigFile] = useState<File | null>(null)
  const [importedModel, setImportedModel] = useState<ImportedModel | null>(null)
  const [statusMessage, setStatusMessage] = useState('请先导入输入源和 ONNX 模型。')
  const [runtimeMessage, setRuntimeMessage] = useState('')
  const [modelBusy, setModelBusy] = useState(false)
  const [discoverBusy, setDiscoverBusy] = useState(false)
  const [imageDetectBusy, setImageDetectBusy] = useState(false)
  const [streamState, setStreamState] = useState<StreamState>('idle')
  const [cameraBusy, setCameraBusy] = useState(false)
  const [cameraDevices, setCameraDevices] = useState<CameraDeviceOption[]>([])
  const [selectedCameraId, setSelectedCameraId] = useState('')
  const [videoDownloadUrl, setVideoDownloadUrl] = useState('')
  const [resultProvider, setResultProvider] = useState('')
  const [currentSourceTime, setCurrentSourceTime] = useState<number | null>(null)
  const [hasRenderedResult, setHasRenderedResult] = useState(false)
  const [onnxInputKey, setOnnxInputKey] = useState(0)
  const [configInputKey, setConfigInputKey] = useState(0)
  const [preprocessorInputKey, setPreprocessorInputKey] = useState(0)
  const [detectionItems, setDetectionItems] = useState<DetectionItem[]>([])
  const [pendingDetectionThreshold, setPendingDetectionThreshold] = useState(DEFAULT_DETECTION_THRESHOLD)
  const [appliedDetectionThreshold, setAppliedDetectionThreshold] = useState(DEFAULT_DETECTION_THRESHOLD)
  const [pendingMaxDetectionsInput, setPendingMaxDetectionsInput] = useState(String(DEFAULT_MAX_DETECTIONS))
  const [appliedMaxDetections, setAppliedMaxDetections] = useState(DEFAULT_MAX_DETECTIONS)
  const [streamFps, setStreamFps] = useState<number | null>(null)

  const sourceVideoRef = useRef<HTMLVideoElement | null>(null)
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null)
  const resultCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const workingCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const cameraStreamRef = useRef<MediaStream | null>(null)
  const frameLoopTokenRef = useRef(0)
  const frameInFlightRef = useRef(false)
  const frameTimestampsRef = useRef<number[]>([])

  useEffect(() => () => {
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl)
    }
  }, [imagePreviewUrl])

  useEffect(() => () => {
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl)
    }
  }, [videoPreviewUrl])

  useEffect(() => () => {
    if (videoDownloadUrl) {
      URL.revokeObjectURL(videoDownloadUrl)
    }
  }, [videoDownloadUrl])

  const supportsDirectoryPicker =
    typeof window !== 'undefined' &&
    typeof window.showDirectoryPicker === 'function'
  const supportsCamera =
    typeof navigator !== 'undefined' &&
    Boolean(navigator.mediaDevices?.getUserMedia)
  const webGpuSupportState = getWebGpuSupportState()
  const isWebGpuSupported = webGpuSupportState.supported
  const preferredVideoMimeType = getPreferredVideoMimeType()
  const supportsVideoExport = Boolean(preferredVideoMimeType)

  const importControls = deriveModelImportControls(
    onnxModelDraft?.family ?? null,
    supportsDirectoryPicker,
  )
  const displayedContract = importedModel?.contract ?? onnxModelDraft?.draftContract ?? null
  const displayedProvider = importedModel?.providerName ?? '-'
  const displayedSidecars = [
    importedModel?.sidecars.configFileName ?? configFile?.name,
    importedModel?.sidecars.preprocessorConfigFileName ?? preprocessorConfigFile?.name,
  ].filter(Boolean).join(', ') || '-'
  const isImportedModelWebGpuCompatible = importedModel?.webGpuCompatibility.supported ?? true
  const importedModelCompatibilityMessage =
    importedModel && !importedModel.webGpuCompatibility.supported
      ? formatWebGpuCompatibilityMessage(importedModel.webGpuCompatibility)
      : ''
  const displayedRuntimeMessage =
    runtimeMessage ||
    importedModelCompatibilityMessage ||
    (importedModel && !isWebGpuSupported ? (webGpuSupportState.message ?? '') : '')
  const displayedPendingThreshold = pendingDetectionThreshold.toFixed(2)
  const displayedAppliedThreshold = appliedDetectionThreshold.toFixed(2)
  const displayedAppliedMaxDetections = String(appliedMaxDetections)
  const displayedFps = streamFps === null ? '-' : streamFps.toFixed(1)
  const currentTimeLabel = inputMode === 'camera' && streamState === 'running'
    ? 'Live'
    : currentSourceTime === null
      ? '-'
      : `${currentSourceTime.toFixed(2)} s`

  const canRunImage =
    inputMode === 'image' &&
    Boolean(imageFile && importedModel && isImportedModelWebGpuCompatible && isWebGpuSupported && !modelBusy && !imageDetectBusy && streamState === 'idle')
  const canExportImage =
    inputMode === 'image' &&
    hasRenderedResult &&
    !imageDetectBusy &&
    streamState === 'idle'
  const canStartVideo =
    inputMode === 'video' &&
    Boolean(videoFile && importedModel && isImportedModelWebGpuCompatible && isWebGpuSupported && streamState === 'idle')
  const canStopVideo =
    inputMode === 'video' &&
    streamState === 'running'
  const canExportVideo =
    inputMode === 'video' &&
    Boolean(videoFile && importedModel && isImportedModelWebGpuCompatible && isWebGpuSupported && streamState === 'idle' && supportsVideoExport)
  const canStartCamera =
    inputMode === 'camera' &&
    Boolean(importedModel && isImportedModelWebGpuCompatible && isWebGpuSupported && streamState === 'idle' && !cameraBusy && supportsCamera)
  const canStopCamera =
    inputMode === 'camera' &&
    streamState === 'running'
  const canAdjustInferenceSettings = Boolean(
    importedModel &&
    isImportedModelWebGpuCompatible &&
    !modelBusy &&
    streamState !== 'exporting',
  )

  function updateImageFile(nextFile: File | null) {
    stopActiveStream()
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl)
    }

    setImageFile(nextFile)
    setImagePreviewUrl(nextFile ? URL.createObjectURL(nextFile) : '')
    clearRecognitionOutputs()
    setStatusMessage(nextFile ? `已选择图片 ${nextFile.name}。` : '未选择图片。')
  }

  function updateVideoFile(nextFile: File | null) {
    stopActiveStream()
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl)
    }

    setVideoFile(nextFile)
    setVideoPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : '')
    clearRecognitionOutputs()
    setStatusMessage(nextFile ? `已选择视频 ${nextFile.name}。` : '未选择视频。')
  }

  function clearRecognitionOutputs() {
    clearCanvas(resultCanvasRef.current)
    setHasRenderedResult(false)
    setDetectionItems([])
    setResultProvider('')
    setRuntimeMessage('')
    setCurrentSourceTime(null)
    setStreamFps(null)
    frameTimestampsRef.current = []

    if (videoDownloadUrl) {
      URL.revokeObjectURL(videoDownloadUrl)
      setVideoDownloadUrl('')
    }
  }

  function clearImportedModelState() {
    setImportedModel(null)
    resetDetectionThreshold()
    resetMaxDetections()
    clearRecognitionOutputs()
  }

  function resetDetectionThreshold(nextValue: number = DEFAULT_DETECTION_THRESHOLD) {
    const normalizedThreshold = clampDetectionThreshold(nextValue)
    setPendingDetectionThreshold(normalizedThreshold)
    setAppliedDetectionThreshold(normalizedThreshold)
  }

  function commitDetectionThreshold(nextValue: number): number {
    const normalizedThreshold = clampDetectionThreshold(nextValue)
    setPendingDetectionThreshold(normalizedThreshold)
    setAppliedDetectionThreshold((currentValue) =>
      Math.abs(currentValue - normalizedThreshold) < 0.0001 ? currentValue : normalizedThreshold)
    return normalizedThreshold
  }

  function resetMaxDetections(nextValue: number = DEFAULT_MAX_DETECTIONS) {
    const normalizedMaxDetections = normalizeMaxDetectionsValue(nextValue)
    setPendingMaxDetectionsInput(String(normalizedMaxDetections))
    setAppliedMaxDetections(normalizedMaxDetections)
  }

  function commitMaxDetections(nextValue: number | string): number {
    const normalizedMaxDetections = normalizeMaxDetectionsValue(nextValue)
    setPendingMaxDetectionsInput(String(normalizedMaxDetections))
    setAppliedMaxDetections((currentValue) =>
      currentValue === normalizedMaxDetections ? currentValue : normalizedMaxDetections)
    return normalizedMaxDetections
  }

  function commitInferenceSettings(): RunDetectionOptions {
    const scoreThresholdOverride = commitDetectionThreshold(pendingDetectionThreshold)
    const maxDetectionsOverride = commitMaxDetections(pendingMaxDetectionsInput)
    return {
      scoreThresholdOverride,
      maxDetectionsOverride,
    }
  }

  function resetSidecarSelections() {
    setConfigFile(null)
    setPreprocessorConfigFile(null)
    setConfigInputKey((value) => value + 1)
    setPreprocessorInputKey((value) => value + 1)
  }

  const stopActiveStream = useCallback((statusText?: string) => {
    frameLoopTokenRef.current += 1
    frameInFlightRef.current = false

    if (sourceVideoRef.current) {
      sourceVideoRef.current.pause()
    }

    if (cameraVideoRef.current) {
      cameraVideoRef.current.pause()
      cameraVideoRef.current.srcObject = null
    }

    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((track) => track.stop())
      cameraStreamRef.current = null
    }

    startTransition(() => {
      setStreamState('idle')
      setCurrentSourceTime(null)
      setStreamFps(null)
      if (statusText) {
        setStatusMessage(statusText)
      }
    })
    frameTimestampsRef.current = []
  }, [])

  const refreshCameraDevices = useCallback(async () => {
    if (!supportsCamera || typeof navigator.mediaDevices.enumerateDevices !== 'function') {
      setCameraDevices([])
      return
    }

    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      const nextDevices = devices
        .filter((item) => item.kind === 'videoinput')
        .map((item, index) => ({
          deviceId: item.deviceId,
          label: item.label || `摄像头 ${index + 1}`,
        }))

      startTransition(() => {
        setCameraDevices(nextDevices)
        if (!nextDevices.some((item) => item.deviceId === selectedCameraId)) {
          setSelectedCameraId(nextDevices[0]?.deviceId ?? '')
        }
      })
    } catch (error) {
      setStatusMessage(`读取摄像头列表失败：${formatError(error)}`)
    }
  }, [selectedCameraId, supportsCamera])

  useEffect(() => () => {
    stopActiveStream()
  }, [stopActiveStream])

  async function handleOnnxSelection(file: File | null) {
    let shouldResetNativeInput = false
    stopActiveStream()
    clearImportedModelState()
    resetSidecarSelections()
    setOnnxModelDraft(null)

    if (!file) {
      setStatusMessage('尚未选择 ONNX 模型文件。')
      setOnnxInputKey((value) => value + 1)
      return
    }

    setModelBusy(true)
    setStatusMessage(`正在解析 ONNX 模型 ${file.name}...`)

    try {
      const nextOnnxModel = await inspectOnnxModelFile(file)

      if (nextOnnxModel.family === 'hf-detr-like') {
        startTransition(() => {
          setOnnxModelDraft(nextOnnxModel)
          resetDetectionThreshold(nextOnnxModel.draftContract.decoder.scoreThreshold)
          resetMaxDetections()
          setStatusMessage(buildPendingHfMessage(null, null, supportsDirectoryPicker))
        })
        return
      }

      const model = await inspectImportedModel({ onnxModel: nextOnnxModel })
      startTransition(() => {
        setOnnxModelDraft(nextOnnxModel)
        setImportedModel(model)
        resetDetectionThreshold(model.contract.decoder.scoreThreshold)
        resetMaxDetections()
        clearRecognitionOutputs()
        setStatusMessage(formatModelReadyMessage(model, webGpuSupportState))
      })
    } catch (error) {
      shouldResetNativeInput = true
      setStatusMessage(`模型解析失败：${formatError(error)}`)
    } finally {
      setModelBusy(false)
      if (shouldResetNativeInput) {
        setOnnxInputKey((value) => value + 1)
      }
    }
  }

  async function handleConfigSelection(file: File | null) {
    let shouldResetNativeInput = false
    if (!file || !onnxModelDraft || onnxModelDraft.family !== 'hf-detr-like') {
      setConfigInputKey((value) => value + 1)
      return
    }

    try {
      validateSidecarFile(file, 'config')
    } catch (error) {
      setStatusMessage(`config.json 导入失败：${formatError(error)}`)
      setConfigInputKey((value) => value + 1)
      return
    }

    setModelBusy(true)
    setStatusMessage(`正在解析 ${CONFIG_FILE_NAME}...`)

    try {
      const model = await inspectImportedModel({
        onnxModel: onnxModelDraft,
        configFile: file,
        preprocessorConfigFile: preprocessorConfigFile ?? undefined,
      })
      startTransition(() => {
        setConfigFile(file)
        setImportedModel(model)
        resetDetectionThreshold(model.contract.decoder.scoreThreshold)
        resetMaxDetections()
        clearRecognitionOutputs()
        setStatusMessage(formatModelReadyMessage(model, webGpuSupportState))
      })
    } catch (error) {
      shouldResetNativeInput = true
      setStatusMessage(`config.json 导入失败：${formatError(error)}`)
    } finally {
      setModelBusy(false)
      if (shouldResetNativeInput) {
        setConfigInputKey((value) => value + 1)
      }
    }
  }

  async function handlePreprocessorSelection(file: File | null) {
    let shouldResetNativeInput = false
    if (!file || !onnxModelDraft || onnxModelDraft.family !== 'hf-detr-like') {
      setPreprocessorInputKey((value) => value + 1)
      return
    }

    try {
      validateSidecarFile(file, 'preprocessor')
    } catch (error) {
      setStatusMessage(`preprocessor_config.json 导入失败：${formatError(error)}`)
      setPreprocessorInputKey((value) => value + 1)
      return
    }

    if (!configFile) {
      startTransition(() => {
        setPreprocessorConfigFile(file)
        setImportedModel(null)
        resetDetectionThreshold(onnxModelDraft.draftContract.decoder.scoreThreshold)
        resetMaxDetections()
        setStatusMessage(buildPendingHfMessage(null, file, supportsDirectoryPicker))
      })
      return
    }

    setModelBusy(true)
    setStatusMessage(`正在解析 ${PREPROCESSOR_CONFIG_FILE_NAME}...`)

    try {
      const model = await inspectImportedModel({
        onnxModel: onnxModelDraft,
        configFile,
        preprocessorConfigFile: file,
      })
      startTransition(() => {
        setPreprocessorConfigFile(file)
        setImportedModel(model)
        resetDetectionThreshold(model.contract.decoder.scoreThreshold)
        resetMaxDetections()
        clearRecognitionOutputs()
        setStatusMessage(formatModelReadyMessage(model, webGpuSupportState))
      })
    } catch (error) {
      shouldResetNativeInput = true
      setStatusMessage(`preprocessor_config.json 导入失败：${formatError(error)}`)
    } finally {
      setModelBusy(false)
      if (shouldResetNativeInput) {
        setPreprocessorInputKey((value) => value + 1)
      }
    }
  }

  async function handleAutoDiscoverSidecars() {
    if (!onnxModelDraft || onnxModelDraft.family !== 'hf-detr-like') {
      return
    }

    if (typeof window.showDirectoryPicker !== 'function') {
      setStatusMessage('当前浏览器不支持目录授权，请手动导入 JSON 配置文件。')
      return
    }

    setModelBusy(true)
    setDiscoverBusy(true)
    setStatusMessage('正在请求目录权限并查找同目录配置文件...')

    try {
      const directoryHandle = await window.showDirectoryPicker()
      const discovered = await discoverHfSidecarsFromDirectory(directoryHandle)
      const nextConfigFile = discovered.configFile ?? configFile
      const nextPreprocessorConfigFile = discovered.preprocessorConfigFile ?? preprocessorConfigFile

      if (!nextConfigFile && !nextPreprocessorConfigFile) {
        setStatusMessage(
          `未在所选目录中找到 ${CONFIG_FILE_NAME} 或 ${PREPROCESSOR_CONFIG_FILE_NAME}。`,
        )
        return
      }

      if (!nextConfigFile) {
        startTransition(() => {
          if (discovered.preprocessorConfigFile) {
            setPreprocessorConfigFile(discovered.preprocessorConfigFile)
          }
          setStatusMessage(
            `已自动找到 ${PREPROCESSOR_CONFIG_FILE_NAME}，但仍缺少 ${CONFIG_FILE_NAME}。`,
          )
        })
        return
      }

      const model = await inspectImportedModel({
        onnxModel: onnxModelDraft,
        configFile: nextConfigFile,
        preprocessorConfigFile: nextPreprocessorConfigFile ?? undefined,
      })
      startTransition(() => {
        if (discovered.configFile) {
          setConfigFile(discovered.configFile)
        }
        if (discovered.preprocessorConfigFile) {
          setPreprocessorConfigFile(discovered.preprocessorConfigFile)
        }
        setImportedModel(model)
        resetDetectionThreshold(model.contract.decoder.scoreThreshold)
        resetMaxDetections()
        clearRecognitionOutputs()
        setStatusMessage(formatAutoDiscoveredMessage(model, discovered, webGpuSupportState))
      })
    } catch (error) {
      if (isAbortError(error)) {
        setStatusMessage('已取消自动查找同目录配置。')
      } else {
        setStatusMessage(`自动查找失败：${formatError(error)}`)
      }
    } finally {
      setModelBusy(false)
      setDiscoverBusy(false)
    }
  }

  async function handleRunImageDetection() {
    if (!imageFile || !importedModel) {
      return
    }

    if (!isWebGpuSupported) {
      setStatusMessage(webGpuSupportState.message ?? '当前浏览器不支持 WebGPU。')
      return
    }

    stopActiveStream()
    setImageDetectBusy(true)
    clearRecognitionOutputs()
    setStatusMessage(`正在使用 ${importedModel.contract.family} 模型执行图片识别...`)
    const detectionOptions = commitInferenceSettings()

    try {
      const detectionRun = await runSingleImageDetection(importedModel, imageFile, detectionOptions)
      applyDetectionRun(detectionRun)
      setStatusMessage(
        detectionRun.recognitionResult.detections.length === 0
          ? '识别完成，当前图片未检测到目标。'
          : `识别完成，共检测到 ${detectionRun.recognitionResult.detections.length} 个目标。`,
      )
    } catch (error) {
      setRuntimeMessage(formatError(error))
      setStatusMessage(`识别失败：${formatError(error)}`)
    } finally {
      setImageDetectBusy(false)
    }
  }

  async function handleExportImage() {
    if (!resultCanvasRef.current || !hasRenderedResult || !imageFile) {
      return
    }

    try {
      await downloadCanvasAsPng(
        resultCanvasRef.current,
        buildImageExportFileName(imageFile.name),
      )
      setStatusMessage('结果图像已导出。')
    } catch (error) {
      setStatusMessage(`导出结果图像失败：${formatError(error)}`)
    }
  }

  async function handleStartVideoDetection() {
    if (!importedModel || !videoFile || !sourceVideoRef.current) {
      return
    }

    if (!isWebGpuSupported) {
      setStatusMessage(webGpuSupportState.message ?? '当前浏览器不支持 WebGPU。')
      return
    }

    clearRecognitionOutputs()
    setStatusMessage(`正在启动视频识别：${videoFile.name}。`)
    const detectionOptions = commitInferenceSettings()

    try {
      const videoElement = sourceVideoRef.current
      await ensureVideoReady(videoElement)
      await videoElement.play()
      startTransition(() => {
        setStreamState('running')
      })
      startVideoLoop(videoElement, 'video', detectionOptions)
      setStatusMessage('视频实时识别已启动。')
    } catch (error) {
      setStreamState('idle')
      setStatusMessage(`启动视频识别失败：${formatError(error)}`)
    }
  }

  async function handleStartCameraDetection() {
    if (!importedModel) {
      return
    }

    if (!isWebGpuSupported) {
      setStatusMessage(webGpuSupportState.message ?? '当前浏览器不支持 WebGPU。')
      return
    }

    if (!supportsCamera) {
      setStatusMessage('当前浏览器不支持摄像头访问。')
      return
    }

    clearRecognitionOutputs()
    setCameraBusy(true)
    setStatusMessage('正在请求摄像头权限...')
    const detectionOptions = commitInferenceSettings()

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: selectedCameraId
          ? { deviceId: { exact: selectedCameraId } }
          : true,
        audio: false,
      })

      const videoElement = cameraVideoRef.current
      if (!videoElement) {
        throw new Error('摄像头预览控件尚未初始化。')
      }

      cameraStreamRef.current = stream
      videoElement.srcObject = stream
      await videoElement.play()
      await refreshCameraDevices()

      startTransition(() => {
        setStreamState('running')
      })
      startVideoLoop(videoElement, 'camera', detectionOptions)
      setStatusMessage('摄像头实时识别已启动。')
    } catch (error) {
      stopActiveStream()
      setStatusMessage(`启动摄像头识别失败：${formatError(error)}`)
    } finally {
      setCameraBusy(false)
    }
  }

  async function handleExportVideo() {
    if (!importedModel || !videoFile || !videoPreviewUrl) {
      return
    }

    if (!isWebGpuSupported) {
      setStatusMessage(webGpuSupportState.message ?? '当前浏览器不支持 WebGPU。')
      return
    }

    if (!preferredVideoMimeType) {
      setStatusMessage('当前浏览器不支持导出 WebM 结果视频。')
      return
    }

    clearRecognitionOutputs()
    setStreamState('exporting')
    setStatusMessage(`正在导出结果视频：${videoFile.name}。`)
    const detectionOptions = commitInferenceSettings()

    const exportVideo = document.createElement('video')
    exportVideo.muted = true
    exportVideo.playsInline = true
    exportVideo.preload = 'auto'
    exportVideo.src = videoPreviewUrl

    const exportCanvas = document.createElement('canvas')
    const chunks: BlobPart[] = []
    let recorder: MediaRecorder | null = null

    try {
      await ensureVideoReady(exportVideo)
      exportCanvas.width = exportVideo.videoWidth
      exportCanvas.height = exportVideo.videoHeight

      const stream = exportCanvas.captureStream(30)
      recorder = new MediaRecorder(stream, { mimeType: preferredVideoMimeType })

      const blobPromise = new Promise<Blob>((resolve, reject) => {
        recorder!.ondataavailable = (event) => {
          if (event.data.size > 0) {
            chunks.push(event.data)
          }
        }
        recorder!.onstop = () => {
          resolve(new Blob(chunks, { type: preferredVideoMimeType }))
        }
        recorder!.onerror = () => {
          reject(new Error('结果视频录制失败。'))
        }
      })

      recorder.start()
      await exportVideo.play()
      await processVideoFrames(exportVideo, 'video', exportCanvas, detectionOptions)
      if (recorder.state !== 'inactive') {
        recorder.stop()
      }

      const videoBlob = await blobPromise
      const nextDownloadUrl = URL.createObjectURL(videoBlob)
      setVideoDownloadUrl(nextDownloadUrl)
      setStatusMessage('结果视频导出完成。')
    } catch (error) {
      if (recorder && recorder.state !== 'inactive') {
        recorder.stop()
      }
      setStatusMessage(`导出结果视频失败：${formatError(error)}`)
    } finally {
      exportVideo.pause()
      exportVideo.removeAttribute('src')
      exportVideo.load()
      setStreamState('idle')
    }
  }

  function startVideoLoop(
    videoElement: HTMLVideoElement,
    sourceKind: 'video' | 'camera',
    detectionOptions: RunDetectionOptions,
  ) {
    void processVideoFrames(videoElement, sourceKind, undefined, detectionOptions)
      .catch((error) => {
        stopActiveStream(`实时识别失败：${formatError(error)}`)
      })
  }

  async function processVideoFrames(
    videoElement: HTMLVideoElement,
    sourceKind: 'video' | 'camera',
    exportCanvas?: HTMLCanvasElement,
    detectionOptions?: RunDetectionOptions,
  ): Promise<void> {
    if (!importedModel) {
      return
    }

    const token = frameLoopTokenRef.current + 1
    frameLoopTokenRef.current = token
    frameInFlightRef.current = false

    await new Promise<void>((resolve, reject) => {
      const step = async () => {
        if (frameLoopTokenRef.current !== token) {
          resolve()
          return
        }

        if (videoElement.ended) {
          resolve()
          return
        }

        if (videoElement.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
          scheduleNext()
          return
        }

        if (frameInFlightRef.current) {
          scheduleNext()
          return
        }

        frameInFlightRef.current = true
        try {
          const sourceCanvas = drawSourceToCanvas(videoElement, workingCanvasRef.current ?? undefined)
          workingCanvasRef.current = sourceCanvas
          const detectionRun = await runDetectionOnCanvas(
            importedModel,
            sourceCanvas,
            buildFrameSourceLabel(sourceKind, videoElement.currentTime, videoFile?.name, selectedCameraId, cameraDevices),
            detectionOptions,
          )
          if (frameLoopTokenRef.current !== token) {
            resolve()
            return
          }

          const nextFps = calculateNextFps(frameTimestampsRef.current, performance.now())
          applyDetectionRun(detectionRun, videoElement.currentTime, exportCanvas, nextFps)
          scheduleNext()
        } catch (error) {
          reject(error)
        } finally {
          frameInFlightRef.current = false
        }
      }

      const scheduleNext = () => {
        if (frameLoopTokenRef.current !== token) {
          resolve()
          return
        }

        requestNextVideoFrame(videoElement, () => {
          void step()
        })
      }

      videoElement.addEventListener('ended', () => resolve(), { once: true })
      scheduleNext()
    })

    if (sourceKind === 'video' && !exportCanvas && frameLoopTokenRef.current === token) {
      stopActiveStream('视频识别已结束。')
    }
  }

  function applyDetectionRun(
    detectionRun: DetectionRun,
    timeSeconds: number | null = null,
    exportCanvas?: HTMLCanvasElement,
    fps: number | null = null,
  ) {
    syncCanvas(detectionRun.annotatedCanvas, resultCanvasRef.current)
    drawRuntimeOverlay(resultCanvasRef.current, fps)
    if (exportCanvas) {
      syncCanvas(detectionRun.annotatedCanvas, exportCanvas)
      drawRuntimeOverlay(exportCanvas, fps)
    }

    startTransition(() => {
      setHasRenderedResult(true)
      setResultProvider(detectionRun.providerName)
      setRuntimeMessage(detectionRun.runtimeMessage ?? '')
      setCurrentSourceTime(timeSeconds)
      setDetectionItems(toDetectionItems(detectionRun))
      setStreamFps(fps)
    })
  }

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero__eyebrow">4C-ai装备识别工具</div>
        <h1>React 19 + TypeScript 的本地 ONNX 工作台</h1>
        <p className="hero__copy">
          当前版本支持图片、本地视频与摄像头三种输入来源，模型导入、推理编排、结果解码与结果导出全部在浏览器端完成。
        </p>
      </section>

      <section className="grid">
        <div className="panel-stack">
          <article className="panel panel--input">
            <header className="panel__header">
              <h2>输入与导入</h2>
              <p>选择输入模式后，再导入 ONNX 模型与可选的 Hugging Face 配置文件。</p>
            </header>

            <div className="mode-switch">
              {(['image', 'video', 'camera'] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  className={`mode-switch__button${inputMode === mode ? ' mode-switch__button--active' : ''}`}
                  disabled={streamState !== 'idle'}
                  onClick={() => {
                    stopActiveStream()
                    clearRecognitionOutputs()
                    setInputMode(mode)
                    if (mode === 'camera') {
                      void refreshCameraDevices()
                    }
                    setStatusMessage(
                      mode === 'image'
                        ? '已切换到图片识别模式。'
                        : mode === 'video'
                          ? '已切换到本地视频识别模式。'
                          : '已切换到摄像头识别模式。',
                    )
                  }}
                >
                  {mode === 'image' ? '图片' : mode === 'video' ? '视频' : '摄像头'}
                </button>
              ))}
            </div>

            {inputMode === 'image' ? (
              <label className="field">
                <span className="field__label">图片文件</span>
                <input
                  type="file"
                  accept="image/*"
                  disabled={streamState !== 'idle'}
                  onChange={(event) => {
                    updateImageFile(event.target.files?.[0] ?? null)
                  }}
                />
                <span className="field__hint">{imageFile?.name ?? '未选择图片。'}</span>
              </label>
            ) : null}

            {inputMode === 'video' ? (
              <label className="field">
                <span className="field__label">视频文件</span>
                <input
                  type="file"
                  accept="video/*"
                  disabled={streamState !== 'idle'}
                  onChange={(event) => {
                    updateVideoFile(event.target.files?.[0] ?? null)
                  }}
                />
                <span className="field__hint">{videoFile?.name ?? '未选择视频。'}</span>
              </label>
            ) : null}

            {inputMode === 'camera' ? (
              <label className="field">
                <span className="field__label">摄像头设备</span>
                <select
                  className="field__select"
                  value={selectedCameraId}
                  disabled={cameraBusy || streamState !== 'idle' || cameraDevices.length === 0}
                  onChange={(event) => {
                    setSelectedCameraId(event.target.value)
                  }}
                >
                  {cameraDevices.length === 0 ? (
                    <option value="">未发现可用摄像头</option>
                  ) : (
                    cameraDevices.map((item) => (
                      <option key={item.deviceId} value={item.deviceId}>
                        {item.label}
                      </option>
                    ))
                  )}
                </select>
                <span className="field__hint">
                  {supportsCamera
                    ? (cameraDevices.length === 0 ? '首次授权前可能无法显示设备标签。' : '选择需要用于实时识别的摄像头。')
                    : '当前浏览器不支持摄像头访问。'}
                </span>
              </label>
            ) : null}

            <label className="field">
              <span className="field__label">ONNX 模型</span>
              <input
                key={onnxInputKey}
                type="file"
                accept=".onnx"
                disabled={modelBusy || streamState !== 'idle'}
                onChange={(event) => {
                  void handleOnnxSelection(event.target.files?.[0] ?? null)
                }}
              />
              <span className="field__hint">
                {onnxModelDraft?.fileName ?? '仅允许导入 .onnx 文件。'}
              </span>
            </label>

            <label className="field">
              <span className="field__label">{CONFIG_FILE_NAME} {importControls.configRequired ? '(必选)' : '(未启用)'}</span>
              <input
                key={configInputKey}
                type="file"
                accept=".json,application/json"
                disabled={!importControls.configEnabled || modelBusy || streamState !== 'idle'}
                onChange={(event) => {
                  void handleConfigSelection(event.target.files?.[0] ?? null)
                }}
              />
              <span className="field__hint">
                {configFile?.name ?? (
                  importControls.configEnabled
                    ? `仅允许导入 ${CONFIG_FILE_NAME}。`
                    : '请先导入 ONNX，并识别为 Hugging Face 风格模型。'
                )}
              </span>
            </label>

            <label className="field">
              <span className="field__label">{PREPROCESSOR_CONFIG_FILE_NAME} {importControls.preprocessorEnabled ? '(可选)' : '(未启用)'}</span>
              <input
                key={preprocessorInputKey}
                type="file"
                accept=".json,application/json"
                disabled={!importControls.preprocessorEnabled || modelBusy || streamState !== 'idle'}
                onChange={(event) => {
                  void handlePreprocessorSelection(event.target.files?.[0] ?? null)
                }}
              />
              <span className="field__hint">
                {preprocessorConfigFile?.name ?? (
                  importControls.preprocessorEnabled
                    ? `仅允许导入 ${PREPROCESSOR_CONFIG_FILE_NAME}。`
                    : '当前模型不需要该配置文件。'
                )}
              </span>
            </label>

            {onnxModelDraft?.family === 'hf-detr-like' ? (
              <div className="actions actions--stacked">
                <button
                  type="button"
                  className="action action--secondary"
                  disabled={!importControls.autoDiscoverEnabled || modelBusy || discoverBusy || streamState !== 'idle'}
                  onClick={() => void handleAutoDiscoverSidecars()}
                >
                  {discoverBusy ? '查找中...' : '自动查找同目录配置'}
                </button>
                {!supportsDirectoryPicker ? (
                  <span className="field__hint">当前浏览器不支持目录授权，请手动导入 JSON 配置文件。</span>
                ) : null}
              </div>
            ) : null}

            <div className="status-box">
              <div className="status-box__title">状态</div>
              <p>{statusMessage}</p>
              {displayedRuntimeMessage ? <p>{displayedRuntimeMessage}</p> : null}
            </div>
          </article>

          <article className="panel panel--settings">
            <header className="panel__header">
              <h2>推理设置</h2>
              <p>在启动识别前统一设置阈值、识别数量和导出相关操作。</p>
            </header>

            <label className="field">
              <span className="field__label">推理阈值</span>
              <div className="threshold-control">
                <input
                  className="threshold-control__slider"
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={pendingDetectionThreshold}
                  disabled={!canAdjustInferenceSettings}
                  onChange={(event) => {
                    setPendingDetectionThreshold(clampDetectionThreshold(Number(event.target.value)))
                  }}
                  onPointerUp={(event) => {
                    commitDetectionThreshold(Number(event.currentTarget.value))
                  }}
                  onKeyUp={(event) => {
                    if (THRESHOLD_COMMIT_KEYS.has(event.key)) {
                      commitDetectionThreshold(Number(event.currentTarget.value))
                    }
                  }}
                  onBlur={(event) => {
                    commitDetectionThreshold(Number(event.currentTarget.value))
                  }}
                />
                <span className="threshold-control__value">{displayedPendingThreshold}</span>
              </div>
              <span className="field__hint">
                {importedModel
                  ? `当前值 ${displayedPendingThreshold}，已生效 ${displayedAppliedThreshold}。拖动过程中不会立即重跑推理，松开后新阈值才会生效。`
                  : '导入模型后可调整推理阈值。'}
              </span>
            </label>

            <label className="field">
              <span className="field__label">识别数量</span>
              <div className="count-control">
                <input
                  className="count-control__input"
                  type="number"
                  min="0"
                  step="1"
                  inputMode="numeric"
                  value={pendingMaxDetectionsInput}
                  disabled={!canAdjustInferenceSettings}
                  onChange={(event) => {
                    setPendingMaxDetectionsInput(event.target.value)
                  }}
                  onBlur={(event) => {
                    commitMaxDetections(event.currentTarget.value)
                  }}
                />
                <span className="count-control__value">{displayedAppliedMaxDetections}</span>
              </div>
              <span className="field__hint">
                {importedModel
                  ? `输入 0 代表全部识别并绘制；当前输入 ${pendingMaxDetectionsInput || '0'}，已生效 ${displayedAppliedMaxDetections}。失焦或开始推理时才会提交新值。`
                  : '导入模型后可设置识别数量上限，0 代表全部识别并绘制。'}
              </span>
            </label>

            <div className="actions actions--stacked">
              {inputMode === 'image' ? (
                <>
                  <button
                    type="button"
                    className="action action--primary"
                    disabled={!canRunImage}
                    onClick={() => void handleRunImageDetection()}
                  >
                    {imageDetectBusy ? '识别中...' : '执行图片识别'}
                  </button>
                  <button
                    type="button"
                    className="action action--secondary"
                    disabled={!canExportImage}
                    onClick={() => void handleExportImage()}
                  >
                    导出结果图像
                  </button>
                </>
              ) : null}

              {inputMode === 'video' ? (
                <>
                  <button
                    type="button"
                    className="action action--primary"
                    disabled={!canStartVideo}
                    onClick={() => void handleStartVideoDetection()}
                  >
                    开始实时识别
                  </button>
                  <button
                    type="button"
                    className="action action--secondary"
                    disabled={!canStopVideo}
                    onClick={() => stopActiveStream('已停止视频实时识别。')}
                  >
                    停止实时识别
                  </button>
                  <button
                    type="button"
                    className="action action--secondary"
                    disabled={!canExportVideo}
                    onClick={() => void handleExportVideo()}
                  >
                    {streamState === 'exporting' ? '导出中...' : '导出结果视频'}
                  </button>
                  {videoDownloadUrl ? (
                    <a
                      className="download-link"
                      href={videoDownloadUrl}
                      download={buildVideoExportFileName(videoFile?.name ?? 'result')}
                    >
                      下载结果 WebM
                    </a>
                  ) : null}
                  {!supportsVideoExport ? (
                    <span className="field__hint">当前浏览器不支持 `MediaRecorder WebM` 导出。</span>
                  ) : null}
                </>
              ) : null}

              {inputMode === 'camera' ? (
                <>
                  <button
                    type="button"
                    className="action action--primary"
                    disabled={!canStartCamera}
                    onClick={() => void handleStartCameraDetection()}
                  >
                    {cameraBusy ? '启动中...' : '开始实时识别'}
                  </button>
                  <button
                    type="button"
                    className="action action--secondary"
                    disabled={!canStopCamera}
                    onClick={() => stopActiveStream('已停止摄像头实时识别。')}
                  >
                    停止实时识别
                  </button>
                </>
              ) : null}
            </div>
          </article>
        </div>

        <article className="panel panel--contract">
          <header className="panel__header">
            <h2>模型契约</h2>
            <p>导入后先根据输入输出签名解析 family、预处理和解码规则。</p>
          </header>

          {displayedContract ? (
            <div className="contract">
              <div className="metric-grid">
                <Metric label="Display Name" value={displayedContract.displayName} />
                <Metric label="Family" value={displayedContract.family} />
                <Metric label="Provider" value={displayedProvider} />
                <Metric label="Label Source" value={displayedContract.labelSource} />
                <Metric label="Sidecars" value={displayedSidecars} />
                <Metric label="Input Tensor" value={displayedContract.preprocess.inputTensorName} />
                <Metric
                  label="Image Size"
                  value={`${displayedContract.preprocess.imageWidth} x ${displayedContract.preprocess.imageHeight}`}
                />
              </div>

              <SignatureTable title="Inputs" items={displayedContract.inputs.map((item) => ({
                name: item.name,
                dims: formatDims(item.dimensions),
              }))} />

              <SignatureTable title="Outputs" items={displayedContract.outputs.map((item) => ({
                name: item.name,
                dims: formatDims(item.dimensions),
              }))} />

              <div className="warnings">
                <div className="warnings__title">Warnings</div>
                {displayedContract.warnings.length === 0 ? (
                  <p>当前模型未发现额外警告。</p>
                ) : (
                  displayedContract.warnings.map((item) => (
                    <p key={item}>{item}</p>
                  ))
                )}
              </div>
            </div>
          ) : (
            <EmptyState text={modelBusy ? '正在建立会话并解析模型签名...' : '尚未导入模型。'} />
          )}
        </article>
      </section>

      <section className="grid grid--results">
        <article className="panel panel--preview">
          <header className="panel__header">
            <h2>统一预览</h2>
            <p>识别前显示输入源，识别后在同一区域显示结果叠加画面。</p>
          </header>

          <div className="preview-card preview-stage">
            <div className="preview-card__title">当前画面</div>
            <div className="preview-stage__media">
              {inputMode === 'image' ? (
                imagePreviewUrl ? (
                  <img
                    className={`preview-stage__image${hasRenderedResult ? ' preview-stage__visual--hidden' : ''}`}
                    src={imagePreviewUrl}
                    alt="输入图片"
                  />
                ) : (
                  <EmptyState text="未选择图片。" />
                )
              ) : null}

              {inputMode === 'video' ? (
                videoPreviewUrl ? (
                  <video
                    ref={sourceVideoRef}
                    className={`preview-stage__video${hasRenderedResult ? ' preview-stage__visual--hidden' : ''}`}
                    src={videoPreviewUrl}
                    controls
                    playsInline
                    muted
                  />
                ) : (
                  <EmptyState text="未选择视频。" />
                )
              ) : null}

              {inputMode === 'camera' ? (
                <>
                  <video
                    ref={cameraVideoRef}
                    className={`preview-stage__video${streamState === 'running' && !hasRenderedResult ? '' : ' preview-stage__visual--hidden'}`}
                    autoPlay
                    muted
                    playsInline
                  />
                  {streamState !== 'running' ? <EmptyState text="尚未启动摄像头。" /> : null}
                </>
              ) : null}

              <canvas
                ref={resultCanvasRef}
                className={`preview-stage__canvas${hasRenderedResult ? '' : ' preview-stage__visual--hidden'}`}
              />
            </div>
          </div>
        </article>

        <article className="panel panel--detections">
          <header className="panel__header">
            <h2>检测结果</h2>
            <p>输出来自 `Contracts` 约束的统一检测结构。</p>
          </header>

          <div className="metric-grid metric-grid--compact">
            <Metric label="Runtime" value={resultProvider || '-'} />
            <Metric label="Source Mode" value={inputMode} />
            <Metric label="Detection Count" value={String(detectionItems.length)} />
            <Metric label="Current Time" value={currentTimeLabel} />
            <Metric label="FPS" value={displayedFps} />
          </div>

          {detectionItems.length === 0 ? (
            <EmptyState text={
              inputMode === 'image'
                ? (imageDetectBusy ? '正在等待图片识别结果...' : '当前没有可展示的检测项。')
                : (streamState === 'running' || streamState === 'exporting')
                  ? '正在等待当前帧的推理结果...'
                  : '当前没有可展示的检测项。'
            } />
          ) : (
            <div className="result-list">
              {detectionItems.map((item, index) => (
                <div className="result-row" key={`${item.label}-${index}`}>
                  <div>
                    <div className="result-row__label">{item.label}</div>
                    <div className="result-row__box">{item.boxSummary}</div>
                  </div>
                  <div className="result-row__score">{(item.confidence * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          )}
        </article>
      </section>
    </main>
  )
}

function Metric(props: { label: string; value: string }) {
  return (
    <div className="metric">
      <div className="metric__label">{props.label}</div>
      <div className="metric__value">{props.value}</div>
    </div>
  )
}

function SignatureTable(props: { title: string; items: Array<{ name: string; dims: string }> }) {
  return (
    <section className="signature-block">
      <h3>{props.title}</h3>
      <div className="signature-list">
        {props.items.map((item) => (
          <div className="signature-row" key={`${props.title}-${item.name}`}>
            <span>{item.name}</span>
            <code>{item.dims}</code>
          </div>
        ))}
      </div>
    </section>
  )
}

function EmptyState(props: { text: string }) {
  return (
    <div className="empty-state">
      <p>{props.text}</p>
    </div>
  )
}

function formatModelReadyMessage(
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
  const compatibilityNote = !model.webGpuCompatibility.supported
    ? ` ${formatWebGpuCompatibilityMessage(model.webGpuCompatibility)}`
    : ''

  return sidecars.length > 0
    ? `模型解析成功：${model.contract.family}，已使用 ${sidecars.join('、')}，当前执行提供器为 ${model.providerName}。${runtimeNote}${compatibilityNote}`
    : `模型解析成功：${model.contract.family}，当前执行提供器为 ${model.providerName}。${runtimeNote}${compatibilityNote}`
}

function formatAutoDiscoveredMessage(
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

function buildPendingHfMessage(
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

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === 'AbortError'
    : Boolean(
      error &&
      typeof error === 'object' &&
      'name' in error &&
      error.name === 'AbortError',
    )
}

function formatDims(dims: Array<number | string | null>): string {
  return `[${dims.map((item) => item ?? '?').join(', ')}]`
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function formatWebGpuCompatibilityMessage(report: { supported: boolean; issues: Array<{ severity: string; message: string }> }): string {
  const errorMessage = report.issues.find((item) => item.severity === 'error')?.message
  return errorMessage ?? '当前模型不兼容 WebGPU，无法执行识别。'
}

function toDetectionItems(run: DetectionRun): DetectionItem[] {
  return run.recognitionResult.detections.map((item) => ({
    label: item.label,
    confidence: item.confidence,
    boxSummary: `${item.box.x.toFixed(1)}, ${item.box.y.toFixed(1)}, ${item.box.width.toFixed(1)}, ${item.box.height.toFixed(1)}`,
  }))
}

function clearCanvas(canvas: HTMLCanvasElement | null) {
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

function syncCanvas(source: HTMLCanvasElement, target: HTMLCanvasElement | null) {
  if (!target) {
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

function drawRuntimeOverlay(canvas: HTMLCanvasElement | null, fps: number | null) {
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

function calculateNextFps(frameTimestamps: number[], timestamp: number): number {
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

function clampDetectionThreshold(value: number): number {
  if (!Number.isFinite(value)) {
    return DEFAULT_DETECTION_THRESHOLD
  }

  return Math.min(Math.max(value, 0), 1)
}

function normalizeMaxDetectionsValue(value: number | string): number {
  const parsedValue = typeof value === 'string'
    ? Number(value.trim())
    : value

  if (!Number.isFinite(parsedValue)) {
    return DEFAULT_MAX_DETECTIONS
  }

  return Math.max(0, Math.trunc(parsedValue))
}

async function ensureVideoReady(videoElement: HTMLVideoElement): Promise<void> {
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

function requestNextVideoFrame(
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

async function downloadCanvasAsPng(
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

function buildImageExportFileName(fileName: string): string {
  return `${stripExtension(fileName)}-detected.png`
}

function buildVideoExportFileName(fileName: string): string {
  return `${stripExtension(fileName)}-detected.webm`
}

function stripExtension(fileName: string): string {
  return fileName.replace(/\.[^.]+$/, '')
}

function buildFrameSourceLabel(
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

function getPreferredVideoMimeType(): string | null {
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

export default App
