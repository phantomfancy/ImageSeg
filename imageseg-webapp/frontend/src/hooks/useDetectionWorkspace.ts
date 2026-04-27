import {
  startTransition,
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from 'react'
import type {
  CameraDeviceOption,
  DetectionItem,
  InputMode,
  StreamState,
} from '../app/types'
import {
  DEFAULT_DETECTION_THRESHOLD,
  DEFAULT_MAX_DETECTIONS,
  buildFrameSourceLabel,
  buildImageExportFileName,
  calculateNextFps,
  clampDetectionThreshold,
  clearCanvas,
  downloadCanvasAsPng,
  drawRuntimeOverlay,
  ensureVideoReady,
  formatError,
  getPreferredVideoMimeType,
  normalizeMaxDetectionsValue,
  requestNextVideoFrame,
  syncCanvas,
  toDetectionItems,
} from '../app/workspaceUtils'
import type {
  DetectionRun,
  ImportedModel,
  RunDetectionOptions,
} from '../lib/onnxRuntime'
import {
  runDetectionOnSource,
  runSingleImageDetection,
} from '../lib/onnxRuntime'

type UseDetectionWorkspaceArgs = {
  cameraStreamRef: RefObject<MediaStream | null>
  cameraVideoRef: RefObject<HTMLVideoElement | null>
  importedModelRef: RefObject<ImportedModel | null>
  isWebGpuSupported: boolean
  resetPreviewViewerRef: RefObject<(() => void) | null>
  resultCanvasRef: RefObject<HTMLCanvasElement | null>
  sourceVideoRef: RefObject<HTMLVideoElement | null>
  supportsCamera: boolean
  webGpuSupportState: { supported: boolean; message?: string }
}

export function useDetectionWorkspace(args: UseDetectionWorkspaceArgs) {
  const {
    cameraStreamRef,
    cameraVideoRef,
    importedModelRef,
    isWebGpuSupported,
    resetPreviewViewerRef,
    resultCanvasRef,
    sourceVideoRef,
    supportsCamera,
    webGpuSupportState,
  } = args

  const [inputMode, setInputMode] = useState<InputMode>('image')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreviewUrl, setImagePreviewUrl] = useState('')
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoPreviewUrl, setVideoPreviewUrl] = useState('')
  const [statusMessage, setStatusMessage] = useState('请先导入输入源和 ONNX 模型。')
  const [runtimeMessage, setRuntimeMessage] = useState('')
  const [imageDetectBusy, setImageDetectBusy] = useState(false)
  const [streamState, setStreamState] = useState<StreamState>('idle')
  const [cameraBusy, setCameraBusy] = useState(false)
  const [cameraDevices, setCameraDevices] = useState<CameraDeviceOption[]>([])
  const [selectedCameraId, setSelectedCameraId] = useState('')
  const [videoDownloadUrl, setVideoDownloadUrl] = useState('')
  const [resultProvider, setResultProvider] = useState('')
  const [currentSourceTime, setCurrentSourceTime] = useState<number | null>(null)
  const [hasRenderedResult, setHasRenderedResult] = useState(false)
  const [detectionItems, setDetectionItems] = useState<DetectionItem[]>([])
  const [pendingDetectionThreshold, setPendingDetectionThreshold] = useState(
    DEFAULT_DETECTION_THRESHOLD,
  )
  const [appliedDetectionThreshold, setAppliedDetectionThreshold] = useState(
    DEFAULT_DETECTION_THRESHOLD,
  )
  const [pendingMaxDetectionsInput, setPendingMaxDetectionsInput] = useState(
    String(DEFAULT_MAX_DETECTIONS),
  )
  const [appliedMaxDetections, setAppliedMaxDetections] = useState(DEFAULT_MAX_DETECTIONS)
  const [streamFps, setStreamFps] = useState<number | null>(null)

  const frameLoopTokenRef = useRef(0)
  const frameInFlightRef = useRef(false)
  const frameTimestampsRef = useRef<number[]>([])

  const preferredVideoMimeType = getPreferredVideoMimeType()

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
      maxDetectionsOverride,
      scoreThresholdOverride,
    }
  }

  const stopActiveStream = useCallback((statusText?: string) => {
    resetPreviewViewerRef.current?.()
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
  }, [cameraStreamRef, cameraVideoRef, resetPreviewViewerRef, sourceVideoRef])

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

  function clearRecognitionOutputs() {
    resetPreviewViewerRef.current?.()
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
    const importedModel = importedModelRef.current
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
          const detectionRun = await runDetectionOnSource(
            importedModel,
            videoElement,
            buildFrameSourceLabel(
              sourceKind,
              videoElement.currentTime,
              videoFile?.name,
              selectedCameraId,
              cameraDevices,
            ),
            detectionOptions,
            resultCanvasRef.current ?? undefined,
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

  async function handleRunImageDetection() {
    const importedModel = importedModelRef.current
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
    const importedModel = importedModelRef.current
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
    const importedModel = importedModelRef.current
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
    const importedModel = importedModelRef.current
    if (!importedModel || !videoFile || !videoPreviewUrl || !preferredVideoMimeType) {
      return
    }

    if (!isWebGpuSupported) {
      setStatusMessage(webGpuSupportState.message ?? '当前浏览器不支持 WebGPU。')
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

  function handleInputModeChange(mode: InputMode) {
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
  }

  return {
    appliedDetectionThreshold,
    appliedMaxDetections,
    cameraBusy,
    cameraDevices,
    cameraStreamRef,
    cameraVideoRef,
    clearRecognitionOutputs,
    commitDetectionThreshold,
    commitMaxDetections,
    currentSourceTime,
    detectionItems,
    handleExportImage,
    handleExportVideo,
    handleInputModeChange,
    handleRunImageDetection,
    handleStartCameraDetection,
    handleStartVideoDetection,
    hasRenderedResult,
    imageDetectBusy,
    imageFile,
    imagePreviewUrl,
    inputMode,
    pendingDetectionThreshold,
    pendingMaxDetectionsInput,
    preferredVideoMimeType,
    refreshCameraDevices,
    resetDetectionThreshold,
    resetMaxDetections,
    resultCanvasRef,
    resultProvider,
    runtimeMessage,
    selectedCameraId,
    setPendingDetectionThreshold,
    setPendingMaxDetectionsInput,
    setSelectedCameraId,
    setStatusMessage,
    sourceVideoRef,
    statusMessage,
    stopActiveStream,
    streamFps,
    streamState,
    updateImageFile,
    updateVideoFile,
    videoDownloadUrl,
    videoFile,
    videoPreviewUrl,
  }
}
