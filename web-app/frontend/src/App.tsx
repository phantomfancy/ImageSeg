import { startTransition, useCallback, useEffect, useRef, useState } from 'react'
import type { DetectionRun, ImportedModel } from './lib/onnxRuntime'
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

  const sourceVideoRef = useRef<HTMLVideoElement | null>(null)
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null)
  const resultCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const workingCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const cameraStreamRef = useRef<MediaStream | null>(null)
  const frameLoopTokenRef = useRef(0)
  const frameInFlightRef = useRef(false)

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

    if (videoDownloadUrl) {
      URL.revokeObjectURL(videoDownloadUrl)
      setVideoDownloadUrl('')
    }
  }

  function clearImportedModelState() {
    setImportedModel(null)
    clearRecognitionOutputs()
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
      if (statusText) {
        setStatusMessage(statusText)
      }
    })
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
          setStatusMessage(buildPendingHfMessage(null, null, supportsDirectoryPicker))
        })
        return
      }

      const model = await inspectImportedModel({ onnxModel: nextOnnxModel })
      startTransition(() => {
        setOnnxModelDraft(nextOnnxModel)
        setImportedModel(model)
        clearRecognitionOutputs()
        setStatusMessage(formatModelReadyMessage(model, webGpuSupportState))
      })
    } catch (error) {
      setStatusMessage(`模型解析失败：${formatError(error)}`)
    } finally {
      setModelBusy(false)
      setOnnxInputKey((value) => value + 1)
    }
  }

  async function handleConfigSelection(file: File | null) {
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
        clearRecognitionOutputs()
        setStatusMessage(formatModelReadyMessage(model, webGpuSupportState))
      })
    } catch (error) {
      setStatusMessage(`config.json 导入失败：${formatError(error)}`)
    } finally {
      setModelBusy(false)
      setConfigInputKey((value) => value + 1)
    }
  }

  async function handlePreprocessorSelection(file: File | null) {
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
        setStatusMessage(buildPendingHfMessage(null, file, supportsDirectoryPicker))
      })
      setPreprocessorInputKey((value) => value + 1)
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
        clearRecognitionOutputs()
        setStatusMessage(formatModelReadyMessage(model, webGpuSupportState))
      })
    } catch (error) {
      setStatusMessage(`preprocessor_config.json 导入失败：${formatError(error)}`)
    } finally {
      setModelBusy(false)
      setPreprocessorInputKey((value) => value + 1)
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

    try {
      const detectionRun = await runSingleImageDetection(importedModel, imageFile)
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

    try {
      const videoElement = sourceVideoRef.current
      await ensureVideoReady(videoElement)
      await videoElement.play()
      startTransition(() => {
        setStreamState('running')
      })
      startVideoLoop(videoElement, 'video')
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
      startVideoLoop(videoElement, 'camera')
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
      await processVideoFrames(exportVideo, 'video', exportCanvas)
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

  function startVideoLoop(videoElement: HTMLVideoElement, sourceKind: 'video' | 'camera') {
    void processVideoFrames(videoElement, sourceKind)
      .catch((error) => {
        stopActiveStream(`实时识别失败：${formatError(error)}`)
      })
  }

  async function processVideoFrames(
    videoElement: HTMLVideoElement,
    sourceKind: 'video' | 'camera',
    exportCanvas?: HTMLCanvasElement,
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
          )
          if (frameLoopTokenRef.current !== token) {
            resolve()
            return
          }

          applyDetectionRun(detectionRun, videoElement.currentTime, exportCanvas)
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
  ) {
    syncCanvas(detectionRun.annotatedCanvas, resultCanvasRef.current)
    if (exportCanvas) {
      syncCanvas(detectionRun.annotatedCanvas, exportCanvas)
    }

    startTransition(() => {
      setHasRenderedResult(true)
      setResultProvider(detectionRun.providerName)
      setRuntimeMessage(detectionRun.runtimeMessage ?? '')
      setCurrentSourceTime(timeSeconds)
      setDetectionItems(toDetectionItems(detectionRun))
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

          <div className="status-box">
            <div className="status-box__title">状态</div>
            <p>{statusMessage}</p>
            {displayedRuntimeMessage ? <p>{displayedRuntimeMessage}</p> : null}
          </div>
        </article>

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
            <h2>输入预览与结果叠加</h2>
            <p>左侧显示当前输入来源，右侧显示识别结果叠加画面。</p>
          </header>

          <div className="preview-grid">
            <div className="preview-card">
              <div className="preview-card__title">输入源</div>
              {inputMode === 'image' ? (
                imagePreviewUrl ? (
                  <img className="preview-card__image" src={imagePreviewUrl} alt="输入图片" />
                ) : (
                  <EmptyState text="未选择图片。" />
                )
              ) : inputMode === 'video' ? (
                <div className="preview-card__media">
                  <video
                    ref={sourceVideoRef}
                    className={`preview-card__video${videoPreviewUrl ? '' : ' preview-card__video--hidden'}`}
                    src={videoPreviewUrl}
                    controls
                    playsInline
                    muted
                  />
                  {!videoPreviewUrl ? <EmptyState text="未选择视频。" /> : null}
                </div>
              ) : (
                <div className="preview-card__media">
                  <video
                    ref={cameraVideoRef}
                    className="preview-card__video"
                    autoPlay
                    muted
                    playsInline
                  />
                  {streamState !== 'running' ? <EmptyState text="尚未启动摄像头。" /> : null}
                </div>
              )}
            </div>

            <div className="preview-card preview-card--canvas">
              <div className="preview-card__title">结果叠加图</div>
              <canvas
                ref={resultCanvasRef}
                className={`preview-card__canvas${hasRenderedResult ? '' : ' preview-card__canvas--hidden'}`}
              />
              {!hasRenderedResult ? (
                <EmptyState
                  text={
                    inputMode === 'image'
                      ? '尚未执行图片识别。'
                      : inputMode === 'video'
                        ? (streamState === 'exporting' ? '正在导出结果视频...' : '尚未启动视频实时识别。')
                        : '尚未启动摄像头实时识别。'
                  }
                />
              ) : null}
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
