import {
  startTransition,
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from 'react'
import helpIcon from './assets/help-circle.svg?raw'
import moonIcon from './assets/moon.svg?raw'
import sunIcon from './assets/sun.svg?raw'
import { AppGlobalStyle } from './app/AppGlobalStyle'
import { Footer } from './components/Footer'
import { HelpModal } from './components/HelpModal'
import { HomePage } from './components/HomePage'
import { PreviewZoomModal } from './components/PreviewZoomModal'
import { TopBar } from './components/TopBar'
import { WorkspacePage } from './components/WorkspacePage'
import { WorkspaceSidebar } from './components/WorkspaceSidebar'
import type { DetectionRun, ImportedModel, RunDetectionOptions } from './lib/onnxRuntime'
import {
  getWebGpuSupportState,
  inspectImportedModel,
  prepareImportedModelSession,
  releaseCachedModelSessions,
  runDetectionOnSource,
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

type InputMode = 'image' | 'video' | 'camera'
type StreamState = 'idle' | 'running' | 'stopping' | 'exporting'
type PageId = 'home' | 'workspace'
type SectionId = 'input-import' | 'inference-settings' | 'results-export'
type PreviewZoomTarget = 'image' | 'video' | 'camera' | 'result-canvas'
type PreviewViewerOffset = { x: number; y: number }
type ThemeMode = 'light' | 'dark' | 'system'
type ResolvedTheme = 'light' | 'dark'

type DetectionItem = {
  label: string
  confidence: number
  boxSummary: string
}

type CameraDeviceOption = {
  deviceId: string
  label: string
}

const DEFAULT_DETECTION_THRESHOLD = 0.8
const DEFAULT_MAX_DETECTIONS = 0
const FPS_WINDOW_MS = 1000
const PREVIEW_VIEWER_DEFAULT_SCALE = 1
const PREVIEW_VIEWER_MIN_SCALE = 0.25
const PREVIEW_VIEWER_MAX_SCALE = 4
const PREVIEW_VIEWER_WHEEL_STEP = 0.2
const PREVIEW_VIEWER_DEFAULT_OFFSET: PreviewViewerOffset = { x: 0, y: 0 }
const THEME_STORAGE_KEY = '4cimageseg-theme-mode'
const NAV_ITEMS: ReadonlyArray<{ id: SectionId; label: string; description: string }> = [
  { id: 'input-import', label: '输入与导入', description: '输入源、模型与配置' },
  { id: 'inference-settings', label: '推理设置', description: '阈值、数量与执行控制' },
  { id: 'results-export', label: '结果与导出', description: '预览、结果与导出入口' },
]

const FOOTER_COPYRIGHT_TEXT = `Copyright © ${new Date().getFullYear()} phantomfancy`

function App() {
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => readThemeMode())
  const [systemTheme, setSystemTheme] = useState<ResolvedTheme>(() => getSystemTheme())
  const [inputMode, setInputMode] = useState<InputMode>('image')
  const [activePageId, setActivePageId] = useState<PageId>('home')
  const [activeSectionId, setActiveSectionId] = useState<SectionId>('input-import')
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(false)
  const [isThemeMenuOpen, setIsThemeMenuOpen] = useState(false)
  const [isHelpOpen, setIsHelpOpen] = useState(false)
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
  const [isPreviewZoomOpen, setIsPreviewZoomOpen] = useState(false)
  const [previewViewerScale, setPreviewViewerScale] = useState(PREVIEW_VIEWER_DEFAULT_SCALE)
  const [previewViewerOffset, setPreviewViewerOffset] = useState<PreviewViewerOffset>(PREVIEW_VIEWER_DEFAULT_OFFSET)
  const [isPreviewViewerFullscreen, setIsPreviewViewerFullscreen] = useState(false)

  const themeMenuRef = useRef<HTMLDivElement | null>(null)
  const sourceVideoRef = useRef<HTMLVideoElement | null>(null)
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null)
  const resultCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const previewZoomDialogRef = useRef<HTMLDivElement | null>(null)
  const previewZoomViewportRef = useRef<HTMLDivElement | null>(null)
  const previewZoomMediaRef = useRef<HTMLDivElement | null>(null)
  const previewZoomVideoRef = useRef<HTMLVideoElement | null>(null)
  const previewZoomCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const cameraStreamRef = useRef<MediaStream | null>(null)
  const frameLoopTokenRef = useRef(0)
  const frameInFlightRef = useRef(false)
  const frameTimestampsRef = useRef<number[]>([])
  const previewPointerDragRef = useRef<{
    originX: number
    originY: number
    pointerId: number
  } | null>(null)

  const resetPreviewViewer = useCallback(() => {
    setIsPreviewZoomOpen(false)
    setPreviewViewerScale(PREVIEW_VIEWER_DEFAULT_SCALE)
    setPreviewViewerOffset(PREVIEW_VIEWER_DEFAULT_OFFSET)
    setIsPreviewViewerFullscreen(false)
    previewPointerDragRef.current = null
  }, [])
  const resolvedTheme = resolveThemeMode(themeMode, systemTheme)

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

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const applyThemePreference = (nextMatches: boolean) => {
      setSystemTheme(nextMatches ? 'dark' : 'light')
    }

    applyThemePreference(mediaQuery.matches)

    const handleChange = (event: MediaQueryListEvent) => {
      applyThemePreference(event.matches)
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => {
      mediaQuery.removeEventListener('change', handleChange)
    }
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = resolvedTheme
    document.documentElement.style.colorScheme = resolvedTheme
    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode)

    return () => {
      delete document.documentElement.dataset.theme
      document.documentElement.style.colorScheme = ''
    }
  }, [resolvedTheme, themeMode])

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
  const displayedRuntimeMessage =
    runtimeMessage ||
    (importedModel && !isWebGpuSupported ? (webGpuSupportState.message ?? '') : '')
  const displayedPendingThreshold = pendingDetectionThreshold.toFixed(2)
  const displayedAppliedMaxDetections = String(appliedMaxDetections)
  const displayedFps = streamFps === null ? '-' : streamFps.toFixed(1)
  const previewZoomScaleLabel = `${Math.round(previewViewerScale * 100)}%`
  const currentTimeLabel = inputMode === 'camera' && streamState === 'running'
    ? 'Live'
    : currentSourceTime === null
      ? '-'
      : `${currentSourceTime.toFixed(2)} s`
  const availablePreviewZoomTarget = resolvePreviewZoomTarget({
    hasRenderedResult,
    imagePreviewUrl,
    inputMode,
    streamState,
    videoPreviewUrl,
  })
  const previewZoomTarget = isPreviewZoomOpen ? availablePreviewZoomTarget : null
  const canOpenPreviewZoom = availablePreviewZoomTarget !== null
  const isScalablePreviewTarget =
    previewZoomTarget !== null &&
    isPreviewScalableTarget(previewZoomTarget)
  const previewOpenLabel = availablePreviewZoomTarget === null
    ? '打开预览'
    : isPreviewScalableTarget(availablePreviewZoomTarget)
      ? '打开查看器'
      : '打开播放器'
  const currentThemeIcon = resolvedTheme === 'dark' ? moonIcon : sunIcon
  const currentThemeLabel = themeMode === 'system'
    ? `跟随系统 · ${resolvedTheme === 'dark' ? '深色' : '浅色'}`
    : themeMode === 'dark'
      ? '深色模式'
      : '浅色模式'
  const isAnyOverlayOpen = isPreviewZoomOpen || isHelpOpen

  const canRunImage =
    inputMode === 'image' &&
    Boolean(imageFile && importedModel && isWebGpuSupported && !modelBusy && !imageDetectBusy && streamState === 'idle')
  const canChangeModel = !modelBusy && !imageDetectBusy && streamState === 'idle'
  const canExportImage =
    inputMode === 'image' &&
    hasRenderedResult &&
    !imageDetectBusy &&
    streamState === 'idle'
  const canStartVideo =
    inputMode === 'video' &&
    Boolean(videoFile && importedModel && isWebGpuSupported && streamState === 'idle')
  const canStopVideo =
    inputMode === 'video' &&
    streamState === 'running'
  const canExportVideo =
    inputMode === 'video' &&
    Boolean(videoFile && importedModel && isWebGpuSupported && streamState === 'idle' && supportsVideoExport)
  const canStartCamera =
    inputMode === 'camera' &&
    Boolean(importedModel && isWebGpuSupported && streamState === 'idle' && !cameraBusy && supportsCamera)
  const canStopCamera =
    inputMode === 'camera' &&
    streamState === 'running'
  const canAdjustInferenceSettings = Boolean(
    importedModel &&
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
    resetPreviewViewer()
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

  function rejectModelChangeWhileBusy(resetNativeInput?: () => void): boolean {
    if (canChangeModel) {
      return false
    }

    setStatusMessage('当前推理或模型处理尚未结束，请等待完成后再切换模型。')
    resetNativeInput?.()
    return true
  }

  const stopActiveStream = useCallback((statusText?: string) => {
    resetPreviewViewer()
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
  }, [resetPreviewViewer])

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

  useEffect(() => () => {
    void releaseCachedModelSessions()
  }, [])

  useEffect(() => {
    if (!isPreviewZoomOpen && !isHelpOpen && !isThemeMenuOpen) {
      return
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') {
        return
      }

      if (isPreviewZoomOpen) {
        if (document.fullscreenElement) {
          void document.exitFullscreen().catch(() => {})
        }

        resetPreviewViewer()
        return
      }

      if (isHelpOpen) {
        setIsHelpOpen(false)
        return
      }

      if (isThemeMenuOpen) {
        setIsThemeMenuOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [isHelpOpen, isPreviewZoomOpen, isThemeMenuOpen, resetPreviewViewer])

  useEffect(() => {
    if (!isAnyOverlayOpen) {
      return
    }

    const { body, documentElement } = document
    const previousBodyOverflow = body.style.overflow
    const previousBodyTouchAction = body.style.touchAction
    const previousHtmlOverflow = documentElement.style.overflow
    const previousHtmlOverscrollBehavior = documentElement.style.overscrollBehavior

    body.style.overflow = 'hidden'
    body.style.touchAction = 'none'
    documentElement.style.overflow = 'hidden'
    documentElement.style.overscrollBehavior = 'none'

    return () => {
      body.style.overflow = previousBodyOverflow
      body.style.touchAction = previousBodyTouchAction
      documentElement.style.overflow = previousHtmlOverflow
      documentElement.style.overscrollBehavior = previousHtmlOverscrollBehavior
    }
  }, [isAnyOverlayOpen])

  useEffect(() => {
    if (!isThemeMenuOpen) {
      return
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (themeMenuRef.current?.contains(event.target as Node)) {
        return
      }

      setIsThemeMenuOpen(false)
    }

    window.addEventListener('pointerdown', handlePointerDown)
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown)
    }
  }, [isThemeMenuOpen])

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsPreviewViewerFullscreen(Boolean(document.fullscreenElement))
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    handleFullscreenChange()
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
    }
  }, [])

  useEffect(() => {
    if (!isPreviewZoomOpen || previewZoomTarget !== 'result-canvas') {
      return
    }

    if (!resultCanvasRef.current) {
      return
    }

    syncCanvas(resultCanvasRef.current, previewZoomCanvasRef.current)
  }, [isPreviewZoomOpen, previewZoomTarget, detectionItems, hasRenderedResult, streamFps])

  useEffect(() => {
    if (!isPreviewZoomOpen || !isScalablePreviewTarget) {
      return
    }

    setPreviewViewerOffset((currentValue) =>
      clampPreviewViewerOffset(
        currentValue,
        previewViewerScale,
        previewZoomViewportRef.current,
        previewZoomMediaRef.current,
      ))
  }, [isPreviewZoomOpen, isScalablePreviewTarget, previewViewerScale, previewZoomTarget])

  useEffect(() => {
    if (!isPreviewZoomOpen || !previewZoomTarget || previewZoomTarget === 'image' || previewZoomTarget === 'result-canvas') {
      return
    }

    const modalVideo = previewZoomVideoRef.current
    if (!modalVideo) {
      return
    }

    modalVideo.playsInline = true
    modalVideo.muted = true

    if (previewZoomTarget === 'video') {
      modalVideo.srcObject = null
      modalVideo.src = videoPreviewUrl
      if (sourceVideoRef.current) {
        modalVideo.currentTime = sourceVideoRef.current.currentTime
      }
      if (sourceVideoRef.current && !sourceVideoRef.current.paused) {
        void modalVideo.play().catch(() => {})
      }
    } else if (previewZoomTarget === 'camera') {
      modalVideo.removeAttribute('src')
      modalVideo.srcObject = cameraStreamRef.current
      void modalVideo.play().catch(() => {})
    }

    return () => {
      modalVideo.pause()
      modalVideo.srcObject = null
      modalVideo.removeAttribute('src')
      modalVideo.load()
    }
  }, [isPreviewZoomOpen, previewZoomTarget, streamState, videoPreviewUrl])

  useEffect(() => {
    if (typeof window === 'undefined' || typeof IntersectionObserver === 'undefined') {
      return
    }

    if (activePageId !== 'workspace') {
      return
    }

    const sections = NAV_ITEMS
      .map((item) => document.getElementById(item.id))
      .filter((section): section is HTMLElement => section instanceof HTMLElement)

    if (sections.length === 0) {
      return
    }

    const visibleSections = new Map<SectionId, number>()
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const sectionId = entry.target.id as SectionId
          if (entry.isIntersecting) {
            visibleSections.set(sectionId, entry.intersectionRatio)
            return
          }

          visibleSections.delete(sectionId)
        })

        const nextActiveSection = NAV_ITEMS
          .map((item, index) => ({
            id: item.id,
            index,
            ratio: visibleSections.get(item.id) ?? -1,
          }))
          .filter((item) => item.ratio >= 0)
          .sort((left, right) => right.ratio - left.ratio || left.index - right.index)[0]

        if (nextActiveSection) {
          setActiveSectionId((currentValue) =>
            currentValue === nextActiveSection.id ? currentValue : nextActiveSection.id)
        }
      },
      {
        rootMargin: '-18% 0px -54% 0px',
        threshold: [0.12, 0.3, 0.48, 0.72],
      },
    )

    sections.forEach((section) => {
      observer.observe(section)
    })

    return () => {
      observer.disconnect()
    }
  }, [activePageId])

  async function handleOnnxSelection(file: File | null) {
    let shouldResetNativeInput = false
    if (rejectModelChangeWhileBusy(() => setOnnxInputKey((value) => value + 1))) {
      return
    }

    stopActiveStream()
    clearImportedModelState()
    resetSidecarSelections()
    setOnnxModelDraft(null)
    setModelBusy(true)
    setStatusMessage('正在释放旧模型推理资源...')
    await releaseCachedModelSessions()

    if (!file) {
      setStatusMessage('尚未选择 ONNX 模型文件。')
      setOnnxInputKey((value) => value + 1)
      setModelBusy(false)
      return
    }

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
      setStatusMessage(`正在预热 ONNX 模型 ${file.name} 的推理会话...`)
      await prepareImportedModelSession(model)
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
      setStatusMessage(`模型解析或会话预热失败：${formatError(error)}`)
    } finally {
      setModelBusy(false)
      if (shouldResetNativeInput) {
        setOnnxInputKey((value) => value + 1)
      }
    }
  }

  function handleNavigate(sectionId: SectionId) {
    const section = document.getElementById(sectionId)
    if (!section) {
      return
    }

    setIsThemeMenuOpen(false)
    setActiveSectionId(sectionId)
    section.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    })
  }

  function handlePageNavigate(pageId: PageId) {
    setIsThemeMenuOpen(false)
    setActivePageId(pageId)

    if (pageId === 'home') {
      setIsSidebarExpanded(false)
      window.scrollTo({ top: 0, behavior: 'smooth' })
      return
    }

    setActiveSectionId('input-import')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  function openPreviewZoom() {
    if (!availablePreviewZoomTarget) {
      return
    }

    setIsThemeMenuOpen(false)
    setIsHelpOpen(false)
    setPreviewViewerScale(PREVIEW_VIEWER_DEFAULT_SCALE)
    setPreviewViewerOffset(PREVIEW_VIEWER_DEFAULT_OFFSET)
    setIsPreviewZoomOpen(true)
  }

  function closePreviewZoom() {
    if (document.fullscreenElement) {
      void document.exitFullscreen().catch(() => {})
    }

    resetPreviewViewer()
  }

  function setPreviewViewerScaleValue(nextScale: number) {
    const normalizedScale = clampPreviewZoomScale(nextScale)
    setPreviewViewerScale(normalizedScale)

    if (normalizedScale <= PREVIEW_VIEWER_DEFAULT_SCALE) {
      setPreviewViewerOffset(PREVIEW_VIEWER_DEFAULT_OFFSET)
      return
    }

    setPreviewViewerOffset((currentValue) =>
      clampPreviewViewerOffset(
        currentValue,
        normalizedScale,
        previewZoomViewportRef.current,
        previewZoomMediaRef.current,
      ))
  }

  function handlePreviewWheel(event: ReactWheelEvent<HTMLDivElement>) {
    if (!isScalablePreviewTarget) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    const direction = event.deltaY < 0 ? 1 : -1
    setPreviewViewerScaleValue(previewViewerScale + direction * PREVIEW_VIEWER_WHEEL_STEP)
  }

  function handlePreviewPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (!isScalablePreviewTarget || previewViewerScale <= PREVIEW_VIEWER_DEFAULT_SCALE) {
      return
    }

    previewPointerDragRef.current = {
      originX: event.clientX - previewViewerOffset.x,
      originY: event.clientY - previewViewerOffset.y,
      pointerId: event.pointerId,
    }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  function handlePreviewPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    if (!isScalablePreviewTarget) {
      return
    }

    const dragState = previewPointerDragRef.current
    if (!dragState || dragState.pointerId !== event.pointerId) {
      return
    }

    setPreviewViewerOffset(
      clampPreviewViewerOffset(
        {
          x: event.clientX - dragState.originX,
          y: event.clientY - dragState.originY,
        },
        previewViewerScale,
        previewZoomViewportRef.current,
        previewZoomMediaRef.current,
      ))
  }

  function handlePreviewPointerRelease(event: ReactPointerEvent<HTMLDivElement>) {
    const dragState = previewPointerDragRef.current
    if (!dragState || dragState.pointerId !== event.pointerId) {
      return
    }

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }

    previewPointerDragRef.current = null
  }

  function handlePreviewMediaDoubleClick() {
    if (!isScalablePreviewTarget) {
      return
    }

    setPreviewViewerScaleValue(
      previewViewerScale > PREVIEW_VIEWER_DEFAULT_SCALE
        ? PREVIEW_VIEWER_DEFAULT_SCALE
        : 2)
  }

  async function togglePreviewViewerFullscreen() {
    const dialogElement = previewZoomDialogRef.current
    if (!dialogElement) {
      return
    }

    if (document.fullscreenElement) {
      await document.exitFullscreen().catch(() => {})
      return
    }

    await dialogElement.requestFullscreen?.().catch(() => {})
  }

  async function handleConfigSelection(file: File | null) {
    let shouldResetNativeInput = false
    if (rejectModelChangeWhileBusy(() => setConfigInputKey((value) => value + 1))) {
      return
    }

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
    clearImportedModelState()
    await releaseCachedModelSessions()

    try {
      const model = await inspectImportedModel({
        onnxModel: onnxModelDraft,
        configFile: file,
        preprocessorConfigFile: preprocessorConfigFile ?? undefined,
      })
      setStatusMessage(`正在预热 ${onnxModelDraft.fileName} 的推理会话...`)
      await prepareImportedModelSession(model)
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
      setStatusMessage(`config.json 导入或会话预热失败：${formatError(error)}`)
    } finally {
      setModelBusy(false)
      if (shouldResetNativeInput) {
        setConfigInputKey((value) => value + 1)
      }
    }
  }

  async function handlePreprocessorSelection(file: File | null) {
    let shouldResetNativeInput = false
    if (rejectModelChangeWhileBusy(() => setPreprocessorInputKey((value) => value + 1))) {
      return
    }

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
    clearImportedModelState()
    await releaseCachedModelSessions()

    try {
      const model = await inspectImportedModel({
        onnxModel: onnxModelDraft,
        configFile,
        preprocessorConfigFile: file,
      })
      setStatusMessage(`正在预热 ${onnxModelDraft.fileName} 的推理会话...`)
      await prepareImportedModelSession(model)
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
      setStatusMessage(`preprocessor_config.json 导入或会话预热失败：${formatError(error)}`)
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

    if (rejectModelChangeWhileBusy()) {
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

      clearImportedModelState()
      await releaseCachedModelSessions()
      const model = await inspectImportedModel({
        onnxModel: onnxModelDraft,
        configFile: nextConfigFile,
        preprocessorConfigFile: nextPreprocessorConfigFile ?? undefined,
      })
      setStatusMessage(`正在预热 ${onnxModelDraft.fileName} 的推理会话...`)
      await prepareImportedModelSession(model)
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
        setStatusMessage(`自动查找或会话预热失败：${formatError(error)}`)
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
          const detectionRun = await runDetectionOnSource(
            importedModel,
            videoElement,
            buildFrameSourceLabel(sourceKind, videoElement.currentTime, videoFile?.name, selectedCameraId, cameraDevices),
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

  function applyDetectionRun(
    detectionRun: DetectionRun,
    timeSeconds: number | null = null,
    exportCanvas?: HTMLCanvasElement,
    fps: number | null = null,
  ) {
    syncCanvas(detectionRun.annotatedCanvas, resultCanvasRef.current)
    drawRuntimeOverlay(resultCanvasRef.current, fps)
    if (isPreviewZoomOpen && previewZoomTarget === 'result-canvas') {
      syncCanvas(detectionRun.annotatedCanvas, previewZoomCanvasRef.current)
      drawRuntimeOverlay(previewZoomCanvasRef.current, fps)
    }
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

  function openHelpDialog() {
    setIsThemeMenuOpen(false)
    setIsHelpOpen(true)
  }

  function handleThemeModeSelect(mode: ThemeMode) {
    setThemeMode(mode)
    setIsThemeMenuOpen(false)
  }
  return (
    <>
      <AppGlobalStyle />
      <div className="app-shell">
        {activePageId === 'workspace' ? (
          <WorkspaceSidebar
            activeSectionId={activeSectionId}
            isExpanded={isSidebarExpanded}
            onNavigate={handleNavigate}
            onToggle={() => {
              setIsSidebarExpanded((currentValue) => !currentValue)
            }}
          />
        ) : null}
        <div className="page-layout">
          <TopBar
            activePageId={activePageId}
            currentThemeIcon={currentThemeIcon}
            currentThemeLabel={currentThemeLabel}
            helpIconMarkup={helpIcon}
            isThemeMenuOpen={isThemeMenuOpen}
            onHelpOpen={openHelpDialog}
            onPageNavigate={handlePageNavigate}
            onThemeMenuToggle={() => {
              setIsThemeMenuOpen((currentValue) => !currentValue)
            }}
            onThemeModeSelect={handleThemeModeSelect}
            themeMenuRef={themeMenuRef}
            themeMode={themeMode}
          />
          <div className="surface">
            <main className="shell">
              {activePageId === 'home' ? (
                <HomePage
                  onEnterWorkspace={() => {
                    handlePageNavigate('workspace')
                  }}
                  onHelpOpen={openHelpDialog}
                />
              ) : null}
              {activePageId === 'workspace' ? (
                <WorkspacePage
                  appliedDetectionThreshold={appliedDetectionThreshold}
                  cameraBusy={cameraBusy}
                  cameraDevices={cameraDevices}
                  cameraVideoRef={cameraVideoRef}
                  canAdjustInferenceSettings={canAdjustInferenceSettings}
                  canChangeModel={canChangeModel}
                  canExportImage={canExportImage}
                  canExportVideo={canExportVideo}
                  canOpenPreviewZoom={canOpenPreviewZoom}
                  canRunImage={canRunImage}
                  canStartCamera={canStartCamera}
                  canStartVideo={canStartVideo}
                  canStopCamera={canStopCamera}
                  canStopVideo={canStopVideo}
                  configInputKey={configInputKey}
                  currentTimeLabel={currentTimeLabel}
                  detectionItems={detectionItems}
                  displayedAppliedMaxDetections={displayedAppliedMaxDetections}
                  displayedContract={displayedContract}
                  displayedFps={displayedFps}
                  displayedPendingThreshold={displayedPendingThreshold}
                  displayedProvider={displayedProvider}
                  displayedRuntimeMessage={displayedRuntimeMessage}
                  displayedSidecars={displayedSidecars}
                  discoverBusy={discoverBusy}
                  formatDims={formatDims}
                  hasRenderedResult={hasRenderedResult}
                  imageDetectBusy={imageDetectBusy}
                  imageFileName={imageFile?.name ?? ''}
                  imagePreviewUrl={imagePreviewUrl}
                  importControls={importControls}
                  importedModelExists={Boolean(importedModel)}
                  inputMode={inputMode}
                  modelBusy={modelBusy}
                  onAutoDiscoverSidecars={handleAutoDiscoverSidecars}
                  onCommitDetectionThreshold={commitDetectionThreshold}
                  onCommitMaxDetections={commitMaxDetections}
                  onConfigSelection={handleConfigSelection}
                  onExportImage={handleExportImage}
                  onExportVideo={handleExportVideo}
                  onImageSelection={updateImageFile}
                  onMaxDetectionsInputChange={setPendingMaxDetectionsInput}
                  onModeChange={handleInputModeChange}
                  onOnnxSelection={handleOnnxSelection}
                  onOpenPreviewZoom={openPreviewZoom}
                  onPreprocessorSelection={handlePreprocessorSelection}
                  onRunImageDetection={handleRunImageDetection}
                  onSelectedCameraChange={setSelectedCameraId}
                  onStartCameraDetection={handleStartCameraDetection}
                  onStartVideoDetection={handleStartVideoDetection}
                  onStopCamera={() => {
                    stopActiveStream('已停止摄像头实时识别。')
                  }}
                  onStopVideo={() => {
                    stopActiveStream('已停止视频实时识别。')
                  }}
                  onThresholdChange={(value) => {
                    setPendingDetectionThreshold(clampDetectionThreshold(value))
                  }}
                  onVideoSelection={updateVideoFile}
                  onnxInputKey={onnxInputKey}
                  onnxModelDraft={onnxModelDraft}
                  pendingDetectionThreshold={pendingDetectionThreshold}
                  pendingMaxDetectionsInput={pendingMaxDetectionsInput}
                  preprocessorInputKey={preprocessorInputKey}
                  previewOpenLabel={previewOpenLabel}
                  resultCanvasRef={resultCanvasRef}
                  resultProvider={resultProvider}
                  selectedCameraId={selectedCameraId}
                  sourceVideoRef={sourceVideoRef}
                  statusMessage={statusMessage}
                  streamState={streamState}
                  supportsCamera={supportsCamera}
                  supportsDirectoryPicker={supportsDirectoryPicker}
                  supportsVideoExport={supportsVideoExport}
                  videoDownloadFileName={buildVideoExportFileName(videoFile?.name ?? 'result')}
                  videoDownloadUrl={videoDownloadUrl}
                  videoFileName={videoFile?.name ?? ''}
                  videoPreviewUrl={videoPreviewUrl}
                />
              ) : null}
            </main>
          </div>
          <Footer copyrightText={FOOTER_COPYRIGHT_TEXT} />
        </div>
        {isHelpOpen ? <HelpModal onClose={() => { setIsHelpOpen(false) }} /> : null}
        {isPreviewZoomOpen && previewZoomTarget ? (
          <PreviewZoomModal
            closePreviewZoom={closePreviewZoom}
            currentTimeLabel={currentTimeLabel}
            detectionItemCount={detectionItems.length}
            handlePreviewMediaDoubleClick={handlePreviewMediaDoubleClick}
            handlePreviewPointerDown={handlePreviewPointerDown}
            handlePreviewPointerMove={handlePreviewPointerMove}
            handlePreviewPointerRelease={handlePreviewPointerRelease}
            handlePreviewWheel={handlePreviewWheel}
            imagePreviewUrl={imagePreviewUrl}
            inputMode={inputMode}
            isPreviewViewerFullscreen={isPreviewViewerFullscreen}
            isScalablePreviewTarget={isScalablePreviewTarget}
            previewViewerOffset={previewViewerOffset}
            previewViewerScale={previewViewerScale}
            previewZoomCanvasRef={previewZoomCanvasRef}
            previewZoomDialogRef={previewZoomDialogRef}
            previewZoomMediaRef={previewZoomMediaRef}
            previewZoomScaleLabel={previewZoomScaleLabel}
            previewZoomTarget={previewZoomTarget}
            previewZoomVideoRef={previewZoomVideoRef}
            previewZoomViewportRef={previewZoomViewportRef}
            resultProvider={resultProvider}
            scaleDecreaseDisabled={previewViewerScale <= PREVIEW_VIEWER_MIN_SCALE + 0.01}
            scaleIncreaseDisabled={previewViewerScale >= PREVIEW_VIEWER_MAX_SCALE - 0.01}
            setPreviewViewerDefault={() => {
              setPreviewViewerScaleValue(PREVIEW_VIEWER_DEFAULT_SCALE)
              setPreviewViewerOffset(PREVIEW_VIEWER_DEFAULT_OFFSET)
            }}
            setPreviewViewerScaleStepDown={() => {
              setPreviewViewerScaleValue(previewViewerScale - PREVIEW_VIEWER_WHEEL_STEP)
            }}
            setPreviewViewerScaleStepUp={() => {
              setPreviewViewerScaleValue(previewViewerScale + PREVIEW_VIEWER_WHEEL_STEP)
            }}
            togglePreviewViewerFullscreen={togglePreviewViewerFullscreen}
          />
        ) : null}
      </div>
    </>
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

  return sidecars.length > 0
    ? `模型解析成功：${model.contract.family}，已使用 ${sidecars.join('、')}，当前执行提供器为 ${model.providerName}。${runtimeNote}`
    : `模型解析成功：${model.contract.family}，当前执行提供器为 ${model.providerName}。${runtimeNote}`
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

  if (source === target) {
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

function resolvePreviewZoomTarget(input: {
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

function readThemeMode(): ThemeMode {
  if (typeof window === 'undefined') {
    return 'system'
  }

  const storedValue = window.localStorage.getItem(THEME_STORAGE_KEY)
  return storedValue === 'light' || storedValue === 'dark' || storedValue === 'system'
    ? storedValue
    : 'system'
}

function getSystemTheme(): ResolvedTheme {
  if (typeof window === 'undefined') {
    return 'light'
  }

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function resolveThemeMode(
  themeMode: ThemeMode,
  systemTheme: ResolvedTheme,
): ResolvedTheme {
  return themeMode === 'system' ? systemTheme : themeMode
}

function isPreviewScalableTarget(target: PreviewZoomTarget): boolean {
  return target === 'image' || target === 'result-canvas'
}

function clampPreviewZoomScale(value: number): number {
  return Math.min(Math.max(value, PREVIEW_VIEWER_MIN_SCALE), PREVIEW_VIEWER_MAX_SCALE)
}

function clampPreviewViewerOffset(
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

