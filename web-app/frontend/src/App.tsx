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
import './App.css'

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
const PAGE_NAV_ITEMS: ReadonlyArray<{ id: PageId; label: string }> = [
  { id: 'home', label: '首页' },
  { id: 'workspace', label: '工作台' },
]
const NAV_ITEMS: ReadonlyArray<{ id: SectionId; label: string; description: string }> = [
  { id: 'input-import', label: '输入与导入', description: '输入源、模型与配置' },
  { id: 'inference-settings', label: '推理设置', description: '阈值、数量与执行控制' },
  { id: 'results-export', label: '结果与导出', description: '预览、结果与导出入口' },
]

const PROJECT_REPOSITORY_URL = 'https://github.com/phantomfancy/ImageSeg'
const PROJECT_LICENSE_URL = `${PROJECT_REPOSITORY_URL}/blob/master/LICENSE`
const FOOTER_CONTACT_EMAIL = 'contact@4cimageseg.local'
const FOOTER_FILING_NUMBER = '备案号待补充'
const FOOTER_CERTIFICATION_INFO = '认证信息待补充'
const FOOTER_COPYRIGHT_TEXT = `Copyright © ${new Date().getFullYear()} ★ASU-全域智侦`
const HELP_CONTENT: ReadonlyArray<{ title: string; body: string }> = [
  {
    title: '输入模式',
    body: '图片模式适合单张检测，视频模式适合回放与导出，摄像头模式适合实时识别与现场观察。',
  },
  {
    title: '模型与配置',
    body: `先导入 ONNX 模型；如识别为 Hugging Face 风格模型，再补充 ${CONFIG_FILE_NAME}，必要时补充 ${PREPROCESSOR_CONFIG_FILE_NAME}。`,
  },
  {
    title: '推理设置',
    body: '推理阈值用于控制结果过滤强度，识别数量用于限制绘制数量；新值在提交后才会真正生效。',
  },
  {
    title: '结果查看与导出',
    body: '统一预览区会在原始输入和叠加结果之间切换，结果区会同步显示检测项列表，并提供图片或视频导出入口。',
  },
  {
    title: '查看器操作',
    body: '图像和结果查看器支持滚轮缩放、拖拽平移、双击切换；视频与摄像头预览则保留更接近标准播放器的操作方式。',
  },
]

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
  const displayedAppliedThreshold = appliedDetectionThreshold.toFixed(2)
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

  return (
    <div className="app-shell">
      {activePageId === 'workspace' ? (
        <aside className={`sidebar${isSidebarExpanded ? ' sidebar--expanded' : ''}`}>
          <button
            type="button"
            className="sidebar__toggle"
            aria-expanded={isSidebarExpanded}
            aria-controls="workspace-navigation"
            aria-label={isSidebarExpanded ? '收起侧栏导航' : '展开侧栏导航'}
            onClick={() => {
              setIsSidebarExpanded((currentValue) => !currentValue)
            }}
          >
            <span className="sidebar__toggle-icon" aria-hidden="true">
              {isSidebarExpanded ? '×' : '≡'}
            </span>
            <span className="sidebar__toggle-text">{isSidebarExpanded ? '收起' : '导航'}</span>
          </button>

          <div className="sidebar__panel" aria-hidden={!isSidebarExpanded}>
            <nav
              id="workspace-navigation"
              className="sidebar__nav"
              aria-label="工作台区块导航"
            >
              {NAV_ITEMS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`sidebar__nav-item${activeSectionId === item.id ? ' sidebar__nav-item--active' : ''}`}
                  aria-current={activeSectionId === item.id ? 'location' : undefined}
                  onClick={() => {
                    handleNavigate(item.id)
                  }}
                >
                  <span className="sidebar__nav-label">{item.label}</span>
                </button>
              ))}
            </nav>
          </div>
        </aside>
      ) : null}

      <div className="page-layout">
        <header className="topbar">
          <div className="topbar__brand">
            <div className="topbar__title">ImageSeg-ai装备识别工具</div>
          </div>

          <nav className="topbar__page-nav" aria-label="页面导航">
            {PAGE_NAV_ITEMS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`topbar__nav-item${activePageId === item.id ? ' topbar__nav-item--active' : ''}`}
                aria-current={activePageId === item.id ? 'page' : undefined}
                onClick={() => {
                  handlePageNavigate(item.id)
                }}
              >
                {item.label}
              </button>
            ))}
          </nav>

          <div className="topbar__action-list">
            <div className="topbar__menu-anchor" ref={themeMenuRef}>
              <button
                type="button"
                className="topbar__action-button"
                aria-haspopup="menu"
                aria-expanded={isThemeMenuOpen}
                aria-label={`主题切换，当前为${currentThemeLabel}`}
                title={`主题切换，当前为${currentThemeLabel}`}
                onClick={() => {
                  setIsThemeMenuOpen((currentValue) => !currentValue)
                }}
              >
                <SvgIcon markup={currentThemeIcon} />
                <span>主题</span>
              </button>

              {isThemeMenuOpen ? (
                <div className="topbar__menu" role="menu" aria-label="主题切换">
                  {([
                    ['dark', '深色'],
                    ['light', '浅色'],
                    ['system', '跟随系统'],
                  ] as const).map(([mode, label]) => (
                    <button
                      key={mode}
                      type="button"
                      role="menuitemradio"
                      aria-checked={themeMode === mode}
                      className={`topbar__menu-item${themeMode === mode ? ' topbar__menu-item--active' : ''}`}
                      onClick={() => {
                        setThemeMode(mode)
                        setIsThemeMenuOpen(false)
                      }}
                    >
                      <span>{label}</span>
                      {themeMode === mode ? <span className="topbar__menu-check">当前</span> : null}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>

            <button
              type="button"
              className="topbar__action-button"
              aria-label="帮助说明"
              title="帮助说明"
              onClick={() => {
                setIsThemeMenuOpen(false)
                setIsHelpOpen(true)
              }}
            >
              <SvgIcon markup={helpIcon} />
              <span>帮助</span>
            </button>
          </div>
        </header>

        <div className="surface">
          <main className="shell">
            {activePageId === 'home' ? (
              <div className="home-page">
                <section className="hero hero--home">
                  <div className="hero__content">
                    <div className="hero__eyebrow">ImageSeg-ai装备识别工具</div>
                    <h1>ai装备识别工具</h1>
                    <p className="hero__copy">
                      面向单图、视频与摄像头场景的本地 ONNX 推理工作台，使用基于CNN或Transformers的目标检测模型进行图像推理。
                      导入、推理编排、结果解码和导出全部在前端完成，适合快速验证装备识别模型。
                    </p>
                    <div className="hero__actions">
                      <button
                        type="button"
                        className="hero__action hero__action--primary"
                        onClick={() => {
                          handlePageNavigate('workspace')
                        }}
                      >
                        进入工作台
                      </button>
                      <button
                        type="button"
                        className="hero__action hero__action--secondary"
                        onClick={() => {
                          setIsThemeMenuOpen(false)
                          setIsHelpOpen(true)
                        }}
                      >
                        查看使用说明
                      </button>
                    </div>
                  </div>

                  <div className="hero__signal" aria-label="当前能力摘要">
                    <div className="hero__signal-card">
                      <span className="hero__signal-label">输入源</span>
                      <strong>图片 / 视频 / 摄像头</strong>
                    </div>
                    <div className="hero__signal-card">
                      <span className="hero__signal-label">推理后端</span>
                      <strong>ONNX Runtime WebGPU</strong>
                    </div>
                    <div className="hero__signal-card">
                      <span className="hero__signal-label">模型契约</span>
                      <strong>统一解码与预处理</strong>
                    </div>
                  </div>
                </section>

                <section className="home-grid" aria-label="首页功能概览">
                  <article className="home-card">
                    <div className="home-card__index">01</div>
                    <h2>纯前端推理</h2>
                    <p>输入源、ONNX 会话、结果绘制和导出都在浏览器端完成，减少服务端部署依赖。</p>
                  </article>
                  <article className="home-card">
                    <div className="home-card__index">02</div>
                    <h2>统一模型输出</h2>
                    <p>通过 Contracts 统一不同检测模型的预处理、输出签名和解码逻辑，降低模型切换成本。</p>
                  </article>
                  <article className="home-card">
                    <div className="home-card__index">03</div>
                    <h2>结果复核与导出</h2>
                    <p>同一预览区支持输入源和叠加结果切换，图像查看器可缩放、平移并复核框选细节。</p>
                  </article>
                </section>
              </div>
            ) : null}

            {activePageId === 'workspace' ? (
              <>
          <section className="operation-grid" aria-label="识别工作台">
            <div className="operation-grid__controls panel-stack">
              <article className="panel panel--input" id="input-import" data-nav-section>
                <header className="panel__header">
                  <h2>输入与导入</h2>
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
                    disabled={!canChangeModel}
                    onChange={(event) => {
                      void handleOnnxSelection(event.target.files?.[0] ?? null)
                    }}
                  />
                </label>

                <label className="field">
                  <span className="field__label">{CONFIG_FILE_NAME} {importControls.configRequired ? '(必选)' : '(未启用)'}</span>
                  <input
                    key={configInputKey}
                    type="file"
                    accept=".json,application/json"
                    disabled={!importControls.configEnabled || !canChangeModel}
                    onChange={(event) => {
                      void handleConfigSelection(event.target.files?.[0] ?? null)
                    }}
                  />
                </label>

                <label className="field">
                  <span className="field__label">{PREPROCESSOR_CONFIG_FILE_NAME} {importControls.preprocessorEnabled ? '(可选)' : '(未启用)'}</span>
                  <input
                    key={preprocessorInputKey}
                    type="file"
                    accept=".json,application/json"
                    disabled={!importControls.preprocessorEnabled || !canChangeModel}
                    onChange={(event) => {
                      void handlePreprocessorSelection(event.target.files?.[0] ?? null)
                    }}
                  />
                </label>

                {onnxModelDraft?.family === 'hf-detr-like' ? (
                  <div className="actions actions--stacked">
                    <button
                      type="button"
                      className="action action--secondary"
                      disabled={!importControls.autoDiscoverEnabled || !canChangeModel || discoverBusy}
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
            </div>

            <article className="panel panel--preview operation-grid__preview" id="results-export" data-nav-section>
              <header className="panel__header">
                <h2>统一预览</h2>
              </header>

              <div className="preview-card preview-stage">
                <div className="preview-card__title">当前画面</div>
                <div className="preview-stage__media">
                  {canOpenPreviewZoom ? (
                    <button
                      type="button"
                      className="preview-stage__zoom-button"
                      aria-label={previewOpenLabel}
                      title={previewOpenLabel}
                      onClick={() => {
                        openPreviewZoom()
                      }}
                    >
                      <span className="preview-stage__zoom-icon" aria-hidden="true">+</span>
                    </button>
                  ) : null}

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

            <article className="panel panel--settings operation-grid__settings" id="inference-settings" data-nav-section>
              <header className="panel__header">
                <h2>推理设置</h2>
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
          </section>

          <section className="results-row" aria-label="检测结果">
            <article className="panel panel--detections results-row__panel">
              <header className="panel__header">
                <h2>检测结果</h2>
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

          <article className="panel panel--contract diagnostics-panel diagnostics-row">
            <header className="panel__header">
              <h2>模型契约</h2>
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
              </>
            ) : null}
          </main>
        </div>

        <footer className="footer">
          <div className="footer__section footer__section--brand">
            <div className="footer__title">ImageSeg-ai装备识别工具</div>
            <p className="footer__copy">{FOOTER_COPYRIGHT_TEXT}</p>
            <p className="footer__copy">项目源码依据 AGPL-3.0 许可进行分发与修改。</p>
          </div>

          <div className="footer__section">
            <div className="footer__heading">联系方式</div>
            <a className="footer__link" href={`mailto:${FOOTER_CONTACT_EMAIL}`}>
              {FOOTER_CONTACT_EMAIL}
            </a>
          </div>

          <div className="footer__section">
            <div className="footer__heading">备案与认证</div>
            <div className="footer__meta-list">
              <div className="footer__meta-item">
                <span className="footer__label">备案信息</span>
                <span className="footer__value">{FOOTER_FILING_NUMBER}</span>
              </div>
              <div className="footer__meta-item">
                <span className="footer__label">认证信息</span>
                <span className="footer__value">{FOOTER_CERTIFICATION_INFO}</span>
              </div>
            </div>
          </div>

          <div className="footer__section">
            <div className="footer__heading">项目链接</div>
            <div className="footer__link-list">
              <a
                className="footer__link"
                href={PROJECT_LICENSE_URL}
                target="_blank"
                rel="noreferrer"
              >
                项目 LICENSE
              </a>
              <a
                className="footer__link"
                href={PROJECT_REPOSITORY_URL}
                target="_blank"
                rel="noreferrer"
              >
                项目仓库
              </a>
            </div>
          </div>
        </footer>
      </div>

      {isHelpOpen ? (
        <div className="help-modal" role="dialog" aria-modal="true" aria-label="帮助说明">
          <button
            type="button"
            className="help-modal__backdrop"
            aria-label="关闭帮助说明"
            onClick={() => {
              setIsHelpOpen(false)
            }}
          />

          <div className="help-modal__dialog">
            <div className="help-modal__header">
              <div>
                <div className="help-modal__eyebrow">帮助</div>
                <h2 className="help-modal__title">工作台使用说明</h2>
                <p className="help-modal__copy">
                  这里整理了当前前端工作台最常用的导入、推理、查看与导出说明。
                </p>
              </div>

              <button
                type="button"
                className="help-modal__close"
                onClick={() => {
                  setIsHelpOpen(false)
                }}
              >
                关闭
              </button>
            </div>

            <div className="help-modal__grid">
              {HELP_CONTENT.map((item) => (
                <section className="help-modal__card" key={item.title}>
                  <h3>{item.title}</h3>
                  <p>{item.body}</p>
                </section>
              ))}
            </div>
          </div>
        </div>
      ) : null}

      {isPreviewZoomOpen && previewZoomTarget ? (
        <div className="preview-zoom" role="dialog" aria-modal="true" aria-label="放大预览">
          <button
            type="button"
            className="preview-zoom__backdrop"
            aria-label="关闭放大预览"
            onClick={() => {
              closePreviewZoom()
            }}
          />

          <div className="preview-zoom__dialog" ref={previewZoomDialogRef}>
            <div className="preview-zoom__header">
              <div className="preview-zoom__summary">
                <div className="preview-zoom__summary-top">
                  <span className="preview-zoom__badge">
                    {isScalablePreviewTarget ? '图像查看器' : '媒体播放器'}
                  </span>
                  {isScalablePreviewTarget ? (
                    <span className="preview-zoom__scale">{previewZoomScaleLabel}</span>
                  ) : null}
                </div>
                <div className="preview-zoom__meta">
                  <span className="preview-zoom__meta-item">模式 {inputMode}</span>
                  <span className="preview-zoom__meta-item">Runtime {resultProvider || '-'}</span>
                  <span className="preview-zoom__meta-item">时间 {currentTimeLabel}</span>
                  {isScalablePreviewTarget ? (
                    <span className="preview-zoom__meta-item">检测项 {detectionItems.length}</span>
                  ) : null}
                </div>
              </div>

              <div className="preview-zoom__toolbar">
                <div className="preview-zoom__toolbar-group">
                  {isScalablePreviewTarget ? (
                    <>
                      <button
                        type="button"
                        className="preview-zoom__action"
                        disabled={previewViewerScale <= PREVIEW_VIEWER_MIN_SCALE + 0.01}
                        onClick={() => {
                          setPreviewViewerScaleValue(previewViewerScale - PREVIEW_VIEWER_WHEEL_STEP)
                        }}
                      >
                        缩小
                      </button>
                      <button
                        type="button"
                        className={`preview-zoom__action${
                          Math.abs(previewViewerScale - PREVIEW_VIEWER_DEFAULT_SCALE) < 0.01
                            ? ' preview-zoom__action--active'
                            : ''
                        }`}
                        onClick={() => {
                          setPreviewViewerScaleValue(PREVIEW_VIEWER_DEFAULT_SCALE)
                          setPreviewViewerOffset(PREVIEW_VIEWER_DEFAULT_OFFSET)
                        }}
                      >
                        适合窗口
                      </button>
                      <button
                        type="button"
                        className="preview-zoom__action"
                        disabled={previewViewerScale >= PREVIEW_VIEWER_MAX_SCALE - 0.01}
                        onClick={() => {
                          setPreviewViewerScaleValue(previewViewerScale + PREVIEW_VIEWER_WHEEL_STEP)
                        }}
                      >
                        放大
                      </button>
                      <button
                        type="button"
                        className="preview-zoom__action"
                        onClick={() => {
                          setPreviewViewerScaleValue(PREVIEW_VIEWER_DEFAULT_SCALE)
                          setPreviewViewerOffset(PREVIEW_VIEWER_DEFAULT_OFFSET)
                        }}
                      >
                        重置视图
                      </button>
                    </>
                  ) : null}
                  <button
                    type="button"
                    className="preview-zoom__action"
                    onClick={() => {
                      void togglePreviewViewerFullscreen()
                    }}
                  >
                    {isPreviewViewerFullscreen ? '退出全屏' : '全屏'}
                  </button>
                  <button
                    type="button"
                    className="preview-zoom__action preview-zoom__action--close"
                    onClick={() => {
                      closePreviewZoom()
                    }}
                  >
                    关闭
                  </button>
                </div>
              </div>
            </div>

            <div
              ref={previewZoomViewportRef}
              className={`preview-zoom__viewport${
                isScalablePreviewTarget ? ' preview-zoom__viewport--interactive' : ''
              }`}
              onWheel={isScalablePreviewTarget ? handlePreviewWheel : undefined}
            >
              <div
                ref={previewZoomMediaRef}
                className={`preview-zoom__media-wrapper${
                  isScalablePreviewTarget && previewViewerScale > PREVIEW_VIEWER_DEFAULT_SCALE
                    ? ' preview-zoom__media-wrapper--draggable'
                    : ''
                }`}
                style={isScalablePreviewTarget
                  ? {
                    transform: `translate(${previewViewerOffset.x}px, ${previewViewerOffset.y}px) scale(${previewViewerScale})`,
                  }
                  : undefined}
                onDoubleClick={isScalablePreviewTarget ? handlePreviewMediaDoubleClick : undefined}
                onPointerCancel={isScalablePreviewTarget ? handlePreviewPointerRelease : undefined}
                onPointerDown={isScalablePreviewTarget ? handlePreviewPointerDown : undefined}
                onPointerMove={isScalablePreviewTarget ? handlePreviewPointerMove : undefined}
                onPointerUp={isScalablePreviewTarget ? handlePreviewPointerRelease : undefined}
              >
                {previewZoomTarget === 'image' ? (
                  <img className="preview-zoom__image" src={imagePreviewUrl} alt="放大预览图片" />
                ) : null}

                {previewZoomTarget === 'video' || previewZoomTarget === 'camera' ? (
                  <video
                    ref={previewZoomVideoRef}
                    className="preview-zoom__video"
                    autoPlay={previewZoomTarget === 'camera'}
                    controls={previewZoomTarget === 'video'}
                    muted
                    playsInline
                  />
                ) : null}

                {previewZoomTarget === 'result-canvas' ? (
                  <canvas ref={previewZoomCanvasRef} className="preview-zoom__canvas" />
                ) : null}
              </div>
            </div>

          </div>
        </div>
      ) : null}
    </div>
  )
}

function SvgIcon(props: { markup: string }) {
  return (
    <span
      className="svg-icon"
      aria-hidden="true"
      dangerouslySetInnerHTML={{ __html: props.markup }}
    />
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
