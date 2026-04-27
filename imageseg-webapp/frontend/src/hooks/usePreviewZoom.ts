import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  type WheelEvent as ReactWheelEvent,
} from 'react'
import type {
  DetectionItem,
  InputMode,
  PreviewViewerOffset,
  StreamState,
} from '../app/types'
import {
  PREVIEW_VIEWER_DEFAULT_OFFSET,
  PREVIEW_VIEWER_DEFAULT_SCALE,
  PREVIEW_VIEWER_MAX_SCALE,
  PREVIEW_VIEWER_MIN_SCALE,
  PREVIEW_VIEWER_WHEEL_STEP,
  clampPreviewViewerOffset,
  clampPreviewZoomScale,
  isPreviewScalableTarget,
  syncCanvas,
  resolvePreviewZoomTarget,
} from '../app/workspaceUtils'

type UsePreviewZoomArgs = {
  cameraStreamRef: RefObject<MediaStream | null>
  detectionItems: DetectionItem[]
  hasRenderedResult: boolean
  imagePreviewUrl: string
  inputMode: InputMode
  resultCanvasRef: RefObject<HTMLCanvasElement | null>
  sourceVideoRef: RefObject<HTMLVideoElement | null>
  streamFps: number | null
  streamState: StreamState
  videoPreviewUrl: string
}

export function usePreviewZoom(args: UsePreviewZoomArgs) {
  const {
    cameraStreamRef,
    detectionItems,
    hasRenderedResult,
    imagePreviewUrl,
    inputMode,
    resultCanvasRef,
    sourceVideoRef,
    streamFps,
    streamState,
    videoPreviewUrl,
  } = args

  const [isPreviewZoomOpen, setIsPreviewZoomOpen] = useState(false)
  const [previewViewerScale, setPreviewViewerScale] = useState(PREVIEW_VIEWER_DEFAULT_SCALE)
  const [previewViewerOffset, setPreviewViewerOffset] = useState<PreviewViewerOffset>(
    PREVIEW_VIEWER_DEFAULT_OFFSET,
  )
  const [isPreviewViewerFullscreen, setIsPreviewViewerFullscreen] = useState(false)

  const previewZoomDialogRef = useRef<HTMLDivElement | null>(null)
  const previewZoomViewportRef = useRef<HTMLDivElement | null>(null)
  const previewZoomMediaRef = useRef<HTMLDivElement | null>(null)
  const previewZoomVideoRef = useRef<HTMLVideoElement | null>(null)
  const previewZoomCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const previewPointerDragRef = useRef<{
    originX: number
    originY: number
    pointerId: number
  } | null>(null)

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
  const previewZoomScaleLabel = `${Math.round(previewViewerScale * 100)}%`

  const resetPreviewViewer = useCallback(() => {
    setIsPreviewZoomOpen(false)
    setPreviewViewerScale(PREVIEW_VIEWER_DEFAULT_SCALE)
    setPreviewViewerOffset(PREVIEW_VIEWER_DEFAULT_OFFSET)
    setIsPreviewViewerFullscreen(false)
    previewPointerDragRef.current = null
  }, [])

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
  }, [detectionItems, hasRenderedResult, isPreviewZoomOpen, previewZoomTarget, resultCanvasRef, streamFps])

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
        if (!sourceVideoRef.current.paused) {
          void modalVideo.play().catch(() => {})
        }
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
  }, [cameraStreamRef, isPreviewZoomOpen, previewZoomTarget, sourceVideoRef, videoPreviewUrl])

  function openPreviewZoom() {
    if (!availablePreviewZoomTarget) {
      return
    }

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

  return {
    canOpenPreviewZoom,
    closePreviewZoom,
    handlePreviewMediaDoubleClick,
    handlePreviewPointerDown,
    handlePreviewPointerMove,
    handlePreviewPointerRelease,
    handlePreviewWheel,
    isPreviewViewerFullscreen,
    isPreviewZoomOpen,
    isScalablePreviewTarget,
    openPreviewZoom,
    previewOpenLabel,
    previewViewerOffset,
    previewViewerScale,
    previewZoomCanvasRef,
    previewZoomDialogRef,
    previewZoomMediaRef,
    previewZoomScaleLabel,
    previewZoomTarget,
    previewZoomVideoRef,
    previewZoomViewportRef,
    resetPreviewViewer,
    setIsPreviewZoomOpen,
    setPreviewViewerDefault() {
      setPreviewViewerScaleValue(PREVIEW_VIEWER_DEFAULT_SCALE)
      setPreviewViewerOffset(PREVIEW_VIEWER_DEFAULT_OFFSET)
    },
    setPreviewViewerScaleStepDown() {
      setPreviewViewerScaleValue(previewViewerScale - PREVIEW_VIEWER_WHEEL_STEP)
    },
    setPreviewViewerScaleStepUp() {
      setPreviewViewerScaleValue(previewViewerScale + PREVIEW_VIEWER_WHEEL_STEP)
    },
    scaleDecreaseDisabled: previewViewerScale <= PREVIEW_VIEWER_MIN_SCALE + 0.01,
    scaleIncreaseDisabled: previewViewerScale >= PREVIEW_VIEWER_MAX_SCALE - 0.01,
    togglePreviewViewerFullscreen,
  }
}
