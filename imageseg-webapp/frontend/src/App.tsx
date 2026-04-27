import { useEffect, useRef, useState } from 'react'
import { AppGlobalStyle } from './app/AppGlobalStyle'
import { Footer } from './components/Footer'
import { HelpModal } from './components/HelpModal'
import { HomePage } from './components/HomePage'
import { PreviewZoomModal } from './components/PreviewZoomModal'
import { TopBar } from './components/TopBar'
import { WorkspacePage } from './components/WorkspacePage'
import { WorkspaceSidebar } from './components/WorkspaceSidebar'
import { deriveModelImportControls } from './lib/modelImportState'
import { getWebGpuSupportState } from './lib/onnxRuntime'
import {
  buildVideoExportFileName,
  formatDims,
} from './app/workspaceUtils'
import { useDetectionWorkspace } from './hooks/useDetectionWorkspace'
import { useModelImport } from './hooks/useModelImport'
import { usePreviewZoom } from './hooks/usePreviewZoom'
import { useThemeMode } from './hooks/useThemeMode'
import { useWorkspaceNavigation } from './hooks/useWorkspaceNavigation'

function App() {
  const [isHelpOpen, setIsHelpOpen] = useState(false)

  const sourceVideoRef = useRef<HTMLVideoElement | null>(null)
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null)
  const resultCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const cameraStreamRef = useRef<MediaStream | null>(null)
  const importedModelRef = useRef<import('./lib/onnxRuntime').ImportedModel | null>(null)
  const resetPreviewViewerRef = useRef<(() => void) | null>(null)

  const { resolvedTheme, setThemeMode, themeMode } = useThemeMode()
  const {
    activePageId,
    activeSectionId,
    handleNavigate,
    handlePageNavigate,
    isSidebarExpanded,
    setIsSidebarExpanded,
  } = useWorkspaceNavigation()

  const supportsDirectoryPicker =
    typeof window !== 'undefined' &&
    typeof window.showDirectoryPicker === 'function'
  const supportsCamera =
    typeof navigator !== 'undefined' &&
    Boolean(navigator.mediaDevices?.getUserMedia)
  const webGpuSupportState = getWebGpuSupportState()
  const isWebGpuSupported = webGpuSupportState.supported

  const detection = useDetectionWorkspace({
    cameraStreamRef,
    cameraVideoRef,
    importedModelRef,
    isWebGpuSupported,
    resetPreviewViewerRef,
    resultCanvasRef,
    sourceVideoRef,
    supportsCamera,
    webGpuSupportState,
  })

  const modelImport = useModelImport({
    canChangeModel: !detection.imageDetectBusy && detection.streamState === 'idle',
    clearRecognitionOutputs: detection.clearRecognitionOutputs,
    resetDetectionThreshold: detection.resetDetectionThreshold,
    resetMaxDetections: detection.resetMaxDetections,
    setStatusMessage: detection.setStatusMessage,
    stopActiveStream: detection.stopActiveStream,
    supportsDirectoryPicker,
    webGpuSupportState,
  })

  const preview = usePreviewZoom({
    cameraStreamRef,
    detectionItems: detection.detectionItems,
    hasRenderedResult: detection.hasRenderedResult,
    imagePreviewUrl: detection.imagePreviewUrl,
    inputMode: detection.inputMode,
    resultCanvasRef,
    sourceVideoRef,
    streamFps: detection.streamFps,
    streamState: detection.streamState,
    videoPreviewUrl: detection.videoPreviewUrl,
  })

  useEffect(() => {
    importedModelRef.current = modelImport.importedModel
  }, [modelImport.importedModel])

  useEffect(() => {
    resetPreviewViewerRef.current = preview.resetPreviewViewer
  }, [preview.resetPreviewViewer])

  const importedModel = modelImport.importedModel
  const importControls = deriveModelImportControls(
    modelImport.onnxModelDraft?.family ?? null,
    supportsDirectoryPicker,
  )
  const displayedContract = importedModel?.contract ?? modelImport.onnxModelDraft?.draftContract ?? null
  const displayedProvider = importedModel?.providerName ?? '-'
  const displayedSidecars = [
    importedModel?.sidecars.configFileName ?? modelImport.configFile?.name,
    importedModel?.sidecars.preprocessorConfigFileName ?? modelImport.preprocessorConfigFile?.name,
  ].filter(Boolean).join(', ') || '-'
  const displayedRuntimeMessage =
    detection.runtimeMessage ||
    (importedModel && !isWebGpuSupported ? (webGpuSupportState.message ?? '') : '')
  const displayedPendingThreshold = detection.pendingDetectionThreshold.toFixed(2)
  const displayedAppliedMaxDetections = String(detection.appliedMaxDetections)
  const displayedFps = detection.streamFps === null ? '-' : detection.streamFps.toFixed(1)
  const currentTimeLabel = detection.inputMode === 'camera' && detection.streamState === 'running'
    ? 'Live'
    : detection.currentSourceTime === null
      ? '-'
      : `${detection.currentSourceTime.toFixed(2)} s`
  const isAnyOverlayOpen = preview.isPreviewZoomOpen || isHelpOpen

  const canChangeModel =
    !modelImport.modelBusy &&
    !detection.imageDetectBusy &&
    detection.streamState === 'idle'
  const canRunImage =
    detection.inputMode === 'image' &&
    Boolean(detection.imageFile && importedModel && isWebGpuSupported && !modelImport.modelBusy && !detection.imageDetectBusy && detection.streamState === 'idle')
  const canExportImage =
    detection.inputMode === 'image' &&
    detection.hasRenderedResult &&
    !detection.imageDetectBusy &&
    detection.streamState === 'idle'
  const canStartVideo =
    detection.inputMode === 'video' &&
    Boolean(detection.videoFile && importedModel && isWebGpuSupported && detection.streamState === 'idle')
  const canStopVideo =
    detection.inputMode === 'video' &&
    detection.streamState === 'running'
  const supportsVideoExport = Boolean(detection.preferredVideoMimeType)
  const canExportVideo =
    detection.inputMode === 'video' &&
    Boolean(detection.videoFile && importedModel && isWebGpuSupported && detection.streamState === 'idle' && supportsVideoExport)
  const canStartCamera =
    detection.inputMode === 'camera' &&
    Boolean(importedModel && isWebGpuSupported && detection.streamState === 'idle' && !detection.cameraBusy && supportsCamera)
  const canStopCamera =
    detection.inputMode === 'camera' &&
    detection.streamState === 'running'
  const canAdjustInferenceSettings = Boolean(
    importedModel &&
    !modelImport.modelBusy &&
    detection.streamState !== 'exporting',
  )

  useEffect(() => {
    if (!preview.isPreviewZoomOpen && !isHelpOpen) {
      return
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') {
        return
      }

      if (preview.isPreviewZoomOpen) {
        preview.closePreviewZoom()
        return
      }

      if (isHelpOpen) {
        setIsHelpOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [isHelpOpen, preview])

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

  function handleWorkspaceNavigate(sectionId: Parameters<typeof handleNavigate>[0]) {
    handleNavigate(sectionId)
  }

  function handleTopPageNavigate(pageId: Parameters<typeof handlePageNavigate>[0]) {
    handlePageNavigate(pageId)
  }

  function openHelpDialog() {
    setIsHelpOpen(true)
  }

  function openPreviewZoomDialog() {
    setIsHelpOpen(false)
    preview.openPreviewZoom()
  }

  return (
    <>
      <AppGlobalStyle />
      <div className="app-shell">
        {activePageId === 'workspace' ? (
          <WorkspaceSidebar
            activeSectionId={activeSectionId}
            isExpanded={isSidebarExpanded}
            onNavigate={handleWorkspaceNavigate}
            onToggle={() => {
              setIsSidebarExpanded((currentValue) => !currentValue)
            }}
          />
        ) : null}

        <div className="page-layout">
          <TopBar
            activePageId={activePageId}
            onHelpOpen={openHelpDialog}
            onPageNavigate={handleTopPageNavigate}
            onThemeModeSelect={setThemeMode}
            resolvedTheme={resolvedTheme}
            themeMode={themeMode}
          />

          <div className="surface">
            <main className="shell">
              {activePageId === 'home' ? (
                <HomePage
                  onEnterWorkspace={() => {
                    handleTopPageNavigate('workspace')
                  }}
                  onHelpOpen={openHelpDialog}
                />
              ) : null}

              {activePageId === 'workspace' ? (
                <WorkspacePage
                  appliedDetectionThreshold={detection.appliedDetectionThreshold}
                  cameraBusy={detection.cameraBusy}
                  cameraDevices={detection.cameraDevices}
                  cameraVideoRef={cameraVideoRef}
                  canAdjustInferenceSettings={canAdjustInferenceSettings}
                  canChangeModel={canChangeModel}
                  canExportImage={canExportImage}
                  canExportVideo={canExportVideo}
                  canOpenPreviewZoom={preview.canOpenPreviewZoom}
                  canRunImage={canRunImage}
                  canStartCamera={canStartCamera}
                  canStartVideo={canStartVideo}
                  canStopCamera={canStopCamera}
                  canStopVideo={canStopVideo}
                  configInputKey={modelImport.configInputKey}
                  currentTimeLabel={currentTimeLabel}
                  detectionItems={detection.detectionItems}
                  displayedAppliedMaxDetections={displayedAppliedMaxDetections}
                  displayedContract={displayedContract}
                  displayedFps={displayedFps}
                  displayedPendingThreshold={displayedPendingThreshold}
                  displayedProvider={displayedProvider}
                  displayedRuntimeMessage={displayedRuntimeMessage}
                  displayedSidecars={displayedSidecars}
                  discoverBusy={modelImport.discoverBusy}
                  formatDims={formatDims}
                  hasRenderedResult={detection.hasRenderedResult}
                  imageDetectBusy={detection.imageDetectBusy}
                  imageFileName={detection.imageFile?.name ?? ''}
                  imagePreviewUrl={detection.imagePreviewUrl}
                  importControls={importControls}
                  importedModelExists={Boolean(importedModel)}
                  inputMode={detection.inputMode}
                  modelBusy={modelImport.modelBusy}
                  onAutoDiscoverSidecars={modelImport.handleAutoDiscoverSidecars}
                  onCommitDetectionThreshold={detection.commitDetectionThreshold}
                  onCommitMaxDetections={detection.commitMaxDetections}
                  onConfigSelection={modelImport.handleConfigSelection}
                  onExportImage={detection.handleExportImage}
                  onExportVideo={detection.handleExportVideo}
                  onImageSelection={detection.updateImageFile}
                  onMaxDetectionsInputChange={detection.setPendingMaxDetectionsInput}
                  onModeChange={detection.handleInputModeChange}
                  onOnnxSelection={modelImport.handleOnnxSelection}
                  onOpenPreviewZoom={openPreviewZoomDialog}
                  onPreprocessorSelection={modelImport.handlePreprocessorSelection}
                  onRunImageDetection={detection.handleRunImageDetection}
                  onSelectedCameraChange={detection.setSelectedCameraId}
                  onStartCameraDetection={detection.handleStartCameraDetection}
                  onStartVideoDetection={detection.handleStartVideoDetection}
                  onStopCamera={() => {
                    detection.stopActiveStream('已停止摄像头实时识别。')
                  }}
                  onStopVideo={() => {
                    detection.stopActiveStream('已停止视频实时识别。')
                  }}
                  onThresholdChange={(value) => {
                    detection.setPendingDetectionThreshold(value)
                  }}
                  onVideoSelection={detection.updateVideoFile}
                  onnxInputKey={modelImport.onnxInputKey}
                  onnxModelDraft={modelImport.onnxModelDraft}
                  pendingDetectionThreshold={detection.pendingDetectionThreshold}
                  pendingMaxDetectionsInput={detection.pendingMaxDetectionsInput}
                  preprocessorInputKey={modelImport.preprocessorInputKey}
                  previewOpenLabel={preview.previewOpenLabel}
                  resultCanvasRef={resultCanvasRef}
                  resultProvider={detection.resultProvider}
                  selectedCameraId={detection.selectedCameraId}
                  sourceVideoRef={sourceVideoRef}
                  statusMessage={detection.statusMessage}
                  streamState={detection.streamState}
                  supportsCamera={supportsCamera}
                  supportsDirectoryPicker={supportsDirectoryPicker}
                  supportsVideoExport={supportsVideoExport}
                  videoDownloadFileName={buildVideoExportFileName(detection.videoFile?.name ?? 'result')}
                  videoDownloadUrl={detection.videoDownloadUrl}
                  videoFileName={detection.videoFile?.name ?? ''}
                  videoPreviewUrl={detection.videoPreviewUrl}
                />
              ) : null}
            </main>
          </div>

          <Footer />
        </div>

        {isHelpOpen ? <HelpModal onClose={() => { setIsHelpOpen(false) }} /> : null}

        {preview.isPreviewZoomOpen && preview.previewZoomTarget ? (
          <PreviewZoomModal
            closePreviewZoom={preview.closePreviewZoom}
            currentTimeLabel={currentTimeLabel}
            detectionItemCount={detection.detectionItems.length}
            handlePreviewMediaDoubleClick={preview.handlePreviewMediaDoubleClick}
            handlePreviewPointerDown={preview.handlePreviewPointerDown}
            handlePreviewPointerMove={preview.handlePreviewPointerMove}
            handlePreviewPointerRelease={preview.handlePreviewPointerRelease}
            handlePreviewWheel={preview.handlePreviewWheel}
            imagePreviewUrl={detection.imagePreviewUrl}
            inputMode={detection.inputMode}
            isPreviewViewerFullscreen={preview.isPreviewViewerFullscreen}
            isScalablePreviewTarget={preview.isScalablePreviewTarget}
            previewViewerOffset={preview.previewViewerOffset}
            previewViewerScale={preview.previewViewerScale}
            previewZoomCanvasRef={preview.previewZoomCanvasRef}
            previewZoomDialogRef={preview.previewZoomDialogRef}
            previewZoomMediaRef={preview.previewZoomMediaRef}
            previewZoomScaleLabel={preview.previewZoomScaleLabel}
            previewZoomTarget={preview.previewZoomTarget}
            previewZoomVideoRef={preview.previewZoomVideoRef}
            previewZoomViewportRef={preview.previewZoomViewportRef}
            resultProvider={detection.resultProvider}
            scaleDecreaseDisabled={preview.scaleDecreaseDisabled}
            scaleIncreaseDisabled={preview.scaleIncreaseDisabled}
            setPreviewViewerDefault={preview.setPreviewViewerDefault}
            setPreviewViewerScaleStepDown={preview.setPreviewViewerScaleStepDown}
            setPreviewViewerScaleStepUp={preview.setPreviewViewerScaleStepUp}
            togglePreviewViewerFullscreen={preview.togglePreviewViewerFullscreen}
          />
        ) : null}
      </div>
    </>
  )
}

export default App
