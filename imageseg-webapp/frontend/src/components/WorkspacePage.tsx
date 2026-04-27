import type { RefObject } from 'react'
import styled from 'styled-components'
import {
  CONFIG_FILE_NAME,
  PREPROCESSOR_CONFIG_FILE_NAME,
  type InspectedOnnxModel,
} from '../lib/modelPackage'
import type { ImportedModel } from '../lib/onnxRuntime'
import type { CameraDeviceOption, DetectionItem, InputMode, StreamState } from '../app/types'
import { EmptyState } from './EmptyState'
import { Metric } from './Metric'
import { SignatureTable } from './SignatureTable'

const Root = styled.div`
  .operation-grid {
    display: grid;
    grid-template-columns: minmax(280px, 340px) minmax(420px, 1fr) minmax(280px, 340px);
    gap: 20px;
    align-items: stretch;
    margin-top: 20px;
  }

  .operation-grid__controls,
  .operation-grid__settings,
  .operation-grid__preview,
  .results-row__panel {
    min-width: 0;
  }

  .operation-grid__controls {
    display: grid;
  }

  .operation-grid__controls .panel,
  .operation-grid__preview,
  .operation-grid__settings {
    height: 100%;
  }

  .operation-grid__preview .preview-stage__media {
    --preview-max-height: min(72vh, 760px);
    min-height: clamp(360px, 42vw, 620px);
  }

  .results-row,
  .diagnostics-row {
    margin-top: 20px;
  }

  .results-row__panel .metric-grid {
    grid-template-columns: repeat(5, minmax(0, 1fr));
  }

  .diagnostics-panel .contract {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
  }

  .diagnostics-panel .metric-grid,
  .diagnostics-panel .warnings {
    grid-column: 1 / -1;
  }

  .diagnostics-panel .metric-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .panel {
    min-width: 0;
    border-radius: 24px;
    padding: 24px;
    background: var(--paper-strong);
    border: 1px solid var(--line);
    box-shadow: var(--card-shadow);
    backdrop-filter: blur(12px);
  }

  .panel__header h2 {
    margin: 0 0 6px;
    font-size: 1.3rem;
  }

  .field {
    display: grid;
    gap: 10px;
    margin-top: 18px;
  }

  .field__label,
  .field__hint,
  .warnings__title,
  .status-box__title,
  .preview-card__title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
  }

  .field input {
    width: 100%;
    padding: 13px 14px;
    border-radius: 16px;
    border: 1px dashed color-mix(in srgb, var(--ink) 24%, transparent);
    background: var(--control-bg);
    color: var(--ink);
  }

  .field__select {
    width: 100%;
    padding: 13px 14px;
    border-radius: 16px;
    border: 1px solid color-mix(in srgb, var(--ink) 16%, transparent);
    background: var(--control-bg);
    color: var(--ink);
  }

  .threshold-control,
  .count-control {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 12px;
    align-items: center;
  }

  .threshold-control__slider,
  .count-control__input {
    width: 100%;
  }

  .threshold-control__slider {
    accent-color: var(--accent);
  }

  .threshold-control__value,
  .count-control__value {
    min-width: 3.2rem;
    padding: 8px 10px;
    border-radius: 12px;
    background: var(--control-bg);
    border: 1px solid color-mix(in srgb, var(--ink) 12%, transparent);
    font-family: var(--mono);
    text-align: center;
    color: var(--ink);
  }

  .field__hint {
    text-transform: none;
    letter-spacing: 0;
    line-height: 1.55;
  }

  .actions {
    margin-top: 20px;
  }

  .actions--stacked {
    display: grid;
    gap: 10px;
  }

  .action {
    border: none;
    border-radius: 999px;
    padding: 14px 20px;
    cursor: pointer;
    transition: transform 140ms ease, opacity 140ms ease;
  }

  .action:hover:not(:disabled) {
    transform: translateY(-1px);
  }

  .action:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .action--primary {
    width: 100%;
    background: linear-gradient(135deg, var(--accent), #db8d34);
    color: var(--primary-action-text);
    font-weight: 700;
  }

  .action--secondary {
    width: 100%;
    border: 1px solid var(--secondary-action-border);
    background: var(--secondary-action-bg);
    color: var(--sea);
    font-weight: 700;
  }

  .mode-switch {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 18px;
  }

  .mode-switch__button {
    border: 1px solid var(--mode-switch-border);
    border-radius: 16px;
    padding: 12px 14px;
    background: var(--mode-switch-bg);
    color: var(--muted);
    cursor: pointer;
    transition: border-color 140ms ease, color 140ms ease, transform 140ms ease;
  }

  .mode-switch__button:hover:not(:disabled) {
    transform: translateY(-1px);
    border-color: rgba(199, 88, 43, 0.4);
    color: var(--ink);
  }

  .mode-switch__button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .mode-switch__button--active {
    border-color: var(--mode-switch-active-border);
    background: var(--mode-switch-active-bg);
    color: var(--accent);
    font-weight: 700;
  }

  .status-box,
  .warnings,
  .preview-card {
    margin-top: 18px;
    padding: 16px;
    border-radius: 18px;
    border: 1px solid var(--line);
    background: var(--control-bg-soft);
  }

  .status-box p,
  .warnings p {
    margin: 8px 0 0;
    color: var(--muted);
    line-height: 1.65;
  }

  .contract {
    margin-top: 18px;
  }

  .metric-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
  }

  .metric-grid--compact {
    margin-top: 18px;
  }

  .preview-stage {
    margin-top: 18px;
  }

  .preview-stage__media {
    --preview-max-height: min(60vh, 680px);
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 240px;
    max-height: var(--preview-max-height);
    margin-top: 12px;
    padding: 12px;
    border-radius: 14px;
    overflow: hidden;
    background: var(--preview-stage-bg);
  }

  .preview-stage__zoom-button {
    position: absolute;
    top: 22px;
    right: 22px;
    z-index: 2;
    display: inline-grid;
    place-items: center;
    width: 42px;
    height: 42px;
    padding: 0;
    border: 1px solid var(--floating-button-border);
    border-radius: 50%;
    background: var(--floating-button-bg);
    color: var(--floating-button-text);
    backdrop-filter: blur(12px);
    cursor: pointer;
    opacity: 0;
    pointer-events: none;
    transition: opacity 160ms ease, background 160ms ease, transform 160ms ease;
  }

  .preview-stage__media:hover .preview-stage__zoom-button,
  .preview-stage__media:focus-within .preview-stage__zoom-button {
    opacity: 1;
    pointer-events: auto;
  }

  .preview-stage__zoom-button:hover {
    transform: translateY(-1px);
    background: var(--floating-button-bg-hover);
  }

  .preview-stage__zoom-icon {
    font-size: 1.6rem;
    line-height: 1;
    transform: translateY(-1px);
  }

  .preview-stage__image,
  .preview-stage__video,
  .preview-stage__canvas {
    display: block;
    max-width: 100%;
    max-height: calc(var(--preview-max-height) - 24px);
    width: auto;
    height: auto;
    border-radius: 14px;
    object-fit: contain;
  }

  .preview-stage__visual--hidden {
    position: absolute;
    inset: 12px;
    opacity: 0;
    pointer-events: none;
  }

  .preview-stage__empty-state > div {
    width: 100%;
    margin-top: 0;
  }

  .result-list {
    display: grid;
    gap: 12px;
    margin-top: 18px;
  }

  .result-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 14px;
    border-radius: 14px;
    background: var(--soft-fill);
  }

  .download-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 48px;
    padding: 12px 18px;
    border-radius: 999px;
    background: var(--download-link-bg);
    border: 1px solid var(--download-link-border);
    color: var(--accent);
    font-weight: 700;
    text-decoration: none;
  }

  .download-link:hover {
    background: var(--download-link-hover-bg);
  }

  .result-row__label {
    font-weight: 700;
  }

  .result-row__box {
    margin-top: 6px;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 0.84rem;
  }

  .result-row__score {
    min-width: 88px;
    text-align: right;
    font-weight: 700;
    color: var(--sea);
  }

  @media (min-width: 1181px) {
    .operation-grid {
      gap: 18px;
    }

    .operation-grid .panel,
    .results-row .panel,
    .diagnostics-row {
      padding: 18px;
      border-radius: 20px;
    }

    .operation-grid .panel__header h2,
    .results-row .panel__header h2,
    .diagnostics-row .panel__header h2 {
      font-size: 1.12rem;
    }

    .operation-grid .field__hint,
    .operation-grid .status-box p,
    .diagnostics-row .warnings p {
      font-size: 0.86rem;
      line-height: 1.5;
    }

    .operation-grid .field {
      gap: 8px;
      margin-top: 14px;
    }

    .operation-grid .field__label,
    .operation-grid .field__hint,
    .operation-grid .status-box__title,
    .operation-grid .preview-card__title,
    .diagnostics-row .warnings__title {
      font-size: 0.72rem;
      letter-spacing: 0.1em;
    }

    .operation-grid .field input,
    .operation-grid .field__select {
      padding: 10px 12px;
      border-radius: 13px;
      font-size: 0.9rem;
    }

    .operation-grid .mode-switch {
      gap: 8px;
      margin-top: 14px;
    }

    .operation-grid .mode-switch__button {
      padding: 10px 12px;
      border-radius: 13px;
      font-size: 0.9rem;
    }

    .operation-grid .threshold-control,
    .operation-grid .count-control {
      gap: 10px;
    }

    .operation-grid .threshold-control__value,
    .operation-grid .count-control__value {
      min-width: 2.9rem;
      padding: 7px 9px;
      border-radius: 10px;
      font-size: 0.84rem;
    }

    .operation-grid .actions {
      margin-top: 16px;
    }

    .operation-grid .actions--stacked {
      gap: 8px;
    }

    .operation-grid .action,
    .operation-grid .download-link {
      min-height: 40px;
      padding: 10px 16px;
      font-size: 0.9rem;
    }

    .operation-grid .status-box,
    .operation-grid .preview-card,
    .results-row .empty-state > div {
      margin-top: 14px;
      padding: 13px;
      border-radius: 15px;
    }

    .diagnostics-row .metric-grid,
    .results-row .metric-grid {
      gap: 10px;
    }

    .results-row .result-list {
      gap: 10px;
    }

    .results-row .result-row {
      gap: 12px;
      padding: 10px 12px;
      border-radius: 12px;
    }

    .results-row .result-row__box {
      font-size: 0.78rem;
    }

    .results-row .result-row__score {
      min-width: 72px;
    }
  }

  @media (max-width: 1180px) {
    .operation-grid {
      grid-template-columns: 1fr;
    }

    .operation-grid__preview .preview-stage__media {
      --preview-max-height: min(60vh, 680px);
      min-height: min(360px, 52vh);
    }

    .results-row__panel .metric-grid,
    .diagnostics-panel .metric-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 1040px) {
    .panel {
      padding: 20px;
    }

    .preview-stage__media {
      --preview-max-height: min(50vh, 520px);
    }

    .operation-grid__preview .preview-stage__media {
      min-height: min(320px, 50vh);
    }
  }

  @media (max-width: 720px) {
    .preview-stage__zoom-button {
      top: 16px;
      right: 16px;
      opacity: 1;
      pointer-events: auto;
    }

    .mode-switch,
    .metric-grid,
    .results-row__panel .metric-grid,
    .diagnostics-panel .contract,
    .diagnostics-panel .metric-grid {
      grid-template-columns: 1fr;
    }

    .operation-grid__preview .preview-stage__media {
      min-height: 260px;
    }
  }

  @media (max-width: 560px) {
    .result-row {
      align-items: flex-start;
      flex-direction: column;
    }

    .result-row__score {
      min-width: 0;
      text-align: left;
    }
  }

  @media (hover: none) {
    .preview-stage__zoom-button {
      opacity: 1;
      pointer-events: auto;
    }
  }
`

type ImportControls = {
  autoDiscoverEnabled: boolean
  configEnabled: boolean
  configRequired: boolean
  preprocessorEnabled: boolean
}

type WorkspacePageProps = {
  appliedDetectionThreshold: number
  cameraBusy: boolean
  cameraDevices: CameraDeviceOption[]
  canAdjustInferenceSettings: boolean
  canChangeModel: boolean
  canExportImage: boolean
  canExportVideo: boolean
  canOpenPreviewZoom: boolean
  canRunImage: boolean
  canStartCamera: boolean
  canStartVideo: boolean
  canStopCamera: boolean
  canStopVideo: boolean
  configInputKey: number
  currentTimeLabel: string
  detectionItems: DetectionItem[]
  displayedAppliedMaxDetections: string
  displayedContract: ImportedModel['contract'] | InspectedOnnxModel['draftContract'] | null
  displayedFps: string
  displayedPendingThreshold: string
  displayedProvider: string
  displayedRuntimeMessage: string
  displayedSidecars: string
  discoverBusy: boolean
  formatDims: (dims: Array<number | string | null>) => string
  hasRenderedResult: boolean
  imageDetectBusy: boolean
  imageFileName: string
  imagePreviewUrl: string
  importControls: ImportControls
  importedModelExists: boolean
  inputMode: InputMode
  modelBusy: boolean
  onAutoDiscoverSidecars: () => Promise<void>
  onCommitDetectionThreshold: (value: number) => void
  onCommitMaxDetections: (value: string) => void
  onConfigSelection: (file: File | null) => Promise<void>
  onExportImage: () => Promise<void>
  onExportVideo: () => Promise<void>
  onImageSelection: (file: File | null) => void
  onMaxDetectionsInputChange: (value: string) => void
  onModeChange: (mode: InputMode) => void
  onOnnxSelection: (file: File | null) => Promise<void>
  onOpenPreviewZoom: () => void
  onPreprocessorSelection: (file: File | null) => Promise<void>
  onRunImageDetection: () => Promise<void>
  onSelectedCameraChange: (value: string) => void
  onStartCameraDetection: () => Promise<void>
  onStartVideoDetection: () => Promise<void>
  onStopCamera: () => void
  onStopVideo: () => void
  onThresholdChange: (value: number) => void
  onVideoSelection: (file: File | null) => void
  onnxInputKey: number
  onnxModelDraft: InspectedOnnxModel | null
  pendingDetectionThreshold: number
  pendingMaxDetectionsInput: string
  preprocessorInputKey: number
  previewOpenLabel: string
  resultCanvasRef: RefObject<HTMLCanvasElement | null>
  resultProvider: string
  selectedCameraId: string
  sourceVideoRef: RefObject<HTMLVideoElement | null>
  statusMessage: string
  streamState: StreamState
  supportsCamera: boolean
  supportsDirectoryPicker: boolean
  supportsVideoExport: boolean
  videoDownloadFileName: string
  videoDownloadUrl: string
  videoFileName: string
  videoPreviewUrl: string
  cameraVideoRef: RefObject<HTMLVideoElement | null>
}

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

export function WorkspacePage(props: WorkspacePageProps) {
  const {
    appliedDetectionThreshold,
    cameraBusy,
    cameraDevices,
    cameraVideoRef,
    canAdjustInferenceSettings,
    canChangeModel,
    canExportImage,
    canExportVideo,
    canOpenPreviewZoom,
    canRunImage,
    canStartCamera,
    canStartVideo,
    canStopCamera,
    canStopVideo,
    configInputKey,
    currentTimeLabel,
    detectionItems,
    displayedAppliedMaxDetections,
    displayedContract,
    displayedFps,
    displayedPendingThreshold,
    displayedProvider,
    displayedRuntimeMessage,
    displayedSidecars,
    discoverBusy,
    formatDims,
    hasRenderedResult,
    imageDetectBusy,
    imageFileName,
    imagePreviewUrl,
    importControls,
    importedModelExists,
    inputMode,
    modelBusy,
    onAutoDiscoverSidecars,
    onCommitDetectionThreshold,
    onCommitMaxDetections,
    onConfigSelection,
    onExportImage,
    onExportVideo,
    onImageSelection,
    onMaxDetectionsInputChange,
    onModeChange,
    onOnnxSelection,
    onOpenPreviewZoom,
    onPreprocessorSelection,
    onRunImageDetection,
    onSelectedCameraChange,
    onStartCameraDetection,
    onStartVideoDetection,
    onStopCamera,
    onStopVideo,
    onThresholdChange,
    onVideoSelection,
    onnxInputKey,
    onnxModelDraft,
    pendingDetectionThreshold,
    pendingMaxDetectionsInput,
    preprocessorInputKey,
    previewOpenLabel,
    resultCanvasRef,
    resultProvider,
    selectedCameraId,
    sourceVideoRef,
    statusMessage,
    streamState,
    supportsCamera,
    supportsDirectoryPicker,
    supportsVideoExport,
    videoDownloadFileName,
    videoDownloadUrl,
    videoFileName,
    videoPreviewUrl,
  } = props

  return (
    <Root>
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
                    onModeChange(mode)
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
                    onImageSelection(event.target.files?.[0] ?? null)
                  }}
                />
                <span className="field__hint">{imageFileName || '未选择图片。'}</span>
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
                    onVideoSelection(event.target.files?.[0] ?? null)
                  }}
                />
                <span className="field__hint">{videoFileName || '未选择视频。'}</span>
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
                    onSelectedCameraChange(event.target.value)
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
                  void onOnnxSelection(event.target.files?.[0] ?? null)
                }}
              />
            </label>

            <label className="field">
              <span className="field__label">
                {CONFIG_FILE_NAME} {importControls.configRequired ? '(必选)' : '(未启用)'}
              </span>
              <input
                key={configInputKey}
                type="file"
                accept=".json,application/json"
                disabled={!importControls.configEnabled || !canChangeModel}
                onChange={(event) => {
                  void onConfigSelection(event.target.files?.[0] ?? null)
                }}
              />
            </label>

            <label className="field">
              <span className="field__label">
                {PREPROCESSOR_CONFIG_FILE_NAME} {importControls.preprocessorEnabled ? '(可选)' : '(未启用)'}
              </span>
              <input
                key={preprocessorInputKey}
                type="file"
                accept=".json,application/json"
                disabled={!importControls.preprocessorEnabled || !canChangeModel}
                onChange={(event) => {
                  void onPreprocessorSelection(event.target.files?.[0] ?? null)
                }}
              />
            </label>

            {onnxModelDraft?.family === 'hf-detr-like' ? (
              <div className="actions actions--stacked">
                <button
                  type="button"
                  className="action action--secondary"
                  disabled={!importControls.autoDiscoverEnabled || !canChangeModel || discoverBusy}
                  onClick={() => void onAutoDiscoverSidecars()}
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
                  onClick={onOpenPreviewZoom}
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
                  <div className="preview-stage__empty-state">
                    <EmptyState text="未选择图片。" />
                  </div>
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
                  <div className="preview-stage__empty-state">
                    <EmptyState text="未选择视频。" />
                  </div>
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
                  {streamState !== 'running' ? (
                    <div className="preview-stage__empty-state">
                      <EmptyState text="尚未启动摄像头。" />
                    </div>
                  ) : null}
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
                  onThresholdChange(Number(event.target.value))
                }}
                onPointerUp={(event) => {
                  onCommitDetectionThreshold(Number(event.currentTarget.value))
                }}
                onKeyUp={(event) => {
                  if (THRESHOLD_COMMIT_KEYS.has(event.key)) {
                    onCommitDetectionThreshold(Number(event.currentTarget.value))
                  }
                }}
                onBlur={(event) => {
                  onCommitDetectionThreshold(Number(event.currentTarget.value))
                }}
              />
              <span className="threshold-control__value">{displayedPendingThreshold}</span>
            </div>
            <span className="field__hint">
              {importedModelExists
                ? `当前值 ${displayedPendingThreshold}，已生效 ${appliedDetectionThreshold.toFixed(2)}。拖动过程中不会立即重跑推理，松开后新阈值才会生效。`
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
                  onMaxDetectionsInputChange(event.target.value)
                }}
                onBlur={(event) => {
                  onCommitMaxDetections(event.currentTarget.value)
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
                  onClick={() => void onRunImageDetection()}
                >
                  {imageDetectBusy ? '识别中...' : '执行图片识别'}
                </button>
                <button
                  type="button"
                  className="action action--secondary"
                  disabled={!canExportImage}
                  onClick={() => void onExportImage()}
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
                  onClick={() => void onStartVideoDetection()}
                >
                  开始实时识别
                </button>
                <button
                  type="button"
                  className="action action--secondary"
                  disabled={!canStopVideo}
                  onClick={onStopVideo}
                >
                  停止实时识别
                </button>
                <button
                  type="button"
                  className="action action--secondary"
                  disabled={!canExportVideo}
                  onClick={() => void onExportVideo()}
                >
                  {streamState === 'exporting' ? '导出中...' : '导出结果视频'}
                </button>
                {videoDownloadUrl ? (
                  <a
                    className="download-link"
                    href={videoDownloadUrl}
                    download={videoDownloadFileName}
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
                  onClick={() => void onStartCameraDetection()}
                >
                  {cameraBusy ? '启动中...' : '开始实时识别'}
                </button>
                <button
                  type="button"
                  className="action action--secondary"
                  disabled={!canStopCamera}
                  onClick={onStopCamera}
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
    </Root>
  )
}
