import type { CSSProperties, PointerEvent as ReactPointerEvent, RefObject, WheelEvent as ReactWheelEvent } from 'react'
import styled from 'styled-components'
import type { InputMode, PreviewViewerOffset, PreviewZoomTarget } from '../app/types'

const Root = styled.div`
  position: fixed;
  inset: 0;
  z-index: 40;
  display: grid;
  place-items: center;
  overscroll-behavior: none;

  .preview-zoom__backdrop {
    position: absolute;
    inset: 0;
    border: none;
    background: var(--modal-backdrop);
    cursor: pointer;
  }

  .preview-zoom__dialog {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto;
    gap: 16px;
    width: min(1180px, calc(100vw - 48px));
    height: min(860px, calc(100vh - 48px));
    max-height: calc(100vh - 48px);
    padding: 18px 18px 20px;
    border-radius: 26px;
    overflow: hidden;
    border: 1px solid var(--preview-zoom-control-border);
    background: var(--preview-zoom-surface);
    box-shadow: 0 24px 56px color-mix(in srgb, var(--ink) 32%, transparent);
  }

  .preview-zoom__dialog:fullscreen {
    width: 100vw;
    max-height: 100vh;
    height: 100vh;
    padding: 18px;
    border: none;
    border-radius: 0;
  }

  .preview-zoom__header {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 18px;
    align-items: start;
  }

  .preview-zoom__summary {
    display: grid;
    gap: 10px;
    min-width: 0;
  }

  .preview-zoom__summary-top {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
  }

  .preview-zoom__meta {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }

  .preview-zoom__meta-item {
    padding: 6px 10px;
    border-radius: 999px;
    background: var(--preview-zoom-control-bg);
    color: var(--preview-zoom-text-muted);
    font-size: 0.82rem;
    line-height: 1.4;
  }

  .preview-zoom__toolbar {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 12px;
    min-width: 0;
    overflow-x: auto;
    scrollbar-width: thin;
  }

  .preview-zoom__toolbar-group {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    justify-content: flex-end;
    min-width: max-content;
  }

  .preview-zoom__badge,
  .preview-zoom__scale {
    padding: 8px 12px;
    border-radius: 999px;
    background: var(--preview-zoom-control-bg);
    color: var(--preview-zoom-text-muted);
  }

  .preview-zoom__badge {
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.74rem;
  }

  .preview-zoom__scale {
    font-family: var(--mono);
  }

  .preview-zoom__action {
    min-height: 40px;
    padding: 10px 14px;
    border: 1px solid var(--preview-zoom-control-border);
    border-radius: 999px;
    background: var(--preview-zoom-control-bg);
    color: var(--preview-zoom-text-muted);
    cursor: pointer;
    white-space: nowrap;
  }

  .preview-zoom__action:hover:not(:disabled) {
    background: var(--preview-zoom-control-bg-hover);
  }

  .preview-zoom__action--active {
    border-color: var(--preview-zoom-control-active-border);
    background: var(--preview-zoom-control-active-bg);
  }

  .preview-zoom__action:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .preview-zoom__action--close {
    border-color: var(--preview-zoom-control-active-border);
    background: var(--preview-zoom-close-bg);
  }

  .preview-zoom__viewport {
    overflow: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    min-width: 0;
    height: 100%;
    padding: 14px;
    border-radius: 20px;
    background: var(--preview-zoom-viewport-bg);
  }

  .preview-zoom__viewport--interactive {
    overflow: hidden;
    overscroll-behavior: none;
    touch-action: none;
  }

  .preview-zoom__media-wrapper {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    max-width: 100%;
    max-height: 100%;
    transform-origin: center center;
    transition: transform 140ms ease;
  }

  .preview-zoom__media-wrapper--draggable {
    cursor: grab;
  }

  .preview-zoom__media-wrapper--draggable:active {
    cursor: grabbing;
  }

  .preview-zoom__image,
  .preview-zoom__video,
  .preview-zoom__canvas {
    display: block;
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    border-radius: 16px;
    object-fit: contain;
  }

  .preview-zoom__video {
    background: var(--modal-video-bg);
    box-shadow: inset 0 0 0 1px var(--modal-video-outline);
  }

  @media (max-width: 1040px) {
    .preview-zoom__dialog {
      width: min(100vw - 28px, 1100px);
      height: calc(100vh - 28px);
      max-height: calc(100vh - 28px);
      padding: 14px;
    }

    .preview-zoom__header {
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 14px;
    }
  }

  @media (max-width: 720px) {
    .preview-zoom__dialog {
      gap: 10px;
      width: calc(100vw - 20px);
      height: calc(100vh - 20px);
      max-height: calc(100vh - 20px);
      padding: 10px;
      border-radius: 18px;
    }

    .preview-zoom__header {
      grid-template-columns: 1fr;
      gap: 8px;
    }

    .preview-zoom__summary {
      gap: 7px;
    }

    .preview-zoom__summary-top,
    .preview-zoom__meta {
      flex-direction: row;
      align-items: center;
      gap: 6px;
    }

    .preview-zoom__toolbar-group {
      flex-wrap: nowrap;
      justify-content: flex-start;
      gap: 6px;
    }

    .preview-zoom__toolbar {
      justify-content: flex-start;
      width: 100%;
    }

    .preview-zoom__action {
      min-height: 32px;
      padding: 7px 9px;
      font-size: 0.76rem;
    }

    .preview-zoom__scale,
    .preview-zoom__badge,
    .preview-zoom__meta-item {
      width: auto;
      padding: 5px 8px;
      justify-content: center;
      text-align: center;
      font-size: 0.7rem;
    }

    .preview-zoom__viewport {
      padding: 8px;
      border-radius: 14px;
    }
  }
`

type PreviewZoomModalProps = {
  closePreviewZoom: () => void
  currentTimeLabel: string
  detectionItemCount: number
  handlePreviewMediaDoubleClick: () => void
  handlePreviewPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void
  handlePreviewPointerMove: (event: ReactPointerEvent<HTMLDivElement>) => void
  handlePreviewPointerRelease: (event: ReactPointerEvent<HTMLDivElement>) => void
  handlePreviewWheel: (event: ReactWheelEvent<HTMLDivElement>) => void
  imagePreviewUrl: string
  inputMode: InputMode
  isPreviewViewerFullscreen: boolean
  isScalablePreviewTarget: boolean
  previewViewerOffset: PreviewViewerOffset
  previewViewerScale: number
  previewZoomCanvasRef: RefObject<HTMLCanvasElement | null>
  previewZoomDialogRef: RefObject<HTMLDivElement | null>
  previewZoomMediaRef: RefObject<HTMLDivElement | null>
  previewZoomScaleLabel: string
  previewZoomTarget: PreviewZoomTarget
  previewZoomVideoRef: RefObject<HTMLVideoElement | null>
  previewZoomViewportRef: RefObject<HTMLDivElement | null>
  resultProvider: string
  scaleDecreaseDisabled: boolean
  scaleIncreaseDisabled: boolean
  setPreviewViewerDefault: () => void
  setPreviewViewerScaleStepDown: () => void
  setPreviewViewerScaleStepUp: () => void
  togglePreviewViewerFullscreen: () => Promise<void>
}

export function PreviewZoomModal(props: PreviewZoomModalProps) {
  const {
    closePreviewZoom,
    currentTimeLabel,
    detectionItemCount,
    handlePreviewMediaDoubleClick,
    handlePreviewPointerDown,
    handlePreviewPointerMove,
    handlePreviewPointerRelease,
    handlePreviewWheel,
    imagePreviewUrl,
    inputMode,
    isPreviewViewerFullscreen,
    isScalablePreviewTarget,
    previewViewerOffset,
    previewViewerScale,
    previewZoomCanvasRef,
    previewZoomDialogRef,
    previewZoomMediaRef,
    previewZoomScaleLabel,
    previewZoomTarget,
    previewZoomVideoRef,
    previewZoomViewportRef,
    resultProvider,
    scaleDecreaseDisabled,
    scaleIncreaseDisabled,
    setPreviewViewerDefault,
    setPreviewViewerScaleStepDown,
    setPreviewViewerScaleStepUp,
    togglePreviewViewerFullscreen,
  } = props

  const mediaWrapperStyle: CSSProperties | undefined = isScalablePreviewTarget
    ? {
      transform: `translate(${previewViewerOffset.x}px, ${previewViewerOffset.y}px) scale(${previewViewerScale})`,
    }
    : undefined

  return (
    <Root role="dialog" aria-modal="true" aria-label="放大预览">
      <button
        type="button"
        className="preview-zoom__backdrop"
        aria-label="关闭放大预览"
        onClick={closePreviewZoom}
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
                <span className="preview-zoom__meta-item">检测项 {detectionItemCount}</span>
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
                    disabled={scaleDecreaseDisabled}
                    onClick={setPreviewViewerScaleStepDown}
                  >
                    缩小
                  </button>
                  <button
                    type="button"
                    className={`preview-zoom__action${Math.abs(previewViewerScale - 1) < 0.01 ? ' preview-zoom__action--active' : ''}`}
                    onClick={setPreviewViewerDefault}
                  >
                    适合窗口
                  </button>
                  <button
                    type="button"
                    className="preview-zoom__action"
                    disabled={scaleIncreaseDisabled}
                    onClick={setPreviewViewerScaleStepUp}
                  >
                    放大
                  </button>
                  <button
                    type="button"
                    className="preview-zoom__action"
                    onClick={setPreviewViewerDefault}
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
                onClick={closePreviewZoom}
              >
                关闭
              </button>
            </div>
          </div>
        </div>

        <div
          ref={previewZoomViewportRef}
          className={`preview-zoom__viewport${isScalablePreviewTarget ? ' preview-zoom__viewport--interactive' : ''}`}
          onWheel={isScalablePreviewTarget ? handlePreviewWheel : undefined}
        >
          <div
            ref={previewZoomMediaRef}
            className={`preview-zoom__media-wrapper${
              isScalablePreviewTarget && previewViewerScale > 1
                ? ' preview-zoom__media-wrapper--draggable'
                : ''
            }`}
            style={mediaWrapperStyle}
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
    </Root>
  )
}
