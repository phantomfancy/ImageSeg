import { startTransition, useEffect, useState } from 'react'
import type { ImportedModel } from './lib/onnxRuntime'
import { inspectImportedModel, runSingleImageDetection } from './lib/onnxRuntime'
import './App.css'

function App() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreviewUrl, setImagePreviewUrl] = useState('')
  const [importedModel, setImportedModel] = useState<ImportedModel | null>(null)
  const [statusMessage, setStatusMessage] = useState('请先导入待识别图片和 ONNX 模型。')
  const [modelBusy, setModelBusy] = useState(false)
  const [detectBusy, setDetectBusy] = useState(false)
  const [annotatedImageUrl, setAnnotatedImageUrl] = useState('')
  const [resultProvider, setResultProvider] = useState('')
  const [detectionItems, setDetectionItems] = useState<Array<{
    label: string
    confidence: number
    boxSummary: string
  }>>([])

  useEffect(() => () => {
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl)
    }
  }, [imagePreviewUrl])

  const canRun = Boolean(imageFile && importedModel && !modelBusy && !detectBusy)
  const deferredContract = importedModel?.contract ?? null

  function updateImageFile(nextFile: File | null) {
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl)
    }

    setImageFile(nextFile)
    setImagePreviewUrl(nextFile ? URL.createObjectURL(nextFile) : '')
  }

  async function handleModelSelection(file: File | null) {
    setImportedModel(null)
    setAnnotatedImageUrl('')
    setDetectionItems([])

    if (!file) {
      setStatusMessage('尚未选择模型文件。')
      return
    }

    setModelBusy(true)
    setStatusMessage(`正在解析模型 ${file.name}...`)

    try {
      const model = await inspectImportedModel(file)
      startTransition(() => {
        setImportedModel(model)
        setStatusMessage(`模型解析成功：${model.contract.family}，当前加载执行提供器为 ${model.providerName}。`)
      })
    } catch (error) {
      setStatusMessage(`模型解析失败：${formatError(error)}`)
    } finally {
      setModelBusy(false)
    }
  }

  async function handleRunDetection() {
    if (!imageFile || !importedModel) {
      return
    }

    setDetectBusy(true)
    setAnnotatedImageUrl('')
    setDetectionItems([])
    setStatusMessage(`正在使用 ${importedModel.contract.family} 模型执行浏览器端识别...`)

    try {
      const detectionRun = await runSingleImageDetection(importedModel, imageFile)
      startTransition(() => {
        setAnnotatedImageUrl(detectionRun.annotatedImageDataUrl)
        setResultProvider(detectionRun.providerName)
        setDetectionItems(
          detectionRun.recognitionResult.detections.map((item) => ({
            label: item.label,
            confidence: item.confidence,
            boxSummary: `${item.box.x.toFixed(1)}, ${item.box.y.toFixed(1)}, ${item.box.width.toFixed(1)}, ${item.box.height.toFixed(1)}`,
          })),
        )
        setStatusMessage(
          detectionRun.recognitionResult.detections.length === 0
            ? '识别完成，当前图片未检测到目标。'
            : `识别完成，共检测到 ${detectionRun.recognitionResult.detections.length} 个目标。`,
        )
      })
    } catch (error) {
      setStatusMessage(`识别失败：${formatError(error)}`)
    } finally {
      setDetectBusy(false)
    }
  }

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero__eyebrow">4C-ai装备识别工具</div>
        <h1>React 19 + TypeScript 的本地 ONNX 工作台</h1>
        <p className="hero__copy">
          当前版本移除了后端识别依赖，模型导入、格式识别、推理编排和输出解码全部在浏览器端完成。
          `Contracts` 同时约束 Hugging Face 风格和 Ultralytics 风格模型。
        </p>
      </section>

      <section className="grid">
        <article className="panel panel--input">
          <header className="panel__header">
            <h2>输入与导入</h2>
            <p>先导入图片，再导入 ONNX 模型。</p>
          </header>

          <label className="field">
            <span className="field__label">图片文件</span>
            <input
              type="file"
              accept="image/*"
              onChange={(event) => {
                updateImageFile(event.target.files?.[0] ?? null)
                setAnnotatedImageUrl('')
                setDetectionItems([])
              }}
            />
          </label>

          <label className="field">
            <span className="field__label">ONNX 模型</span>
            <input
              type="file"
              accept=".onnx,application/octet-stream"
              onChange={(event) => {
                void handleModelSelection(event.target.files?.[0] ?? null)
              }}
            />
          </label>

          <div className="actions">
            <button type="button" className="action action--primary" disabled={!canRun} onClick={() => void handleRunDetection()}>
              {detectBusy ? '识别中...' : '执行单图识别'}
            </button>
          </div>

          <div className="status-box">
            <div className="status-box__title">状态</div>
            <p>{statusMessage}</p>
          </div>
        </article>

        <article className="panel panel--contract">
          <header className="panel__header">
            <h2>模型契约</h2>
            <p>导入后先根据输入输出签名解析 family、预处理和解码规则。</p>
          </header>

          {deferredContract ? (
            <div className="contract">
              <div className="metric-grid">
                <Metric label="Display Name" value={deferredContract.displayName} />
                <Metric label="Family" value={deferredContract.family} />
                <Metric label="Provider" value={importedModel?.providerName ?? '-'} />
                <Metric label="Label Source" value={deferredContract.labelSource} />
                <Metric label="Input Tensor" value={deferredContract.preprocess.inputTensorName} />
                <Metric
                  label="Image Size"
                  value={`${deferredContract.preprocess.imageWidth} x ${deferredContract.preprocess.imageHeight}`}
                />
              </div>

              <SignatureTable title="Inputs" items={deferredContract.inputs.map((item) => ({
                name: item.name,
                dims: formatDims(item.dimensions),
              }))} />

              <SignatureTable title="Outputs" items={deferredContract.outputs.map((item) => ({
                name: item.name,
                dims: formatDims(item.dimensions),
              }))} />

              <div className="warnings">
                <div className="warnings__title">Warnings</div>
                {deferredContract.warnings.length === 0 ? (
                  <p>当前模型未发现额外警告。</p>
                ) : (
                  deferredContract.warnings.map((item) => (
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
            <h2>图片预览</h2>
            <p>左侧显示原图，识别完成后显示结果叠加图。</p>
          </header>

          <div className="preview-grid">
            <PreviewCard title="原图" imageUrl={imagePreviewUrl} emptyText="未选择图片。" />
            <PreviewCard title="结果叠加图" imageUrl={annotatedImageUrl} emptyText="尚未执行识别。" />
          </div>
        </article>

        <article className="panel panel--detections">
          <header className="panel__header">
            <h2>检测结果</h2>
            <p>输出来自 `Contracts` 约束的统一检测结构。</p>
          </header>

          <div className="metric-grid metric-grid--compact">
            <Metric label="Runtime" value={resultProvider || '-'} />
            <Metric label="Detection Count" value={String(detectionItems.length)} />
          </div>

          {detectionItems.length === 0 ? (
            <EmptyState text={detectBusy ? '正在等待推理结果...' : '当前没有可展示的检测项。'} />
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

function PreviewCard(props: { title: string; imageUrl: string; emptyText: string }) {
  return (
    <div className="preview-card">
      <div className="preview-card__title">{props.title}</div>
      {props.imageUrl ? (
        <img className="preview-card__image" src={props.imageUrl} alt={props.title} />
      ) : (
        <EmptyState text={props.emptyText} />
      )}
    </div>
  )
}

function EmptyState(props: { text: string }) {
  return (
    <div className="empty-state">
      <p>{props.text}</p>
    </div>
  )
}

function formatDims(dims: Array<number | null>): string {
  return `[${dims.map((item) => item ?? '?').join(', ')}]`
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export default App
