import * as ort from 'onnxruntime-web/webgpu'
import {
  decodeDetections,
  resolveModelContract,
  type RecognitionResult,
  type ResolvedModelContract,
  type TensorLike,
} from '@contracts'
import { inspectOnnxModelBytes } from './modelIntrospection'

ort.env.wasm.proxy = false
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = {
  mjs: new URL('/ort/ort-wasm-simd-threaded.jsep.mjs', window.location.href),
  wasm: new URL('/ort/ort-wasm-simd-threaded.jsep.wasm', window.location.href),
}

if (ort.env.webgpu) {
  ort.env.webgpu.powerPreference = 'high-performance'
}

type SessionState = {
  session: ort.InferenceSession
  providerName: 'webgpu' | 'wasm'
}

const sessionCache = new Map<string, Promise<SessionState>>()

export interface ImportedModel {
  key: string
  fileName: string
  bytes: Uint8Array
  contract: ResolvedModelContract
  providerName: 'webgpu' | 'wasm' | 'unavailable'
}

export interface DetectionRun {
  providerName: 'webgpu' | 'wasm'
  annotatedImageDataUrl: string
  recognitionResult: RecognitionResult
}

export async function inspectImportedModel(file: File): Promise<ImportedModel> {
  const bytes = new Uint8Array(await file.arrayBuffer())
  const key = `${file.name}:${file.size}:${file.lastModified}`
  const parsedModel = inspectOnnxModelBytes(bytes)
  const contract = resolveModelContract({
    inputs: parsedModel.inputs,
    outputs: parsedModel.outputs,
    displayName: file.name,
    labels: parsedModel.labels,
  })

  try {
    const sessionState = await getSessionState(key, bytes)

    return {
      key,
      fileName: file.name,
      bytes,
      contract,
      providerName: sessionState.providerName,
    }
  } catch (error) {
    return {
      key,
      fileName: file.name,
      bytes,
      contract: {
        ...contract,
        warnings: [
          `当前运行时暂时无法为该模型建立推理会话：${formatError(error)}`,
          ...contract.warnings,
        ],
      },
      providerName: 'unavailable',
    }
  }
}

export async function runSingleImageDetection(
  model: ImportedModel,
  imageFile: File,
): Promise<DetectionRun> {
  const sessionState = await getSessionState(model.key, model.bytes)
  const sourceCanvas = await loadImageCanvas(imageFile)
  const inputTensor = createInputTensor(sourceCanvas, model.contract)

  const outputMap = await sessionState.session.run({
    [model.contract.preprocess.inputTensorName]: inputTensor,
  })

  const normalizedOutputs = await normalizeOutputs(outputMap)
  const detections = decodeDetections(
    model.contract,
    normalizedOutputs,
    sourceCanvas.width,
    sourceCanvas.height,
  )

  return {
    providerName: sessionState.providerName,
    annotatedImageDataUrl: drawDetections(sourceCanvas, detections),
    recognitionResult: {
      inputSource: imageFile.name,
      modelVersion: model.fileName,
      detectedAtUtc: new Date().toISOString(),
      detections,
    },
  }
}

async function getSessionState(key: string, bytes: Uint8Array): Promise<SessionState> {
  const existing = sessionCache.get(key)
  if (existing) {
    return existing
  }

  const promise = createSessionWithFallback(bytes).catch((error: unknown) => {
    sessionCache.delete(key)
    throw error
  })

  sessionCache.set(key, promise)
  return promise
}

async function createSessionWithFallback(bytes: Uint8Array): Promise<SessionState> {
  const plans: Array<{
    providerName: 'webgpu' | 'wasm'
    options: ort.InferenceSession.SessionOptions
  }> = []

  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    plans.push({
      providerName: 'webgpu',
      options: {
        executionProviders: ['webgpu'],
      },
    })
  }

  plans.push({
    providerName: 'wasm',
    options: {
      executionProviders: ['wasm'],
    },
  })

  const errors: string[] = []
  for (const plan of plans) {
    try {
      const session = await ort.InferenceSession.create(bytes, plan.options)
      return {
        session,
        providerName: plan.providerName,
      }
    } catch (error) {
      errors.push(`${plan.providerName}: ${formatError(error)}`)
    }
  }

  throw new Error(`无法加载 ONNX 模型。${errors.join('；')}`)
}

async function normalizeOutputs(
  outputMap: ort.InferenceSession.ReturnType,
): Promise<Record<string, TensorLike>> {
  const entries = await Promise.all(
    Object.entries(outputMap).map(async ([name, value]) => {
      const tensor = value as ort.Tensor
      const rawData = typeof tensor.getData === 'function'
        ? await tensor.getData()
        : tensor.data
      const data = ensureNumericTensorData(rawData, name)

      return [name, {
        dims: [...tensor.dims],
        data,
      } satisfies TensorLike] as const
    }),
  )

  return Object.fromEntries(entries)
}

async function loadImageCanvas(file: File): Promise<HTMLCanvasElement> {
  if (typeof createImageBitmap === 'function') {
    const bitmap = await createImageBitmap(file)
    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const context = get2dContext(canvas)
    context.drawImage(bitmap, 0, 0)
    bitmap.close()
    return canvas
  }

  const objectUrl = URL.createObjectURL(file)
  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const element = new Image()
      element.onload = () => resolve(element)
      element.onerror = () => reject(new Error('无法加载待识别图片。'))
      element.src = objectUrl
    })

    const canvas = document.createElement('canvas')
    canvas.width = image.naturalWidth || image.width
    canvas.height = image.naturalHeight || image.height
    get2dContext(canvas).drawImage(image, 0, 0)
    return canvas
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}

function createInputTensor(
  sourceCanvas: HTMLCanvasElement,
  contract: ResolvedModelContract,
): ort.Tensor {
  const targetWidth = contract.preprocess.imageWidth
  const targetHeight = contract.preprocess.imageHeight
  const preprocessCanvas = document.createElement('canvas')
  preprocessCanvas.width = targetWidth
  preprocessCanvas.height = targetHeight

  const context = get2dContext(preprocessCanvas)
  context.drawImage(sourceCanvas, 0, 0, targetWidth, targetHeight)

  const imageData = context.getImageData(0, 0, targetWidth, targetHeight)
  const tensorData = new Float32Array(3 * targetWidth * targetHeight)
  const channelSize = targetWidth * targetHeight

  for (let index = 0; index < channelSize; index += 1) {
    const pixelOffset = index * 4
    tensorData[index] = imageData.data[pixelOffset] / 255
    tensorData[channelSize + index] = imageData.data[pixelOffset + 1] / 255
    tensorData[channelSize * 2 + index] = imageData.data[pixelOffset + 2] / 255
  }

  return new ort.Tensor('float32', tensorData, [1, 3, targetHeight, targetWidth])
}

function ensureNumericTensorData(data: ort.Tensor['data'], outputName: string): ArrayLike<number> {
  if (Array.isArray(data)) {
    throw new Error(`输出 ${outputName} 不是数值 tensor。`)
  }

  return data as ArrayLike<number>
}

function drawDetections(
  sourceCanvas: HTMLCanvasElement,
  detections: RecognitionResult['detections'],
): string {
  const canvas = document.createElement('canvas')
  canvas.width = sourceCanvas.width
  canvas.height = sourceCanvas.height

  const context = get2dContext(canvas)
  context.drawImage(sourceCanvas, 0, 0)
  context.lineWidth = 3
  context.font = "15px 'Cascadia Code', 'Consolas', monospace"
  context.textBaseline = 'top'

  detections.forEach((item, index) => {
    const color = pickColor(index)
    const label = `${item.label} ${(item.confidence * 100).toFixed(1)}%`
    const metrics = context.measureText(label)
    const textWidth = metrics.width + 14
    const textHeight = 26
    const textY = item.box.y > textHeight ? item.box.y - textHeight : item.box.y

    context.strokeStyle = color
    context.fillStyle = color
    context.strokeRect(item.box.x, item.box.y, item.box.width, item.box.height)
    context.fillRect(item.box.x, textY, textWidth, textHeight)
    context.fillStyle = '#f8f5ed'
    context.fillText(label, item.box.x + 7, textY + 5)
  })

  return canvas.toDataURL('image/png')
}

function get2dContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error('浏览器不支持 Canvas 2D。')
  }

  return context
}

function pickColor(index: number): string {
  const palette = ['#d94f30', '#11698e', '#0f9d58', '#d68910', '#6a4c93', '#2f4858']
  return palette[index % palette.length]
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
