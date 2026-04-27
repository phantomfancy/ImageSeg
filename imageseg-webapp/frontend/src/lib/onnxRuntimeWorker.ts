import * as ort from 'onnxruntime-web/webgpu'
import type { TensorLike } from '../../../contracts/src/index.ts'
import type { RuntimeWorkerInitOptions, RuntimeWorkerSessionMetadata } from './onnxRuntime.ts'

const runtimeBaseHref = typeof self !== 'undefined' && self.location?.href
  ? self.location.href
  : import.meta.url

ort.env.wasm.proxy = false
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = {
  mjs: new URL('/ort/ort-wasm-simd-threaded.asyncify.mjs', runtimeBaseHref),
  wasm: new URL('/ort/ort-wasm-simd-threaded.asyncify.wasm', runtimeBaseHref),
}

if (ort.env.webgpu) {
  ort.env.webgpu.powerPreference = 'high-performance'
}

const GPU_BUFFER_USAGE_COPY_DST = 0x0008
const GPU_BUFFER_USAGE_STORAGE = 0x0080

type WorkerRequest =
  | (RuntimeWorkerInitOptions & { type: 'init'; requestId: number })
  | { type: 'run'; requestId: number; inputData: Float32Array; fetchNames: readonly string[] }
  | { type: 'release'; requestId: number }

type WorkerResponse =
  | { type: 'success'; requestId: number; result: RuntimeWorkerSessionMetadata | Record<string, TensorLike> | undefined }
  | { type: 'error'; requestId: number; message: string }

type WorkerSuccessResult = RuntimeWorkerSessionMetadata | Record<string, TensorLike> | undefined

const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null
  postMessage(message: WorkerResponse, transfers?: Transferable[]): void
}

type WorkerSessionState = {
  session: ort.InferenceSession
  inputName: string
  outputNames: readonly string[]
  inputTensor: ort.Tensor
  inputGpuBuffer: GPUBuffer
  inputByteLength: number
  gpuDevice: GPUDevice
}

type NumericTensorData =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array

let sessionState: WorkerSessionState | null = null

workerScope.onmessage = (event: MessageEvent<WorkerRequest>) => {
  void handleRequest(event.data)
}

async function handleRequest(request: WorkerRequest) {
  try {
    if (request.type === 'init') {
      const metadata = await initializeSession(request)
      postSuccess(request.requestId, metadata)
      return
    }

    if (request.type === 'release') {
      await releaseSession()
      postSuccess(request.requestId, undefined)
      return
    }

    const outputs = await runInference(request)
    const transfers = Object.values(outputs).map((tensor) => getTransferableBuffer(tensor.data))
    postSuccess(request.requestId, outputs, transfers)
  } catch (error) {
    postError(request.requestId, formatError(error))
  }
}

async function initializeSession(request: RuntimeWorkerInitOptions): Promise<RuntimeWorkerSessionMetadata> {
  await releaseSession()

  let session: ort.InferenceSession | null = null
  let inputGpuBuffer: GPUBuffer | null = null
  try {
    session = await ort.InferenceSession.create(request.bytes, {
      executionProviders: ['webgpu'],
      enableGraphCapture: request.enableGraphCapture,
      preferredOutputLocation: 'gpu-buffer',
    })
    const inputName = resolveInputName(session.inputNames, request.preferredInputName)
    const gpuDevice = await ort.env.webgpu.device
    const inputByteLength = calculateInputByteLength(request.inputDims)
    inputGpuBuffer = gpuDevice.createBuffer({
      size: inputByteLength,
      usage: GPU_BUFFER_USAGE_COPY_DST | GPU_BUFFER_USAGE_STORAGE,
    })
    const inputTensor = ort.Tensor.fromGpuBuffer(inputGpuBuffer, {
      dataType: 'float32',
      dims: request.inputDims,
    })

    sessionState = {
      session,
      inputName,
      outputNames: session.outputNames,
      inputTensor,
      inputGpuBuffer,
      inputByteLength,
      gpuDevice,
    }

    return {
      inputName,
      outputNames: [...session.outputNames],
    }
  } catch (error) {
    await safelyReleaseSession(session)
    inputGpuBuffer?.destroy()
    throw error
  }
}

async function runInference(request: Extract<WorkerRequest, { type: 'run' }>): Promise<Record<string, TensorLike>> {
  if (!sessionState) {
    throw new Error('ONNX Runtime Worker 尚未初始化。')
  }

  if (request.inputData.byteLength !== sessionState.inputByteLength) {
    throw new Error('推理输入尺寸与当前模型不匹配。')
  }

  sessionState.gpuDevice.queue.writeBuffer(sessionState.inputGpuBuffer, 0, request.inputData)
  const outputMap = await sessionState.session.run({
    [sessionState.inputName]: sessionState.inputTensor,
  }, request.fetchNames)
  const outputs = await normalizeOutputs(outputMap)
  await waitForGpuDeviceIdle(sessionState.gpuDevice)
  return outputs
}

async function normalizeOutputs(
  outputMap: ort.InferenceSession.ReturnType,
): Promise<Record<string, TensorLike>> {
  const entries = await Promise.all(
    Object.entries(outputMap).map(async ([name, value]) => {
      const tensor = value as ort.Tensor
      try {
        const rawData = typeof tensor.getData === 'function'
          ? await tensor.getData(true)
          : tensor.data
        const data = copyNumericTensorData(rawData, name)

        return [name, {
          dims: [...tensor.dims],
          data,
        } satisfies TensorLike] as const
      } finally {
        try {
          tensor.dispose()
        } catch {
          // 输出 tensor 可能已由 ORT 释放；忽略重复释放。
        }
      }
    }),
  )

  return Object.fromEntries(entries)
}

async function releaseSession() {
  if (!sessionState) {
    return
  }

  const currentState = sessionState
  sessionState = null
  await waitForGpuDeviceIdle(currentState.gpuDevice)
  await safelyReleaseSession(currentState.session)
  try {
    currentState.inputTensor.dispose()
  } catch {
    // 忽略重复释放。
  }
  currentState.inputGpuBuffer.destroy()
}

async function safelyReleaseSession(session: ort.InferenceSession | null) {
  if (!session || typeof session.release !== 'function') {
    return
  }

  try {
    await session.release()
  } catch {
    // Worker 会在模型切换时被终止；释放失败不需要污染下一次 runtime。
  }
}

async function waitForGpuDeviceIdle(device: GPUDevice) {
  try {
    await device.queue.onSubmittedWorkDone()
  } catch {
    // 设备丢失时让调用方继续走错误或终止 worker。
  }
}

function calculateInputByteLength(inputDims: readonly number[]): number {
  return inputDims.reduce((total, item) => total * item, 1) * Float32Array.BYTES_PER_ELEMENT
}

function resolveInputName(inputNames: readonly string[], preferredName: string): string {
  return resolveTensorName(inputNames, preferredName, '输入')
}

function resolveTensorName(
  actualNames: readonly string[],
  preferredName: string,
  kind: '输入' | '输出',
): string {
  if (preferredName) {
    const normalizedPreferredName = preferredName.toLowerCase()
    const exactMatch = actualNames.find((name) => name.toLowerCase() === normalizedPreferredName)
    if (exactMatch) {
      return exactMatch
    }

    const partialMatch = actualNames.find((name) => name.toLowerCase().includes(normalizedPreferredName))
    if (partialMatch) {
      return partialMatch
    }
  }

  if (actualNames.length === 1) {
    return actualNames[0]
  }

  throw new Error(`无法在模型${kind}中匹配 ${preferredName || '目标名称'}。`)
}

function copyNumericTensorData(data: ort.Tensor['data'], outputName: string): NumericTensorData {
  if (Array.isArray(data) || data instanceof BigInt64Array || data instanceof BigUint64Array) {
    throw new Error(`输出 ${outputName} 不是数值 tensor。`)
  }

  if (!ArrayBuffer.isView(data)) {
    throw new Error(`输出 ${outputName} 不是数值 tensor。`)
  }

  return data.slice() as NumericTensorData
}

function getTransferableBuffer(data: TensorLike['data']): ArrayBuffer {
  return (data as unknown as ArrayBufferView<ArrayBuffer>).buffer
}

function postSuccess(requestId: number, result: WorkerSuccessResult, transfers: Transferable[] = []) {
  workerScope.postMessage({
    type: 'success',
    requestId,
    result,
  } satisfies WorkerResponse, transfers)
}

function postError(requestId: number, message: string) {
  workerScope.postMessage({
    type: 'error',
    requestId,
    message,
  } satisfies WorkerResponse)
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
