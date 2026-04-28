import assert from 'node:assert/strict'

import {
  prepareImportedModelSession,
  releaseCachedModelSessions,
  runDetectionOnCanvas,
  setWebGpuRuntimeHooks,
  type ImportedModel,
  type RuntimeWorkerClient,
  type RuntimeWorkerInitOptions,
  type WebGpuRuntimeHooks,
} from '../src/lib/onnxRuntime.ts'
import {
  resolveModelContract,
  toTensorDescriptors,
  type TensorLike,
} from '../../contracts/src/index.ts'

const globalScope = globalThis as typeof globalThis & {
  HTMLCanvasElement?: typeof HTMLCanvasElement
}
const previousHTMLCanvasElement = globalScope.HTMLCanvasElement

const workerCreates: number[] = []
const workerTerminates: number[] = []
const workerInits: Array<{
  workerId: number
  modelId: number
  enableGraphCapture: boolean
}> = []
const workerRuns: Array<{
  workerId: number
  modelId: number
  inputByteLength: number
  fetchNames: readonly string[]
}> = []
const workerReleases: number[] = []

let nextWorkerId = 1

class FakeCanvasContext {
  fillStyle = '#000'
  strokeStyle = '#000'
  lineWidth = 1
  font = ''
  textBaseline: CanvasTextBaseline = 'top'

  clearRect() {}

  drawImage() {}

  fillRect() {}

  strokeRect() {}

  fillText() {}

  measureText(text: string) {
    return { width: text.length * 8 } as TextMetrics
  }

  getImageData(_x: number, _y: number, width: number, height: number) {
    return {
      data: new Uint8ClampedArray(width * height * 4),
    }
  }
}

class FakeCanvasElement {
  width = 0
  height = 0
  private readonly context = new FakeCanvasContext()

  getContext(kind: string): CanvasRenderingContext2D | null {
    return kind === '2d' ? (this.context as unknown as CanvasRenderingContext2D) : null
  }
}

class FakeWorkerClient implements RuntimeWorkerClient {
  private readonly workerId = nextWorkerId++
  private terminated = false
  private modelId = 0

  constructor() {
    workerCreates.push(this.workerId)
  }

  async init(options: RuntimeWorkerInitOptions) {
    this.assertActive()
    this.modelId = options.bytes[0] ?? 0
    workerInits.push({
      workerId: this.workerId,
      modelId: this.modelId,
      enableGraphCapture: options.enableGraphCapture,
    })

    if (this.modelId === 3) {
      throw new Error('simulated worker init failure for model 3')
    }

    return {
      inputName: 'images',
      outputNames: ['output0'],
    }
  }

  async run(inputData: Float32Array, fetchNames: readonly string[]) {
    this.assertActive()
    workerRuns.push({
      workerId: this.workerId,
      modelId: this.modelId,
      inputByteLength: inputData.byteLength,
      fetchNames: [...fetchNames],
    })

    return {
      output0: createOutputTensor(),
    }
  }

  async release() {
    this.assertActive()
    workerReleases.push(this.workerId)
  }

  terminate() {
    if (this.terminated) {
      return
    }

    this.terminated = true
    workerTerminates.push(this.workerId)
  }

  private assertActive() {
    if (this.terminated) {
      throw new Error(`worker ${this.workerId} already terminated`)
    }
  }
}

function createOutputTensor(): TensorLike {
  const anchorCount = 20
  const data = new Float32Array(6 * anchorCount)
  data[0] = 0.5
  data[anchorCount] = 0.5
  data[anchorCount * 2] = 0.2
  data[anchorCount * 3] = 0.2
  data[anchorCount * 4] = 0.95
  data[anchorCount * 5] = 0.05

  return {
    dims: [1, 6, anchorCount],
    data,
  }
}

function createTestModel(modelId: number, fileName: string): ImportedModel {
  const contract = resolveModelContract({
    inputs: toTensorDescriptors([
      { name: 'images', shape: [1, 3, 640, 640] },
    ]),
    outputs: toTensorDescriptors([
      { name: 'output0', shape: [1, 6, 20] },
    ]),
    displayName: fileName,
  })

  return {
    key: `${fileName}:${modelId}`,
    fileName,
    sourceFile: new File([new Uint8Array([modelId])], fileName, { lastModified: modelId }),
    contract,
    providerName: 'webgpu',
    sidecars: {},
    webGpuCompatibility: {
      supported: true,
      issues: [],
    },
  }
}

const hooks: WebGpuRuntimeHooks = {
  isWebGpuSupported: () => true,
  createCanvas: () => new FakeCanvasElement() as unknown as HTMLCanvasElement,
  createWorkerClient: () => new FakeWorkerClient(),
}

globalScope.HTMLCanvasElement = FakeCanvasElement as unknown as typeof HTMLCanvasElement
setWebGpuRuntimeHooks(hooks)

const modelA = createTestModel(1, 'model-a.onnx')
const modelB = createTestModel(2, 'model-b.onnx')
const modelC = createTestModel(3, 'model-c.onnx')
const modelD = createTestModel(4, 'model-d.onnx')
const sourceCanvas = new FakeCanvasElement()
sourceCanvas.width = 640
sourceCanvas.height = 640

try {
  await prepareImportedModelSession(modelA)
  const resultA = await runDetectionOnCanvas(
    modelA,
    sourceCanvas as unknown as HTMLCanvasElement,
    'frame-a',
    undefined,
    sourceCanvas as unknown as HTMLCanvasElement,
  )
  assert.equal(resultA.providerName, 'webgpu')
  assert.equal(resultA.recognitionResult.detections.length, 1)

  await prepareImportedModelSession(modelB)
  const resultB = await runDetectionOnCanvas(
    modelB,
    sourceCanvas as unknown as HTMLCanvasElement,
    'frame-b',
    undefined,
    sourceCanvas as unknown as HTMLCanvasElement,
  )
  assert.equal(resultB.providerName, 'webgpu')
  assert.equal(resultB.recognitionResult.detections.length, 1)

  await assert.rejects(
    () => prepareImportedModelSession(modelC),
    /simulated worker init failure for model 3/,
  )

  await prepareImportedModelSession(modelD)
  const resultD = await runDetectionOnCanvas(
    modelD,
    sourceCanvas as unknown as HTMLCanvasElement,
    'frame-d',
    undefined,
    sourceCanvas as unknown as HTMLCanvasElement,
  )
  assert.equal(resultD.providerName, 'webgpu')
  assert.equal(resultD.recognitionResult.detections.length, 1)

  assert.deepEqual(workerCreates, [1, 2, 3, 4])
  assert.deepEqual(workerTerminates, [1, 2, 3])
  assert.deepEqual(workerInits, [
    { workerId: 1, modelId: 1, enableGraphCapture: true },
    { workerId: 2, modelId: 2, enableGraphCapture: true },
    { workerId: 3, modelId: 3, enableGraphCapture: true },
    { workerId: 4, modelId: 4, enableGraphCapture: true },
  ])
  assert.deepEqual(workerRuns, [
    { workerId: 1, modelId: 1, inputByteLength: 3 * 640 * 640 * 4, fetchNames: ['output0'] },
    { workerId: 2, modelId: 2, inputByteLength: 3 * 640 * 640 * 4, fetchNames: ['output0'] },
    { workerId: 4, modelId: 4, inputByteLength: 3 * 640 * 640 * 4, fetchNames: ['output0'] },
  ])
  assert.deepEqual(workerReleases, [1, 2])

  await releaseCachedModelSessions()
  assert.deepEqual(workerTerminates, [1, 2, 3, 4])
  assert.deepEqual(workerReleases, [1, 2, 4])
} finally {
  await releaseCachedModelSessions().catch(() => {})
  setWebGpuRuntimeHooks(null)
  globalScope.HTMLCanvasElement = previousHTMLCanvasElement
}

console.log('frontend onnx runtime session switch tests passed')
