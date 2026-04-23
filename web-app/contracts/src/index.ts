export type ModelFamily = 'hf-detr-like' | 'ultralytics-yolo-detect' | 'ultralytics-rtdetr'

export type OutputLayoutKind = 'logits-and-pred-boxes' | 'ultralytics-anchors' | 'ultralytics-queries'

export type LabelSourceKind = 'sidecar-manifest' | 'embedded-metadata' | 'fallback-class-index'

export type TensorDimension = number | string | null

export interface TensorDescriptor {
  name: string
  dimensions: Array<TensorDimension>
}

export interface RuntimeValueMetadata {
  name: string
  isTensor?: boolean
  shape?: readonly unknown[]
  dimensions?: readonly unknown[]
}

export interface RuntimeCompatibility {
  supportsOnnxRuntime: boolean
  supportsOnnxRuntimeWeb: boolean
  preferredExecutionProviders: string[]
}

export interface PreprocessContract {
  inputTensorName: string
  imageWidth: number
  imageHeight: number
  layout: 'nchw'
  channelOrder: 'rgb'
  resizeMode: 'stretch' | 'pad'
  normalization: 'zero-to-one'
}

export interface DecoderContract {
  layoutKind: OutputLayoutKind
  outputTensorNames: string[]
  scoreThreshold: number
  maxDetections: number
  nmsIouThreshold: number | null
}

export interface ModelManifest {
  schemaVersion: string
  displayName: string
  family: ModelFamily
  runtimeHints: RuntimeCompatibility
  preprocess: PreprocessContract
  decoder: DecoderContract
  labels: Record<number, string>
}

export interface ResolvedModelContract {
  displayName: string
  family: ModelFamily
  runtimeHints: RuntimeCompatibility
  preprocess: PreprocessContract
  decoder: DecoderContract
  labels: Record<number, string>
  labelSource: LabelSourceKind
  warnings: string[]
  inputs: TensorDescriptor[]
  outputs: TensorDescriptor[]
}

export interface DetectionBox {
  x: number
  y: number
  width: number
  height: number
}

export interface Detection {
  label: string
  confidence: number
  box: DetectionBox
}

export interface RecognitionResult {
  inputSource: string
  modelVersion: string
  detectedAtUtc: string
  detections: Detection[]
}

export interface TensorLike {
  dims: readonly number[]
  data: ArrayLike<number>
}

export interface PreprocessedImageGeometry {
  sourceWidth: number
  sourceHeight: number
  targetWidth: number
  targetHeight: number
  resizeMode: PreprocessContract['resizeMode']
  scaleX: number
  scaleY: number
  padLeft: number
  padTop: number
  contentWidth: number
  contentHeight: number
}

export type MetadataEntries = Map<string, unknown> | Record<string, unknown> | null | undefined

const defaultRuntimeCompatibility: RuntimeCompatibility = {
  supportsOnnxRuntime: true,
  supportsOnnxRuntimeWeb: true,
  preferredExecutionProviders: ['webgpu'],
}

export const knownModelContracts: Record<ModelFamily, ModelManifest> = {
  'hf-detr-like': {
    schemaVersion: '2026-04-19',
    displayName: 'Hugging Face / DETR-like ONNX',
    family: 'hf-detr-like',
    runtimeHints: defaultRuntimeCompatibility,
    preprocess: {
      inputTensorName: 'pixel_values',
      imageWidth: 640,
      imageHeight: 640,
      layout: 'nchw',
      channelOrder: 'rgb',
      resizeMode: 'stretch',
      normalization: 'zero-to-one',
    },
    decoder: {
      layoutKind: 'logits-and-pred-boxes',
      outputTensorNames: ['logits', 'pred_boxes'],
      scoreThreshold: 0.8,
      maxDetections: 20,
      nmsIouThreshold: null,
    },
    labels: {},
  },
  'ultralytics-yolo-detect': {
    schemaVersion: '2026-04-19',
    displayName: 'Ultralytics YOLO Detect ONNX',
    family: 'ultralytics-yolo-detect',
    runtimeHints: defaultRuntimeCompatibility,
    preprocess: {
      inputTensorName: 'images',
      imageWidth: 640,
      imageHeight: 640,
      layout: 'nchw',
      channelOrder: 'rgb',
      resizeMode: 'stretch',
      normalization: 'zero-to-one',
    },
    decoder: {
      layoutKind: 'ultralytics-anchors',
      outputTensorNames: ['output0'],
      scoreThreshold: 0.35,
      maxDetections: 20,
      nmsIouThreshold: 0.45,
    },
    labels: {},
  },
  'ultralytics-rtdetr': {
    schemaVersion: '2026-04-19',
    displayName: 'Ultralytics RT-DETR ONNX',
    family: 'ultralytics-rtdetr',
    runtimeHints: defaultRuntimeCompatibility,
    preprocess: {
      inputTensorName: 'images',
      imageWidth: 640,
      imageHeight: 640,
      layout: 'nchw',
      channelOrder: 'rgb',
      resizeMode: 'stretch',
      normalization: 'zero-to-one',
    },
    decoder: {
      layoutKind: 'ultralytics-queries',
      outputTensorNames: ['output0'],
      scoreThreshold: 0.35,
      maxDetections: 20,
      nmsIouThreshold: null,
    },
    labels: {},
  },
}

export function detectModelFamily(inputs: TensorDescriptor[], outputs: TensorDescriptor[]): ModelFamily {
  if (outputs.some((item) => item.name.toLowerCase() === 'logits') &&
      outputs.some((item) => item.name.toLowerCase() === 'pred_boxes')) {
    return 'hf-detr-like'
  }

  if (outputs.length !== 1 || outputs[0].dimensions.length !== 3) {
    throw new Error('无法根据模型输入输出签名识别模型格式。')
  }

  const output = outputs[0]
  const [, dim1, dim2] = output.dimensions
  const isImagesInput = inputs.some((item) => item.name.toLowerCase() === 'images')
  const isOutput0 = output.name.toLowerCase() === 'output0'

  if (isImagesInput && isOutput0) {
    if (typeof dim1 === 'number' && typeof dim2 === 'number') {
      if (dim1 >= 64 && dim2 <= 16) {
        return 'ultralytics-rtdetr'
      }

      if (dim2 >= 64 && dim1 <= 16) {
        return 'ultralytics-yolo-detect'
      }

      return dim1 > dim2 ? 'ultralytics-rtdetr' : 'ultralytics-yolo-detect'
    }

    if (typeof dim1 === 'number' && dim1 >= 64) {
      return 'ultralytics-rtdetr'
    }

    return 'ultralytics-yolo-detect'
  }

  throw new Error('无法根据模型输入输出签名识别模型格式。')
}

export function toTensorDescriptors(metadata: readonly RuntimeValueMetadata[]): TensorDescriptor[] {
  return metadata.map((item) => ({
    name: item.name,
    dimensions: normalizeDimensions(item),
  }))
}

export function resolveLabelsFromMetadata(entries: MetadataEntries): Record<number, string> {
  if (!entries) {
    return {}
  }

  const normalized = entries instanceof Map
    ? Object.fromEntries(entries.entries())
    : entries

  const candidates = [
    normalized.labels,
    normalized.names,
    normalized.id2label,
  ]

  for (const candidate of candidates) {
    const parsed = toLabelRecord(candidate)
    if (Object.keys(parsed).length > 0) {
      return parsed
    }
  }

  return {}
}

export function resolveModelContract(options: {
  inputs: TensorDescriptor[]
  outputs: TensorDescriptor[]
  manifest?: ModelManifest
  displayName?: string
  labels?: Record<number, string>
  labelSource?: LabelSourceKind
  preprocess?: Partial<PreprocessContract>
  warnings?: string[]
}): ResolvedModelContract {
  const family = options.manifest?.family ?? detectModelFamily(options.inputs, options.outputs)
  const defaults = knownModelContracts[family]
  const input = options.inputs[0]
  const inputWidth = input?.dimensions.at(-1)
  const inputHeight = input?.dimensions.at(-2)
  const warnings = [...(options.warnings ?? [])]
  const labels = options.labels ?? options.manifest?.labels ?? defaults.labels
  const labelSource: LabelSourceKind =
    options.labelSource ??
    (options.manifest ? 'sidecar-manifest' :
      options.labels ? 'embedded-metadata' :
        'fallback-class-index')
  const preprocess = options.preprocess ?? {}

  if (Object.keys(labels).length === 0) {
    warnings.push('模型未提供类别标签，将使用 class-N 占位标签。')
  }

  return {
    displayName: options.displayName?.trim() || options.manifest?.displayName || defaults.displayName,
    family,
    runtimeHints: options.manifest?.runtimeHints ?? defaults.runtimeHints,
    preprocess: {
      ...(options.manifest?.preprocess ?? defaults.preprocess),
      inputTensorName: preprocess.inputTensorName ?? options.manifest?.preprocess.inputTensorName ?? input?.name ?? defaults.preprocess.inputTensorName,
      imageWidth: typeof preprocess.imageWidth === 'number'
        ? preprocess.imageWidth
        : typeof inputWidth === 'number'
          ? inputWidth
          : (options.manifest?.preprocess.imageWidth ?? defaults.preprocess.imageWidth),
      imageHeight: typeof preprocess.imageHeight === 'number'
        ? preprocess.imageHeight
        : typeof inputHeight === 'number'
          ? inputHeight
          : (options.manifest?.preprocess.imageHeight ?? defaults.preprocess.imageHeight),
      resizeMode: preprocess.resizeMode ?? options.manifest?.preprocess.resizeMode ?? defaults.preprocess.resizeMode,
      normalization: preprocess.normalization ?? options.manifest?.preprocess.normalization ?? defaults.preprocess.normalization,
    },
    decoder: {
      ...(options.manifest?.decoder ?? defaults.decoder),
      outputTensorNames: options.manifest?.decoder.outputTensorNames?.length
        ? options.manifest.decoder.outputTensorNames
        : options.outputs.map((item) => item.name),
      nmsIouThreshold:
        options.manifest?.decoder.nmsIouThreshold ??
        defaults.decoder.nmsIouThreshold,
    },
    labels,
    labelSource,
    warnings,
    inputs: options.inputs,
    outputs: options.outputs,
  }
}

export function resolveLabel(contract: ResolvedModelContract, classId: number): string {
  return contract.labels[classId] ?? `class-${classId}`
}

export function normalizeScore(value: number): number {
  return value >= 0 && value <= 1 ? value : sigmoid(value)
}

export function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value))
}

export function decodeDetections(
  contract: ResolvedModelContract,
  outputs: Record<string, TensorLike>,
  geometry: PreprocessedImageGeometry,
): Detection[] {
  switch (contract.decoder.layoutKind) {
    case 'logits-and-pred-boxes':
      return decodeHfDetrLike(contract, outputs, geometry)
    case 'ultralytics-anchors':
      return decodeUltralyticsAnchors(contract, outputs, geometry)
    case 'ultralytics-queries':
      return decodeUltralyticsQueries(contract, outputs, geometry)
    default:
      throw new Error(`不支持的输出布局：${contract.decoder.layoutKind}`)
  }
}

function decodeHfDetrLike(
  contract: ResolvedModelContract,
  outputs: Record<string, TensorLike>,
  geometry: PreprocessedImageGeometry,
): Detection[] {
  const logits = resolveTensor(outputs, contract.decoder.outputTensorNames, 'logits')
  const boxes = resolveTensor(outputs, contract.decoder.outputTensorNames, 'pred_boxes')
  const queryCount = logits.dims.at(-2) ?? 0
  const classCount = logits.dims.at(-1) ?? 0
  const detections: Detection[] = []

  for (let queryIndex = 0; queryIndex < queryCount; queryIndex += 1) {
    let bestClassId = 0
    let bestScore = 0

    if (classCount === 1) {
      bestScore = sigmoid(readLogits(logits, queryIndex, 0))
    } else {
      const probabilities = softmax(logits, queryIndex, classCount)
      for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
        if (probabilities[classIndex] > bestScore) {
          bestScore = probabilities[classIndex]
          bestClassId = classIndex
        }
      }
    }

    if (bestScore < contract.decoder.scoreThreshold) {
      continue
    }

    detections.push({
      label: resolveLabel(contract, bestClassId),
      confidence: bestScore,
      box: toDetectionBox(
        readBoxes(boxes, queryIndex, 0),
        readBoxes(boxes, queryIndex, 1),
        readBoxes(boxes, queryIndex, 2),
        readBoxes(boxes, queryIndex, 3),
        geometry,
        true,
      ),
    })
  }

  return finalizeDetections(
    detections,
    contract.decoder.maxDetections,
    contract.decoder.nmsIouThreshold,
  )
}

function decodeUltralyticsAnchors(
  contract: ResolvedModelContract,
  outputs: Record<string, TensorLike>,
  geometry: PreprocessedImageGeometry,
): Detection[] {
  const output = resolveTensor(outputs, contract.decoder.outputTensorNames, 'output0')
  const channelCount = output.dims[1] ?? 0
  const anchorCount = output.dims[2] ?? 0
  const classCount = Math.max(1, channelCount - 4)
  const normalized = guessNormalized(output, anchorCount, true)
  const detections: Detection[] = []

  for (let anchorIndex = 0; anchorIndex < anchorCount; anchorIndex += 1) {
    let bestClassId = 0
    let bestScore = 0

    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      const score = normalizeScore(readAnchor(output, classIndex + 4, anchorIndex))
      if (score > bestScore) {
        bestScore = score
        bestClassId = classIndex
      }
    }

    if (bestScore < contract.decoder.scoreThreshold) {
      continue
    }

    detections.push({
      label: resolveLabel(contract, bestClassId),
      confidence: bestScore,
      box: toDetectionBox(
        readAnchor(output, 0, anchorIndex),
        readAnchor(output, 1, anchorIndex),
        readAnchor(output, 2, anchorIndex),
        readAnchor(output, 3, anchorIndex),
        geometry,
        normalized,
      ),
    })
  }

  return finalizeDetections(
    detections,
    contract.decoder.maxDetections,
    contract.decoder.nmsIouThreshold,
  )
}

function decodeUltralyticsQueries(
  contract: ResolvedModelContract,
  outputs: Record<string, TensorLike>,
  geometry: PreprocessedImageGeometry,
): Detection[] {
  const output = resolveTensor(outputs, contract.decoder.outputTensorNames, 'output0')
  const queryCount = output.dims[1] ?? 0
  const vectorLength = output.dims[2] ?? 0
  const classCount = Math.max(1, vectorLength - 4)
  const normalized = guessNormalized(output, queryCount, false)
  const detections: Detection[] = []

  for (let queryIndex = 0; queryIndex < queryCount; queryIndex += 1) {
    let bestClassId = 0
    let bestScore = 0

    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      const score = normalizeScore(readQuery(output, queryIndex, classIndex + 4))
      if (score > bestScore) {
        bestScore = score
        bestClassId = classIndex
      }
    }

    if (bestScore < contract.decoder.scoreThreshold) {
      continue
    }

    detections.push({
      label: resolveLabel(contract, bestClassId),
      confidence: bestScore,
      box: toDetectionBox(
        readQuery(output, queryIndex, 0),
        readQuery(output, queryIndex, 1),
        readQuery(output, queryIndex, 2),
        readQuery(output, queryIndex, 3),
        geometry,
        normalized,
      ),
    })
  }

  return finalizeDetections(
    detections,
    contract.decoder.maxDetections,
    contract.decoder.nmsIouThreshold,
  )
}

function resolveTensor(
  outputs: Record<string, TensorLike>,
  preferredNames: string[],
  expectedName: string,
): TensorLike {
  const normalizedExpectedName = expectedName.toLowerCase()
  const exactPreferredNames = preferredNames.filter((name) => name.toLowerCase() === normalizedExpectedName)

  for (const name of exactPreferredNames) {
    const tensor = outputs[name]
    if (tensor) {
      return tensor
    }
  }

  const exactKey = Object.keys(outputs).find((item) => item.toLowerCase() === normalizedExpectedName)
  if (exactKey) {
    return outputs[exactKey]
  }

  const partialPreferredName = preferredNames.find((name) => name.toLowerCase().includes(normalizedExpectedName))
  if (partialPreferredName && outputs[partialPreferredName]) {
    return outputs[partialPreferredName]
  }

  const partialKey = Object.keys(outputs).find((item) => item.toLowerCase().includes(normalizedExpectedName))
  if (partialKey) {
    return outputs[partialKey]
  }

  if (preferredNames.length === 1 && outputs[preferredNames[0]]) {
    return outputs[preferredNames[0]]
  }

  throw new Error(`模型输出中未找到 ${expectedName}`)
}

function readLogits(tensor: TensorLike, queryIndex: number, classIndex: number): number {
  const classCount = tensor.dims.at(-1) ?? 0
  return Number(tensor.data[queryIndex * classCount + classIndex] ?? 0)
}

function readBoxes(tensor: TensorLike, queryIndex: number, itemIndex: number): number {
  return Number(tensor.data[queryIndex * 4 + itemIndex] ?? 0)
}

function readAnchor(tensor: TensorLike, channelIndex: number, anchorIndex: number): number {
  const anchorCount = tensor.dims[2] ?? 0
  return Number(tensor.data[channelIndex * anchorCount + anchorIndex] ?? 0)
}

function readQuery(tensor: TensorLike, queryIndex: number, itemIndex: number): number {
  const vectorLength = tensor.dims[2] ?? 0
  return Number(tensor.data[queryIndex * vectorLength + itemIndex] ?? 0)
}

function softmax(tensor: TensorLike, queryIndex: number, classCount: number): number[] {
  const values = new Array<number>(classCount).fill(0)
  let maxValue = Number.NEGATIVE_INFINITY

  for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
    const current = readLogits(tensor, queryIndex, classIndex)
    if (current > maxValue) {
      maxValue = current
    }
  }

  let sum = 0
  for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
    values[classIndex] = Math.exp(readLogits(tensor, queryIndex, classIndex) - maxValue)
    sum += values[classIndex]
  }

  if (sum <= 0) {
    return values
  }

  return values.map((item) => item / sum)
}

function guessNormalized(tensor: TensorLike, itemCount: number, anchorLayout: boolean): boolean {
  let maxAbs = 0
  const scanCount = Math.min(itemCount, 8)

  for (let index = 0; index < scanCount; index += 1) {
    const values = anchorLayout
      ? [
        readAnchor(tensor, 0, index),
        readAnchor(tensor, 1, index),
        readAnchor(tensor, 2, index),
        readAnchor(tensor, 3, index),
      ]
      : [
        readQuery(tensor, index, 0),
        readQuery(tensor, index, 1),
        readQuery(tensor, index, 2),
        readQuery(tensor, index, 3),
      ]

    for (const value of values) {
      maxAbs = Math.max(maxAbs, Math.abs(value))
    }
  }

  return maxAbs <= 2
}

function toDetectionBox(
  cx: number,
  cy: number,
  width: number,
  height: number,
  geometry: PreprocessedImageGeometry,
  normalized: boolean,
): DetectionBox {
  const targetCx = normalized ? cx * geometry.targetWidth : cx
  const targetCy = normalized ? cy * geometry.targetHeight : cy
  const targetWidth = normalized ? width * geometry.targetWidth : width
  const targetHeight = normalized ? height * geometry.targetHeight : height
  const left = targetCx - targetWidth / 2
  const top = targetCy - targetHeight / 2
  const right = targetCx + targetWidth / 2
  const bottom = targetCy + targetHeight / 2
  const sourceLeft = projectModelX(left, geometry)
  const sourceTop = projectModelY(top, geometry)
  const sourceRight = projectModelX(right, geometry)
  const sourceBottom = projectModelY(bottom, geometry)

  return {
    x: sourceLeft,
    y: sourceTop,
    width: Math.max(0, sourceRight - sourceLeft),
    height: Math.max(0, sourceBottom - sourceTop),
  }
}

function projectModelX(value: number, geometry: PreprocessedImageGeometry): number {
  if (geometry.resizeMode === 'pad') {
    return clamp((value - geometry.padLeft) / geometry.scaleX, 0, geometry.sourceWidth)
  }

  return clamp(value / geometry.scaleX, 0, geometry.sourceWidth)
}

function projectModelY(value: number, geometry: PreprocessedImageGeometry): number {
  if (geometry.resizeMode === 'pad') {
    return clamp((value - geometry.padTop) / geometry.scaleY, 0, geometry.sourceHeight)
  }

  return clamp(value / geometry.scaleY, 0, geometry.sourceHeight)
}

function finalizeDetections(
  detections: Detection[],
  maxDetections: number,
  nmsIouThreshold: number | null,
): Detection[] {
  const sorted = [...detections]
    .sort((left, right) => right.confidence - left.confidence)
  const normalizedMaxDetections = Math.trunc(maxDetections)

  if (normalizedMaxDetections <= 0) {
    if (typeof nmsIouThreshold !== 'number' || nmsIouThreshold <= 0 || sorted.length <= 1) {
      return sorted
    }

    const selected: Detection[] = []
    for (const detection of sorted) {
      const overlapsExisting = selected.some((item) => calculateIoU(item.box, detection.box) >= nmsIouThreshold)
      if (!overlapsExisting) {
        selected.push(detection)
      }
    }

    return selected
  }

  if (typeof nmsIouThreshold !== 'number' || nmsIouThreshold <= 0 || sorted.length <= 1) {
    return sorted.slice(0, normalizedMaxDetections)
  }

  const selected: Detection[] = []
  for (const detection of sorted) {
    const overlapsExisting = selected.some((item) => calculateIoU(item.box, detection.box) >= nmsIouThreshold)
    if (!overlapsExisting) {
      selected.push(detection)
    }

    if (selected.length >= normalizedMaxDetections) {
      break
    }
  }

  return selected
}

function calculateIoU(left: DetectionBox, right: DetectionBox): number {
  const intersectionLeft = Math.max(left.x, right.x)
  const intersectionTop = Math.max(left.y, right.y)
  const intersectionRight = Math.min(left.x + left.width, right.x + right.width)
  const intersectionBottom = Math.min(left.y + left.height, right.y + right.height)
  const intersectionWidth = Math.max(0, intersectionRight - intersectionLeft)
  const intersectionHeight = Math.max(0, intersectionBottom - intersectionTop)
  const intersectionArea = intersectionWidth * intersectionHeight
  if (intersectionArea <= 0) {
    return 0
  }

  const leftArea = left.width * left.height
  const rightArea = right.width * right.height
  const unionArea = leftArea + rightArea - intersectionArea
  if (unionArea <= 0) {
    return 0
  }

  return intersectionArea / unionArea
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function normalizeDimensions(item: RuntimeValueMetadata): Array<TensorDimension> {
  if (item.isTensor === false) {
    return []
  }

  const rawDimensions = item.shape ?? item.dimensions ?? []
  return Array.from(rawDimensions, (dimension) =>
    typeof dimension === 'number'
      ? dimension
      : typeof dimension === 'string' && dimension.trim()
        ? dimension
        : null)
}

export function toLabelRecord(payload: unknown): Record<number, string> {
  if (payload === null || payload === undefined) {
    return {}
  }

  if (Array.isArray(payload)) {
    return Object.fromEntries(payload.map((item, index) => [index, String(item)]))
  }

  if (typeof payload === 'object') {
    return Object.fromEntries(
      Object.entries(payload)
        .filter(([key]) => !Number.isNaN(Number(key)))
        .map(([key, value]) => [Number(key), String(value)]),
    )
  }

  if (typeof payload !== 'string' || !payload.trim()) {
    return {}
  }

  const candidates = [payload.trim()]
  const pythonLike = normalizePythonLikeObjectLiteral(payload.trim())
  if (pythonLike !== payload.trim()) {
    candidates.push(pythonLike)
  }

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as unknown
      return toLabelRecord(parsed)
    } catch {
      continue
    }
  }

  return {}
}

function normalizePythonLikeObjectLiteral(value: string): string {
  return value
    .replace(/([{,]\s*)([A-Za-z0-9_]+)\s*:/g, '$1"$2":')
    .replaceAll("'", '"')
}
