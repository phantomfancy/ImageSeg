import onnxProtoSource from './onnxProtoSource.generated.ts'

import {
  resolveLabelsFromMetadata,
  type TensorDimension,
  type TensorDescriptor,
} from '../../../contracts/src/index.ts'

type DecodedModel = {
  graph?: {
    input?: Array<unknown>
    output?: Array<unknown>
    valueInfo?: Array<unknown>
    initializer?: Array<TensorInfo>
    node?: Array<NodeInfo>
  }
  metadataProps?: Array<{ key?: string | null; value?: string | null }>
}

type ValueInfo = {
  name?: string | null
  type?: {
    tensorType?: {
      elemType?: number | LongLike | null
      shape?: {
        dim?: Array<{
          dimValue?: number | LongLike | null
          dimParam?: string | null
        }>
      }
    }
  }
}

type TensorInfo = {
  name?: string | null
  dataType?: number | LongLike | null
}

type NodeInfo = {
  name?: string | null
  opType?: string | null
  input?: Array<string | null>
  output?: Array<string | null>
  attribute?: Array<AttributeInfo>
}

type AttributeInfo = {
  name?: string | null
  i?: number | LongLike | null
}

type LongLike = {
  toNumber?: () => number
  low?: number
}

export interface ParsedOnnxModel {
  inputs: TensorDescriptor[]
  outputs: TensorDescriptor[]
  labels: Record<number, string>
  webGpuCompatibility: WebGpuCompatibilityReport
}

export interface WebGpuCompatibilityIssue {
  severity: 'warning' | 'error'
  code: string
  message: string
  nodeName?: string
  opType?: string
  tensorName?: string
}

export interface WebGpuCompatibilityReport {
  supported: boolean
  issues: WebGpuCompatibilityIssue[]
}

let cachedOnnxProtoPromise:
  | Promise<{ onnx?: { ModelProto: { decode: (input: Uint8Array) => DecodedModel } } }>
  | null
  = null

export async function inspectOnnxModelBytes(bytes: Uint8Array): Promise<ParsedOnnxModel> {
  const model = await decodeModel(bytes)
  const initializerNames = new Set(
    (model.graph?.initializer ?? [])
      .map((item) => item.name ?? '')
      .filter(Boolean),
  )

  const inputs = (model.graph?.input ?? [])
    .map((item) => toTensorDescriptor(item as ValueInfo))
    .filter((item) => item.name.length > 0 && !initializerNames.has(item.name))

  const outputs = (model.graph?.output ?? [])
    .map((item) => toTensorDescriptor(item as ValueInfo))
    .filter((item) => item.name.length > 0)

  const labels = resolveLabelsFromMetadata(
    Object.fromEntries(
      (model.metadataProps ?? [])
        .filter((item) => item.key && item.value)
        .map((item) => [item.key!, item.value!]),
    ),
  )

  return {
    inputs,
    outputs,
    labels,
    webGpuCompatibility: analyzeWebGpuCompatibility(model),
  }
}

function analyzeWebGpuCompatibility(model: DecodedModel): WebGpuCompatibilityReport {
  const issues: WebGpuCompatibilityIssue[] = []
  const graph = model.graph
  if (!graph) {
    return {
      supported: true,
      issues,
    }
  }

  const tensorTypes = new Map<string, string>()
  for (const item of [...(graph.input ?? []), ...(graph.output ?? []), ...(graph.valueInfo ?? [])]) {
    const valueInfo = item as ValueInfo
    const name = valueInfo.name?.trim()
    const elementType = toTensorElementTypeName(valueInfo.type?.tensorType?.elemType)
    if (name && elementType) {
      tensorTypes.set(name, elementType)
    }
  }

  for (const initializer of graph.initializer ?? []) {
    const name = initializer.name?.trim()
    const elementType = toTensorElementTypeName(initializer.dataType)
    if (name && elementType) {
      tensorTypes.set(name, elementType)
    }
  }

  for (const node of graph.node ?? []) {
    if ((node.opType ?? '').toLowerCase() !== 'cast') {
      continue
    }

    const nodeName = node.name?.trim() || undefined
    const inputNames = normalizeTensorNames(node.input)
    const outputNames = normalizeTensorNames(node.output)
    const attributeTargetType = toTensorElementTypeName(readAttributeInt(node.attribute, 'to'))
    const involvedTensorTypes = [
      ...inputNames.map((name) => tensorTypes.get(name)),
      ...outputNames.map((name) => tensorTypes.get(name)),
      attributeTargetType,
    ].filter((item): item is string => Boolean(item))

    if (!involvedTensorTypes.includes('int64')) {
      continue
    }

    const tensorName = outputNames[0] ?? inputNames[0]
    issues.push({
      severity: 'error',
      code: 'webgpu-cast-int64-unsupported',
      message:
        `检测到 Cast 节点 ${nodeName ? `"${nodeName}" ` : ''}` +
        `涉及 int64 张量${tensorName ? `（${tensorName}）` : ''}。` +
        'ORT WebGPU 当前不支持该类型组合，模型无法在当前前端执行。',
      nodeName,
      opType: node.opType ?? undefined,
      tensorName,
    })
  }

  return {
    supported: !issues.some((item) => item.severity === 'error'),
    issues,
  }
}

async function decodeModel(bytes: Uint8Array): Promise<DecodedModel> {
  const onnxProto = await getOnnxProto()
  const root = onnxProto.onnx
  if (!root?.ModelProto) {
    throw new Error('无法加载 ONNX protobuf 解析器。')
  }

  return root.ModelProto.decode(bytes)
}

async function getOnnxProto(): Promise<{ onnx?: { ModelProto: { decode: (input: Uint8Array) => DecodedModel } } }> {
  if (cachedOnnxProtoPromise) {
    return cachedOnnxProtoPromise
  }

  cachedOnnxProtoPromise = loadOnnxProto()
  return cachedOnnxProtoPromise
}

function toTensorDescriptor(valueInfo: ValueInfo): TensorDescriptor {
  const dimensions = valueInfo.type?.tensorType?.shape?.dim?.map((item) => {
    const dimValue = toTensorDimension(item.dimValue, item.dimParam)
    return typeof dimValue === 'number' && Number.isFinite(dimValue)
      ? dimValue
      : typeof dimValue === 'string' && dimValue.trim()
        ? dimValue
        : null
  }) ?? []

  return {
    name: valueInfo.name ?? '',
    dimensions,
  }
}

function toTensorElementTypeName(value: number | LongLike | null | undefined): string | null {
  const elementType = toInteger(value)
  switch (elementType) {
    case 1:
      return 'float32'
    case 6:
      return 'int32'
    case 7:
      return 'int64'
    case 9:
      return 'bool'
    case 10:
      return 'float16'
    case 12:
      return 'uint32'
    default:
      return null
  }
}

function toTensorDimension(
  value: number | LongLike | null | undefined,
  dimParam: string | null | undefined,
): TensorDimension {
  const numericValue = toInteger(value)
  if (typeof numericValue === 'number') {
    return numericValue
  }

  if (typeof dimParam === 'string' && dimParam.trim()) {
    return dimParam
  }

  return null
}

function toInteger(value: number | LongLike | null | undefined): number | null {
  if (typeof value === 'number') {
    return value
  }

  if (value && typeof value.toNumber === 'function') {
    return value.toNumber()
  }

  if (value && typeof value.low === 'number') {
    return value.low
  }

  return null
}

function normalizeTensorNames(values: Array<string | null> | null | undefined): string[] {
  return (values ?? [])
    .map((item) => item?.trim() ?? '')
    .filter(Boolean)
}

function readAttributeInt(
  attributes: Array<AttributeInfo> | null | undefined,
  name: string,
): number | null {
  const attribute = (attributes ?? []).find((item) => item.name === name)
  return toInteger(attribute?.i)
}

async function loadOnnxProto(): Promise<{ onnx?: { ModelProto: { decode: (input: Uint8Array) => DecodedModel } } }> {
  const protobufModule = await import('protobufjs/minimal.js')
  const protobuf = (
    'default' in protobufModule
      ? protobufModule.default
      : protobufModule
  ) as Record<string, unknown>
  const module = { exports: {} as unknown }
  const exports = module.exports
  const require = (id: string): unknown => {
    if (id === 'protobufjs/minimal') {
      return protobuf
    }

    throw new Error(`不支持的 ONNX 解析器依赖: ${id}`)
  }

  const factory = new Function(
    'require',
    'module',
    'exports',
    `${onnxProtoSource}\nreturn module.exports;`,
  ) as (
    require: (id: string) => unknown,
    module: { exports: unknown },
    exports: unknown,
  ) => unknown

  return factory(require, module, exports) as {
    onnx?: { ModelProto: { decode: (input: Uint8Array) => DecodedModel } }
  }
}
