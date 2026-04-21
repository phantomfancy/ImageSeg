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
  if (!model.graph) {
    return {
      supported: true,
      issues: [],
    }
  }

  return {
    supported: true,
    issues: [],
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
