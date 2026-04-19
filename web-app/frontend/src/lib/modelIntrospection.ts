import onnxProto from '../../node_modules/onnxruntime-web/lib/onnxjs/ort-schema/protobuf/onnx.js'

import {
  resolveLabelsFromMetadata,
  type TensorDescriptor,
} from '../../../contracts/src/index.ts'

type DecodedModel = {
  graph?: {
    input?: Array<unknown>
    output?: Array<unknown>
    initializer?: Array<{ name?: string | null }>
  }
  metadataProps?: Array<{ key?: string | null; value?: string | null }>
}

type ValueInfo = {
  name?: string | null
  type?: {
    tensorType?: {
      shape?: {
        dim?: Array<{
          dimValue?: number | LongLike | null
        }>
      }
    }
  }
}

type LongLike = {
  toNumber?: () => number
  low?: number
}

export interface ParsedOnnxModel {
  inputs: TensorDescriptor[]
  outputs: TensorDescriptor[]
  labels: Record<number, string>
}

export function inspectOnnxModelBytes(bytes: Uint8Array): ParsedOnnxModel {
  const model = decodeModel(bytes)
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
  }
}

function decodeModel(bytes: Uint8Array): DecodedModel {
  const root = (onnxProto as { onnx?: { ModelProto: { decode: (input: Uint8Array) => DecodedModel } } }).onnx
  if (!root?.ModelProto) {
    throw new Error('无法加载 ONNX protobuf 解析器。')
  }

  return root.ModelProto.decode(bytes)
}

function toTensorDescriptor(valueInfo: ValueInfo): TensorDescriptor {
  const dimensions = valueInfo.type?.tensorType?.shape?.dim?.map((item) => {
    const dimValue = toDimensionNumber(item.dimValue)
    return typeof dimValue === 'number' && Number.isFinite(dimValue) ? dimValue : null
  }) ?? []

  return {
    name: valueInfo.name ?? '',
    dimensions,
  }
}

function toDimensionNumber(value: number | LongLike | null | undefined): number | null {
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
