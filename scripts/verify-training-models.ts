import assert from 'node:assert/strict'
import { readFile, readdir } from 'node:fs/promises'
import path from 'node:path'

import {
  finalizeModelImport,
  inspectOnnxModelFile,
} from '../web-app/frontend/src/lib/modelPackage.ts'

const trainingResultDirectory = path.resolve('pytorch-training', 'training_result')
const cases = [
  {
    onnx: 'yolov8n__himars_style_ultralytics.onnx',
    expected: {
      family: 'ultralytics-yolo-detect',
      webGpuSupported: true,
      labels: { 0: 'himars' },
      labelSource: 'embedded-metadata',
      inputs: [{ name: 'images', dimensions: [1, 3, 640, 640] }],
      outputs: [{ name: 'output0', dimensions: [1, 5, 8400] }],
    },
  },
  {
    onnx: 'rtdetrv2_equipments_style_ultralytics.onnx',
    expected: {
      family: 'ultralytics-rtdetr',
      webGpuSupported: true,
      labels: { 0: 'Aircraft', 1: 'heavy armour', 2: 'light armour', 3: 'uav' },
      labelSource: 'embedded-metadata',
      inputs: [{ name: 'images', dimensions: [1, 3, 640, 640] }],
      outputs: [{ name: 'output0', dimensions: [1, 300, 8] }],
    },
  },
  {
    onnx: 'rtdetrv2_himars_style_hugginface/model_fp16.onnx',
    config: 'rtdetrv2_himars_style_hugginface/config.json',
    preprocessor: 'rtdetrv2_himars_style_hugginface/preprocessor_config.json',
    expected: {
      family: 'hf-detr-like',
      webGpuSupported: false,
      webGpuIssueCode: 'webgpu-cast-int64-unsupported',
      labels: { 0: 'himars' },
      labelSource: 'sidecar-manifest',
      inputs: [{ name: 'pixel_values', dimensions: ['batch_size', 3, 'height', 'width'] }],
      outputs: [
        { name: 'logits', dimensions: ['batch_size', 'num_queries', 1] },
        { name: 'pred_boxes', dimensions: ['batch_size', 'num_queries', 'Gatherpred_boxes_dim_2'] },
      ],
    },
  },
  {
    onnx: 'rtdetrv2_original_style_hugginface/model_fp16.onnx',
    config: 'rtdetrv2_original_style_hugginface/config.json',
    preprocessor: 'rtdetrv2_original_style_hugginface/preprocessor_config.json',
    expected: {
      family: 'hf-detr-like',
      webGpuSupported: false,
      webGpuIssueCode: 'webgpu-cast-int64-unsupported',
      labels: { 0: 'person', 79: 'toothbrush' },
      labelSource: 'sidecar-manifest',
      inputs: [{ name: 'pixel_values', dimensions: ['batch_size', 3, 'height', 'width'] }],
      outputs: [
        { name: 'logits', dimensions: ['batch_size', 300, 80] },
        { name: 'pred_boxes', dimensions: ['batch_size', 300, 4] },
      ],
    },
  },
] as const

const files = await collectOnnxFiles(trainingResultDirectory)
if (files.length !== 4) {
  throw new Error(`未在 ${trainingResultDirectory} 下找到完整的 ONNX 模型集。`)
}

for (const item of cases) {
  const onnxModel = await inspectOnnxModelFile(
    await readAsFile(path.join(trainingResultDirectory, item.onnx)),
  )
  const model = await finalizeModelImport({
    onnxModel,
    configFile: item.config
      ? await readAsFile(path.join(trainingResultDirectory, item.config))
      : undefined,
    preprocessorConfigFile: item.preprocessor
      ? await readAsFile(path.join(trainingResultDirectory, item.preprocessor))
      : undefined,
  })

  assert.equal(model.contract.family, item.expected.family)
  assert.equal(model.contract.labelSource, item.expected.labelSource)
  assert.equal(onnxModel.webGpuCompatibility.supported, item.expected.webGpuSupported)
  assert.equal(model.webGpuCompatibility.supported, item.expected.webGpuSupported)

  if (item.expected.webGpuIssueCode) {
    assert.equal(
      onnxModel.webGpuCompatibility.issues.some((issue) => issue.code === item.expected.webGpuIssueCode),
      true,
    )
  }

  for (const [labelId, label] of Object.entries(item.expected.labels)) {
    assert.equal(model.contract.labels[Number(labelId)], label)
  }

  assert.deepEqual(model.contract.inputs, item.expected.inputs)
  assert.deepEqual(model.contract.outputs, item.expected.outputs)

  console.log(`${item.onnx} -> ${model.contract.family} (webgpu=${model.webGpuCompatibility.supported ? 'ok' : 'blocked'})`)
}

console.log(`verified ${cases.length} training_result model(s)`)

async function collectOnnxFiles(directory: string): Promise<string[]> {
  const items = await readdir(directory, { withFileTypes: true })
  const files: string[] = []

  for (const item of items) {
    const fullPath = path.join(directory, item.name)
    if (item.isDirectory()) {
      files.push(...await collectOnnxFiles(fullPath))
      continue
    }

    if (item.isFile() && item.name.toLowerCase().endsWith('.onnx')) {
      files.push(fullPath)
    }
  }

  return files.sort()
}

async function readAsFile(filePath: string): Promise<File> {
  const buffer = await readFile(filePath)
  const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)

  return new File(
    [new Uint8Array(arrayBuffer)],
    path.basename(filePath),
    { lastModified: 0 },
  )
}
