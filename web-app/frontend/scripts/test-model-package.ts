import assert from 'node:assert/strict'
import path from 'node:path'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

import { deriveModelImportControls } from '../src/lib/modelImportState.ts'
import {
  CONFIG_FILE_NAME,
  PREPROCESSOR_CONFIG_FILE_NAME,
  discoverHfSidecarsFromDirectory,
  finalizeModelImport,
  inspectOnnxModelFile,
  validateOnnxFile,
  validateSidecarFile,
} from '../src/lib/modelPackage.ts'

const scriptDirectory = fileURLToPath(new URL('.', import.meta.url))
const trainingResultDirectory = path.resolve(scriptDirectory, '..', '..', '..', 'pytorch-training', 'training_result')

class MockDirectoryHandle {
  readonly #files: Map<string, File>

  constructor(files: File[]) {
    this.#files = new Map(files.map((file) => [file.name, file]))
  }

  async getFileHandle(name: string): Promise<{ getFile(): Promise<File> }> {
    const file = this.#files.get(name)
    if (!file) {
      throw new DOMException('not found', 'NotFoundError')
    }

    return {
      getFile: async () => file,
    }
  }
}

const ultralyticsOnnx = await readAsFile(path.join(trainingResultDirectory, 'yolov8n__himars_style_ultralytics.onnx'))
const ultralyticsDraft = await inspectOnnxModelFile(ultralyticsOnnx)
assert.equal(ultralyticsDraft.family, 'ultralytics-yolo-detect')
assert.equal(ultralyticsDraft.draftContract.labelSource, 'embedded-metadata')
assert.equal(ultralyticsDraft.webGpuCompatibility.supported, true)
assert.deepEqual(deriveModelImportControls(ultralyticsDraft.family, true), {
  configEnabled: false,
  configRequired: false,
  preprocessorEnabled: false,
  autoDiscoverEnabled: false,
})

const ultralyticsModel = await finalizeModelImport({
  onnxModel: ultralyticsDraft,
})
assert.equal(ultralyticsModel.contract.family, 'ultralytics-yolo-detect')
assert.deepEqual(ultralyticsModel.contract.labels, { 0: 'himars' })
assert.equal(ultralyticsModel.webGpuCompatibility.supported, true)
assert.equal(ultralyticsModel.contract.decoder.scoreThreshold, 0.08)
assert.equal(ultralyticsModel.contract.decoder.nmsIouThreshold, 0.45)

const hfOnnx = await readAsFile(path.join(trainingResultDirectory, 'rtdetrv2_himars_style_hugginface', 'model_fp16.onnx'))
const hfConfig = await readAsFile(path.join(trainingResultDirectory, 'rtdetrv2_himars_style_hugginface', CONFIG_FILE_NAME))
const hfPreprocessor = await readAsFile(path.join(trainingResultDirectory, 'rtdetrv2_himars_style_hugginface', PREPROCESSOR_CONFIG_FILE_NAME))

const hfDraft = await inspectOnnxModelFile(hfOnnx)
assert.equal(hfDraft.family, 'hf-detr-like')
assert.equal(hfDraft.webGpuCompatibility.supported, true)
assert.equal(hfDraft.webGpuCompatibility.issues.length, 0)
assert.deepEqual(deriveModelImportControls(hfDraft.family, true), {
  configEnabled: true,
  configRequired: true,
  preprocessorEnabled: true,
  autoDiscoverEnabled: true,
})

await assert.rejects(
  async () => finalizeModelImport({
    onnxModel: hfDraft,
  }),
  /缺少必选 config\.json/,
)

const hfModel = await finalizeModelImport({
  onnxModel: hfDraft,
  configFile: hfConfig,
  preprocessorConfigFile: hfPreprocessor,
})
assert.equal(hfModel.contract.family, 'hf-detr-like')
assert.equal(hfModel.contract.labelSource, 'sidecar-manifest')
assert.deepEqual(hfModel.contract.labels, { 0: 'himars' })
assert.equal(hfModel.contract.preprocess.resizeMode, 'pad')
assert.equal(hfModel.contract.preprocess.imageWidth, 640)
assert.equal(hfModel.contract.preprocess.imageHeight, 640)
assert.equal(hfModel.webGpuCompatibility.supported, true)
assert.equal(hfModel.webGpuCompatibility.issues.length, 0)

const withoutPreprocessor = await finalizeModelImport({
  onnxModel: hfDraft,
  configFile: hfConfig,
})
assert.equal(withoutPreprocessor.sidecars.preprocessorConfigFileName, undefined)
assert.match(
  withoutPreprocessor.contract.warnings.join('\n'),
  /未提供 preprocessor_config\.json/,
)

assert.throws(
  () => validateOnnxFile(new File(['{}'], 'config.json', { lastModified: 0 })),
  /只能选择 \.onnx 文件/,
)
assert.throws(
  () => validateSidecarFile(new File(['{}'], 'wrong.json', { lastModified: 0 }), 'config'),
  /必须使用文件名 config\.json/,
)
assert.throws(
  () => validateSidecarFile(new File(['{}'], 'preprocessor_config.txt', { lastModified: 0 }), 'preprocessor'),
  /只能选择 \.json 文件/,
)

const discovered = await discoverHfSidecarsFromDirectory(
  new MockDirectoryHandle([
    hfConfig,
    hfPreprocessor,
  ]),
)
assert.equal(discovered.configFile?.name, CONFIG_FILE_NAME)
assert.equal(discovered.preprocessorConfigFile?.name, PREPROCESSOR_CONFIG_FILE_NAME)

const missingConfigDiscovery = await discoverHfSidecarsFromDirectory(
  new MockDirectoryHandle([hfPreprocessor]),
)
assert.equal(missingConfigDiscovery.configFile, undefined)
assert.equal(missingConfigDiscovery.preprocessorConfigFile?.name, PREPROCESSOR_CONFIG_FILE_NAME)

console.log('frontend model package tests passed')

async function readAsFile(filePath: string): Promise<File> {
  return new File(
    [await readFile(filePath)],
    path.basename(filePath),
    { lastModified: 0 },
  )
}
