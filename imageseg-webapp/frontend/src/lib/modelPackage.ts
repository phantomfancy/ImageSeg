import {
  detectModelFamily,
  resolveLabelsFromMetadata,
  resolveModelContract,
  type LabelSourceKind,
  type ModelFamily,
  type PreprocessContract,
  type ResolvedModelContract,
  type TensorDescriptor,
} from '../../../contracts/src/index.ts'
import {
  inspectOnnxModelBytes,
  type WebGpuCompatibilityReport,
} from './modelIntrospection.ts'

type JsonObject = Record<string, unknown>

export const CONFIG_FILE_NAME = 'config.json'
export const PREPROCESSOR_CONFIG_FILE_NAME = 'preprocessor_config.json'

export type SidecarKind = 'config' | 'preprocessor'

export interface ModelPackageSidecars {
  configFileName?: string
  preprocessorConfigFileName?: string
}

export interface InspectedOnnxModel {
  key: string
  fileName: string
  bytes: Uint8Array
  family: ModelFamily
  inputs: TensorDescriptor[]
  outputs: TensorDescriptor[]
  embeddedLabels: Record<number, string>
  draftContract: ResolvedModelContract
  webGpuCompatibility: WebGpuCompatibilityReport
}

export interface FinalizeModelImportOptions {
  onnxModel: InspectedOnnxModel
  configFile?: File | null
  preprocessorConfigFile?: File | null
}

export interface InspectedModelPackage {
  key: string
  fileName: string
  bytes: Uint8Array
  contract: ResolvedModelContract
  sidecars: ModelPackageSidecars
  webGpuCompatibility: WebGpuCompatibilityReport
}

export interface DirectoryFileHandleLike {
  getFile(): Promise<File>
}

export interface DirectoryHandleLike {
  getFileHandle(name: string): Promise<DirectoryFileHandleLike>
}

export interface DiscoveredModelSidecars {
  configFile?: File
  preprocessorConfigFile?: File
}

export async function inspectOnnxModelFile(file: File): Promise<InspectedOnnxModel> {
  validateOnnxFile(file)

  const bytes = new Uint8Array(await file.arrayBuffer())
  const parsedModel = await inspectOnnxModelBytes(bytes)
  const family = detectModelFamily(parsedModel.inputs, parsedModel.outputs)
  const embeddedLabels = parsedModel.labels
  const labelSource: LabelSourceKind =
    Object.keys(embeddedLabels).length > 0 ? 'embedded-metadata' : 'fallback-class-index'
  const warnings = family === 'hf-detr-like'
    ? [`检测到 Hugging Face 风格模型，必须导入 ${CONFIG_FILE_NAME}；${PREPROCESSOR_CONFIG_FILE_NAME} 可选。`]
    : []
  warnings.push(...toWebGpuCompatibilityWarnings(parsedModel.webGpuCompatibility))

  return {
    key: buildModelKey(file),
    fileName: file.name,
    bytes,
    family,
    inputs: parsedModel.inputs,
    outputs: parsedModel.outputs,
    embeddedLabels,
    draftContract: resolveModelContract({
      inputs: parsedModel.inputs,
      outputs: parsedModel.outputs,
      displayName: file.name,
      labels: embeddedLabels,
      labelSource,
      warnings,
    }),
    webGpuCompatibility: parsedModel.webGpuCompatibility,
  }
}

export async function finalizeModelImport(options: FinalizeModelImportOptions): Promise<InspectedModelPackage> {
  const { onnxModel } = options
  const warnings: string[] = toWebGpuCompatibilityWarnings(onnxModel.webGpuCompatibility)
  let labels = onnxModel.embeddedLabels
  let labelSource: LabelSourceKind =
    Object.keys(labels).length > 0 ? 'embedded-metadata' : 'fallback-class-index'
  let preprocess: Partial<PreprocessContract> | undefined
  let configFileName: string | undefined
  let preprocessorConfigFileName: string | undefined

  if (onnxModel.family === 'hf-detr-like') {
    const configFile = requireSidecarFile(options.configFile ?? null, 'config')
    const config = await readJsonFile(configFile)
    const sidecarLabels = resolveLabelsFromMetadata(config)
    configFileName = configFile.name

    if (Object.keys(sidecarLabels).length > 0) {
      labels = sidecarLabels
      labelSource = 'sidecar-manifest'
    } else {
      warnings.push('config.json 未提供 id2label，将继续使用模型内嵌标签或 class-N 占位标签。')
    }

    if (options.preprocessorConfigFile) {
      const preprocessorConfigFile = validateSidecarFile(options.preprocessorConfigFile, 'preprocessor')
      preprocess = toPreprocessOverrides(await readJsonFile(preprocessorConfigFile))
      preprocessorConfigFileName = preprocessorConfigFile.name
    } else {
      warnings.push('未提供 preprocessor_config.json，将使用默认预处理参数。')
    }
  } else {
    if (options.configFile) {
      throw new Error(`${onnxModel.family} 模型不需要导入 ${CONFIG_FILE_NAME}。`)
    }

    if (options.preprocessorConfigFile) {
      throw new Error(`${onnxModel.family} 模型不需要导入 ${PREPROCESSOR_CONFIG_FILE_NAME}。`)
    }
  }

  return {
    key: onnxModel.key,
    fileName: onnxModel.fileName,
    bytes: onnxModel.bytes,
    contract: resolveModelContract({
      inputs: onnxModel.inputs,
      outputs: onnxModel.outputs,
      displayName: onnxModel.fileName,
      labels,
      labelSource,
      preprocess,
      warnings,
    }),
    sidecars: {
      configFileName,
      preprocessorConfigFileName,
    },
    webGpuCompatibility: onnxModel.webGpuCompatibility,
  }
}

export async function inspectModelPackageFiles(files: readonly File[]): Promise<InspectedModelPackage> {
  const normalizedFiles = [...files]
  const onnxFiles = normalizedFiles.filter((file) => file.name.toLowerCase().endsWith('.onnx'))

  if (onnxFiles.length === 0) {
    throw new Error('尚未选择 ONNX 模型文件。')
  }

  if (onnxFiles.length > 1) {
    throw new Error('一次只能导入一个 ONNX 模型，请移除多余的 .onnx 文件。')
  }

  const onnxModel = await inspectOnnxModelFile(onnxFiles[0])

  return finalizeModelImport({
    onnxModel,
    configFile: findFileByName(normalizedFiles, CONFIG_FILE_NAME),
    preprocessorConfigFile: findFileByName(normalizedFiles, PREPROCESSOR_CONFIG_FILE_NAME),
  })
}

export async function discoverHfSidecarsFromDirectory(
  directoryHandle: DirectoryHandleLike,
): Promise<DiscoveredModelSidecars> {
  const configFile = await tryGetFileFromDirectory(directoryHandle, CONFIG_FILE_NAME)
  const preprocessorConfigFile = await tryGetFileFromDirectory(directoryHandle, PREPROCESSOR_CONFIG_FILE_NAME)

  if (configFile) {
    validateSidecarFile(configFile, 'config')
  }

  if (preprocessorConfigFile) {
    validateSidecarFile(preprocessorConfigFile, 'preprocessor')
  }

  return {
    configFile: configFile ?? undefined,
    preprocessorConfigFile: preprocessorConfigFile ?? undefined,
  }
}

export function validateOnnxFile(file: File): File {
  if (!file.name.toLowerCase().endsWith('.onnx')) {
    throw new Error('ONNX 模型导入项只能选择 .onnx 文件。')
  }

  return file
}

export function validateSidecarFile(file: File, kind: SidecarKind): File {
  if (!file.name.toLowerCase().endsWith('.json')) {
    throw new Error(`${expectedFileName(kind)} 导入项只能选择 .json 文件。`)
  }

  if (file.name !== expectedFileName(kind)) {
    throw new Error(`${expectedFileName(kind)} 导入项必须使用文件名 ${expectedFileName(kind)}。`)
  }

  return file
}

function expectedFileName(kind: SidecarKind): string {
  return kind === 'config' ? CONFIG_FILE_NAME : PREPROCESSOR_CONFIG_FILE_NAME
}

function requireSidecarFile(file: File | null, kind: SidecarKind): File {
  if (!file) {
    throw new Error(`Hugging Face 风格模型缺少必选 ${expectedFileName(kind)}，请先导入该文件。`)
  }

  return validateSidecarFile(file, kind)
}

function findFileByName(files: readonly File[], fileName: string): File | undefined {
  return files.find((file) => file.name === fileName)
}

function buildModelKey(file: File): string {
  return `${file.name}:${file.size}:${file.lastModified}`
}

async function tryGetFileFromDirectory(
  directoryHandle: DirectoryHandleLike,
  fileName: string,
): Promise<File | null> {
  try {
    const fileHandle = await directoryHandle.getFileHandle(fileName)
    return await fileHandle.getFile()
  } catch (error) {
    if (isNotFoundError(error)) {
      return null
    }

    throw error
  }
}

function isNotFoundError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === 'NotFoundError'
    : Boolean(
      error &&
      typeof error === 'object' &&
      'name' in error &&
      error.name === 'NotFoundError',
    )
}

async function readJsonFile(file: File): Promise<JsonObject> {
  const text = await file.text()
  const parsed = JSON.parse(text) as unknown
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${file.name} 不是有效的 JSON 对象。`)
  }

  return parsed as JsonObject
}

function toPreprocessOverrides(config: JsonObject): Partial<PreprocessContract> {
  const padSize = toSize(config.pad_size)
  const size = toSize(config.size)
  const selectedSize = config.do_pad === true && padSize ? padSize : size ?? padSize
  const preprocess: Partial<PreprocessContract> = {}

  if (selectedSize) {
    preprocess.imageWidth = selectedSize.width
    preprocess.imageHeight = selectedSize.height
  }

  preprocess.resizeMode = config.do_pad === true && padSize ? 'pad' : 'stretch'

  if (config.do_rescale === true) {
    preprocess.normalization = 'zero-to-one'
  }

  return preprocess
}

function toSize(value: unknown): { width: number; height: number } | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return undefined
  }

  const width = 'width' in value && typeof value.width === 'number' ? value.width : undefined
  const height = 'height' in value && typeof value.height === 'number' ? value.height : undefined
  if (typeof width !== 'number' || typeof height !== 'number') {
    return undefined
  }

  return { width, height }
}

function toWebGpuCompatibilityWarnings(report: WebGpuCompatibilityReport): string[] {
  return report.issues.map((item) => item.message)
}
