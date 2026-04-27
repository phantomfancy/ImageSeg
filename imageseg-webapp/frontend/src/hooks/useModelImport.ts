import { startTransition, useEffect, useState } from 'react'
import {
  CONFIG_FILE_NAME,
  PREPROCESSOR_CONFIG_FILE_NAME,
  discoverHfSidecarsFromDirectory,
  inspectOnnxModelFile,
  validateSidecarFile,
  type InspectedOnnxModel,
} from '../lib/modelPackage'
import type { ImportedModel } from '../lib/onnxRuntime'
import {
  inspectImportedModel,
  prepareImportedModelSession,
  releaseCachedModelSessions,
} from '../lib/onnxRuntime'
import {
  buildPendingHfMessage,
  formatAutoDiscoveredMessage,
  formatError,
  formatModelReadyMessage,
  isAbortError,
} from '../app/workspaceUtils'

type UseModelImportArgs = {
  canChangeModel: boolean
  clearRecognitionOutputs: () => void
  resetDetectionThreshold: (nextValue?: number) => void
  resetMaxDetections: (nextValue?: number) => void
  setStatusMessage: (message: string) => void
  stopActiveStream: () => void
  supportsDirectoryPicker: boolean
  webGpuSupportState: { supported: boolean; message?: string }
}

export function useModelImport(args: UseModelImportArgs) {
  const {
    canChangeModel,
    clearRecognitionOutputs,
    resetDetectionThreshold,
    resetMaxDetections,
    setStatusMessage,
    stopActiveStream,
    supportsDirectoryPicker,
    webGpuSupportState,
  } = args

  const [onnxModelDraft, setOnnxModelDraft] = useState<InspectedOnnxModel | null>(null)
  const [configFile, setConfigFile] = useState<File | null>(null)
  const [preprocessorConfigFile, setPreprocessorConfigFile] = useState<File | null>(null)
  const [importedModel, setImportedModel] = useState<ImportedModel | null>(null)
  const [modelBusy, setModelBusy] = useState(false)
  const [discoverBusy, setDiscoverBusy] = useState(false)
  const [onnxInputKey, setOnnxInputKey] = useState(0)
  const [configInputKey, setConfigInputKey] = useState(0)
  const [preprocessorInputKey, setPreprocessorInputKey] = useState(0)

  useEffect(() => () => {
    void releaseCachedModelSessions()
  }, [])

  function clearImportedModelState() {
    setImportedModel(null)
    resetDetectionThreshold()
    resetMaxDetections()
    clearRecognitionOutputs()
  }

  function resetSidecarSelections() {
    setConfigFile(null)
    setPreprocessorConfigFile(null)
    setConfigInputKey((value) => value + 1)
    setPreprocessorInputKey((value) => value + 1)
  }

  function rejectModelChangeWhileBusy(resetNativeInput?: () => void): boolean {
    if (!modelBusy && canChangeModel) {
      return false
    }

    setStatusMessage('当前推理或模型处理尚未结束，请等待完成后再切换模型。')
    resetNativeInput?.()
    return true
  }

  async function handleOnnxSelection(file: File | null) {
    let shouldResetNativeInput = false
    if (rejectModelChangeWhileBusy(() => setOnnxInputKey((value) => value + 1))) {
      return
    }

    stopActiveStream()
    clearImportedModelState()
    resetSidecarSelections()
    setOnnxModelDraft(null)
    setModelBusy(true)
    setStatusMessage('正在释放旧模型推理资源...')
    await releaseCachedModelSessions()

    if (!file) {
      setStatusMessage('尚未选择 ONNX 模型文件。')
      setOnnxInputKey((value) => value + 1)
      setModelBusy(false)
      return
    }

    setStatusMessage(`正在解析 ONNX 模型 ${file.name}...`)

    try {
      const nextOnnxModel = await inspectOnnxModelFile(file)

      if (nextOnnxModel.family === 'hf-detr-like') {
        startTransition(() => {
          setOnnxModelDraft(nextOnnxModel)
          resetDetectionThreshold(nextOnnxModel.draftContract.decoder.scoreThreshold)
          resetMaxDetections()
          setStatusMessage(buildPendingHfMessage(null, null, supportsDirectoryPicker))
        })
        return
      }

      const model = await inspectImportedModel({ onnxModel: nextOnnxModel })
      setStatusMessage(`正在预热 ONNX 模型 ${file.name} 的推理会话...`)
      await prepareImportedModelSession(model)
      startTransition(() => {
        setOnnxModelDraft(nextOnnxModel)
        setImportedModel(model)
        resetDetectionThreshold(model.contract.decoder.scoreThreshold)
        resetMaxDetections()
        clearRecognitionOutputs()
        setStatusMessage(formatModelReadyMessage(model, webGpuSupportState))
      })
    } catch (error) {
      shouldResetNativeInput = true
      setStatusMessage(`模型解析或会话预热失败：${formatError(error)}`)
    } finally {
      setModelBusy(false)
      if (shouldResetNativeInput) {
        setOnnxInputKey((value) => value + 1)
      }
    }
  }

  async function handleConfigSelection(file: File | null) {
    let shouldResetNativeInput = false
    if (rejectModelChangeWhileBusy(() => setConfigInputKey((value) => value + 1))) {
      return
    }

    if (!file || !onnxModelDraft || onnxModelDraft.family !== 'hf-detr-like') {
      setConfigInputKey((value) => value + 1)
      return
    }

    try {
      validateSidecarFile(file, 'config')
    } catch (error) {
      setStatusMessage(`config.json 导入失败：${formatError(error)}`)
      setConfigInputKey((value) => value + 1)
      return
    }

    setModelBusy(true)
    setStatusMessage(`正在解析 ${CONFIG_FILE_NAME}...`)
    clearImportedModelState()
    await releaseCachedModelSessions()

    try {
      const model = await inspectImportedModel({
        onnxModel: onnxModelDraft,
        configFile: file,
        preprocessorConfigFile: preprocessorConfigFile ?? undefined,
      })
      setStatusMessage(`正在预热 ${onnxModelDraft.fileName} 的推理会话...`)
      await prepareImportedModelSession(model)
      startTransition(() => {
        setConfigFile(file)
        setImportedModel(model)
        resetDetectionThreshold(model.contract.decoder.scoreThreshold)
        resetMaxDetections()
        clearRecognitionOutputs()
        setStatusMessage(formatModelReadyMessage(model, webGpuSupportState))
      })
    } catch (error) {
      shouldResetNativeInput = true
      setStatusMessage(`config.json 导入或会话预热失败：${formatError(error)}`)
    } finally {
      setModelBusy(false)
      if (shouldResetNativeInput) {
        setConfigInputKey((value) => value + 1)
      }
    }
  }

  async function handlePreprocessorSelection(file: File | null) {
    let shouldResetNativeInput = false
    if (rejectModelChangeWhileBusy(() => setPreprocessorInputKey((value) => value + 1))) {
      return
    }

    if (!file || !onnxModelDraft || onnxModelDraft.family !== 'hf-detr-like') {
      setPreprocessorInputKey((value) => value + 1)
      return
    }

    try {
      validateSidecarFile(file, 'preprocessor')
    } catch (error) {
      setStatusMessage(`preprocessor_config.json 导入失败：${formatError(error)}`)
      setPreprocessorInputKey((value) => value + 1)
      return
    }

    if (!configFile) {
      startTransition(() => {
        setPreprocessorConfigFile(file)
        setImportedModel(null)
        resetDetectionThreshold(onnxModelDraft.draftContract.decoder.scoreThreshold)
        resetMaxDetections()
        setStatusMessage(buildPendingHfMessage(null, file, supportsDirectoryPicker))
      })
      return
    }

    setModelBusy(true)
    setStatusMessage(`正在解析 ${PREPROCESSOR_CONFIG_FILE_NAME}...`)
    clearImportedModelState()
    await releaseCachedModelSessions()

    try {
      const model = await inspectImportedModel({
        onnxModel: onnxModelDraft,
        configFile,
        preprocessorConfigFile: file,
      })
      setStatusMessage(`正在预热 ${onnxModelDraft.fileName} 的推理会话...`)
      await prepareImportedModelSession(model)
      startTransition(() => {
        setPreprocessorConfigFile(file)
        setImportedModel(model)
        resetDetectionThreshold(model.contract.decoder.scoreThreshold)
        resetMaxDetections()
        clearRecognitionOutputs()
        setStatusMessage(formatModelReadyMessage(model, webGpuSupportState))
      })
    } catch (error) {
      shouldResetNativeInput = true
      setStatusMessage(`preprocessor_config.json 导入或会话预热失败：${formatError(error)}`)
    } finally {
      setModelBusy(false)
      if (shouldResetNativeInput) {
        setPreprocessorInputKey((value) => value + 1)
      }
    }
  }

  async function handleAutoDiscoverSidecars() {
    if (!onnxModelDraft || onnxModelDraft.family !== 'hf-detr-like') {
      return
    }

    if (rejectModelChangeWhileBusy()) {
      return
    }

    if (typeof window.showDirectoryPicker !== 'function') {
      setStatusMessage('当前浏览器不支持目录授权，请手动导入 JSON 配置文件。')
      return
    }

    setModelBusy(true)
    setDiscoverBusy(true)
    setStatusMessage('正在请求目录权限并查找同目录配置文件...')

    try {
      const directoryHandle = await window.showDirectoryPicker()
      const discovered = await discoverHfSidecarsFromDirectory(directoryHandle)
      const nextConfigFile = discovered.configFile ?? configFile
      const nextPreprocessorConfigFile = discovered.preprocessorConfigFile ?? preprocessorConfigFile

      if (!nextConfigFile && !nextPreprocessorConfigFile) {
        setStatusMessage(
          `未在所选目录中找到 ${CONFIG_FILE_NAME} 或 ${PREPROCESSOR_CONFIG_FILE_NAME}。`,
        )
        return
      }

      if (!nextConfigFile) {
        startTransition(() => {
          if (discovered.preprocessorConfigFile) {
            setPreprocessorConfigFile(discovered.preprocessorConfigFile)
          }
          setStatusMessage(
            `已自动找到 ${PREPROCESSOR_CONFIG_FILE_NAME}，但仍缺少 ${CONFIG_FILE_NAME}。`,
          )
        })
        return
      }

      clearImportedModelState()
      await releaseCachedModelSessions()
      const model = await inspectImportedModel({
        onnxModel: onnxModelDraft,
        configFile: nextConfigFile,
        preprocessorConfigFile: nextPreprocessorConfigFile ?? undefined,
      })
      setStatusMessage(`正在预热 ${onnxModelDraft.fileName} 的推理会话...`)
      await prepareImportedModelSession(model)
      startTransition(() => {
        if (discovered.configFile) {
          setConfigFile(discovered.configFile)
        }
        if (discovered.preprocessorConfigFile) {
          setPreprocessorConfigFile(discovered.preprocessorConfigFile)
        }
        setImportedModel(model)
        resetDetectionThreshold(model.contract.decoder.scoreThreshold)
        resetMaxDetections()
        clearRecognitionOutputs()
        setStatusMessage(formatAutoDiscoveredMessage(model, discovered, webGpuSupportState))
      })
    } catch (error) {
      if (isAbortError(error)) {
        setStatusMessage('已取消自动查找同目录配置。')
      } else {
        setStatusMessage(`自动查找或会话预热失败：${formatError(error)}`)
      }
    } finally {
      setModelBusy(false)
      setDiscoverBusy(false)
    }
  }

  return {
    configFile,
    configInputKey,
    discoverBusy,
    handleAutoDiscoverSidecars,
    handleConfigSelection,
    handleOnnxSelection,
    handlePreprocessorSelection,
    importedModel,
    modelBusy,
    onnxInputKey,
    onnxModelDraft,
    preprocessorConfigFile,
    preprocessorInputKey,
  }
}
