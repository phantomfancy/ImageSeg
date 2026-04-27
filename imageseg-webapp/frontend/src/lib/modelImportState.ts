import type { ModelFamily } from '../../../contracts/src/index.ts'

export interface ModelImportControls {
  configEnabled: boolean
  configRequired: boolean
  preprocessorEnabled: boolean
  autoDiscoverEnabled: boolean
}

export function deriveModelImportControls(
  family: ModelFamily | null,
  supportsDirectoryPicker: boolean,
): ModelImportControls {
  const isHuggingFace = family === 'hf-detr-like'

  return {
    configEnabled: isHuggingFace,
    configRequired: isHuggingFace,
    preprocessorEnabled: isHuggingFace,
    autoDiscoverEnabled: isHuggingFace && supportsDirectoryPicker,
  }
}
