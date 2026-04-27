export type InputMode = 'image' | 'video' | 'camera'

export type StreamState = 'idle' | 'running' | 'stopping' | 'exporting'

export type PageId = 'home' | 'workspace'

export type SectionId = 'input-import' | 'inference-settings' | 'results-export'

export type PreviewZoomTarget = 'image' | 'video' | 'camera' | 'result-canvas'

export type PreviewViewerOffset = {
  x: number
  y: number
}

export type ThemeMode = 'light' | 'dark' | 'system'

export type ResolvedTheme = 'light' | 'dark'

export type DetectionItem = {
  label: string
  confidence: number
  boxSummary: string
}

export type CameraDeviceOption = {
  deviceId: string
  label: string
}

export type NavItem<TId extends string> = {
  id: TId
  label: string
}

export type WorkspaceNavItem = NavItem<SectionId> & {
  description: string
}

export type HelpContentItem = {
  title: string
  body: string
}
