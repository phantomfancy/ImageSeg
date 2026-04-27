import {
  CONFIG_FILE_NAME,
  PREPROCESSOR_CONFIG_FILE_NAME,
} from '../lib/modelPackage'
import type { HelpContentItem, WorkspaceNavItem } from './types'

export const NAV_ITEMS: ReadonlyArray<WorkspaceNavItem> = [
  { id: 'input-import', label: '输入与导入', description: '输入源、模型与配置' },
  { id: 'inference-settings', label: '推理设置', description: '阈值、数量与执行控制' },
  { id: 'results-export', label: '结果与导出', description: '预览、结果与导出入口' },
]

export const HELP_CONTENT: ReadonlyArray<HelpContentItem> = [
  {
    title: '输入模式',
    body: '图片模式适合单张检测，视频模式适合回放与导出，摄像头模式适合实时识别与现场观察。',
  },
  {
    title: '模型与配置',
    body: `先导入 ONNX 模型；如识别为 Hugging Face 风格模型，再补充 ${CONFIG_FILE_NAME}，必要时补充 ${PREPROCESSOR_CONFIG_FILE_NAME}。`,
  },
  {
    title: '推理设置',
    body: '推理阈值用于控制结果过滤强度，识别数量用于限制绘制数量；新值在提交后才会真正生效。',
  },
  {
    title: '结果查看与导出',
    body: '统一预览区会在原始输入和叠加结果之间切换，结果区会同步显示检测项列表，并提供图片或视频导出入口。',
  },
  {
    title: '查看器操作',
    body: '图像和结果查看器支持滚轮缩放、拖拽平移、双击切换；视频与摄像头预览则保留更接近标准播放器的操作方式。',
  },
]
