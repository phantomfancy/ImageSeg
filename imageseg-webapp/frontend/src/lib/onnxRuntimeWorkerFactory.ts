// Keep Vite's ?worker import isolated from onnxRuntime.ts. Node-based tests import
// onnxRuntime.ts directly and inject a fake worker client, so they must not parse
// this Vite-only module unless the real browser worker path is used.
import OnnxRuntimeWorker from './onnxRuntimeWorker.ts?worker'

export default OnnxRuntimeWorker
