import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'

const projectRoot = resolve(import.meta.dirname, '..')
const distRoot = resolve(projectRoot, 'node_modules', 'onnxruntime-web', 'dist')
const publicRoot = resolve(projectRoot, 'public', 'ort')
const files = [
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
]
const legacyFiles = [
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.jsep.wasm',
]

if (!existsSync(distRoot)) {
  throw new Error(`未找到 onnxruntime-web 产物目录：${distRoot}`)
}

mkdirSync(publicRoot, { recursive: true })

for (const fileName of files) {
  cpSync(resolve(distRoot, fileName), resolve(publicRoot, fileName), { force: true })
}

for (const fileName of legacyFiles) {
  rmSync(resolve(publicRoot, fileName), { force: true })
}
