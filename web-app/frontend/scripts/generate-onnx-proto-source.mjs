import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDirectory = fileURLToPath(new URL('.', import.meta.url))
const frontendDirectory = path.resolve(scriptDirectory, '..')
const sourcePath = path.join(
  frontendDirectory,
  'node_modules',
  'onnxruntime-web',
  'lib',
  'onnxjs',
  'ort-schema',
  'protobuf',
  'onnx.js',
)
const outputPath = path.join(frontendDirectory, 'src', 'lib', 'onnxProtoSource.generated.ts')

const source = await readFile(sourcePath, 'utf8')
await mkdir(path.dirname(outputPath), { recursive: true })
await writeFile(
  outputPath,
  `const onnxProtoSource = ${JSON.stringify(source)} as const\n\nexport default onnxProtoSource\n`,
  'utf8',
)
