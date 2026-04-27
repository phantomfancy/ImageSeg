import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const frontendDirectory = resolve(scriptDirectory, '..')
const sourcePath = join(
  frontendDirectory,
  'node_modules',
  'onnxruntime-web',
  'lib',
  'onnxjs',
  'ort-schema',
  'protobuf',
  'onnx.js',
)
const outputPath = join(frontendDirectory, 'src', 'lib', 'onnxProtoSource.generated.ts')

const source = await readFile(sourcePath, 'utf8')
await mkdir(dirname(outputPath), { recursive: true })
await writeFile(
  outputPath,
  `const onnxProtoSource = ${JSON.stringify(source)} as const\n\nexport default onnxProtoSource\n`,
  'utf8',
)
