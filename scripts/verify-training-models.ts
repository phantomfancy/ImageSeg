import { readdir } from 'node:fs/promises'
import path from 'node:path'

import { resolveModelContract } from '../web-app/contracts/src/index.ts'
import { inspectOnnxModelBytes } from '../web-app/frontend/src/lib/modelIntrospection.ts'

const trainingResultDirectory = path.resolve('pytorch-training', '4CImageSeg.Training', 'training_result')

const files = await collectOnnxFiles(trainingResultDirectory)
if (files.length === 0) {
  throw new Error(`未在 ${trainingResultDirectory} 下找到 ONNX 模型。`)
}

for (const file of files) {
  const bytes = new Uint8Array(await BunLikeFileRead(file))
  const parsedModel = inspectOnnxModelBytes(bytes)

  const contract = resolveModelContract({
    inputs: parsedModel.inputs,
    outputs: parsedModel.outputs,
    displayName: path.basename(file),
    labels: parsedModel.labels,
  })

  console.log(`${path.relative(process.cwd(), file)} -> ${contract.family}`)
}

console.log(`verified ${files.length} training_result model(s)`)

async function collectOnnxFiles(directory: string): Promise<string[]> {
  const items = await readdir(directory, { withFileTypes: true })
  const files: string[] = []

  for (const item of items) {
    const fullPath = path.join(directory, item.name)
    if (item.isDirectory()) {
      files.push(...await collectOnnxFiles(fullPath))
      continue
    }

    if (item.isFile() && item.name.toLowerCase().endsWith('.onnx')) {
      files.push(fullPath)
    }
  }

  return files.sort()
}
async function BunLikeFileRead(file: string): Promise<ArrayBuffer> {
  const { readFile } = await import('node:fs/promises')
  const buffer = await readFile(file)
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
}
