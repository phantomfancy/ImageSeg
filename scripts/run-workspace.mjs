import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const [, , workspaceName, scriptName] = process.argv

if (!workspaceName || !scriptName) {
  console.error('Usage: node ./scripts/run-workspace.mjs <workspace-name> <script-name>')
  process.exit(1)
}

const workspacePaths = {
  '@4cimageseg/contracts': 'imageseg-webapp/contracts',
  '@4cimageseg/frontend': 'imageseg-webapp/frontend',
}

const workspacePath = workspacePaths[workspaceName]

if (!workspacePath) {
  console.error(`Unknown workspace: ${workspaceName}`)
  process.exit(1)
}

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url))
const repositoryRoot = path.resolve(scriptDirectory, '..')
const resolvedWorkspacePath = path.resolve(repositoryRoot, workspacePath)
const userAgent = process.env.npm_config_user_agent ?? ''
const isBun = /\bbun\//.test(userAgent) || 'Bun' in process.versions
const command = isBun ? 'bun' : 'npm'
const args = isBun
  ? ['run', scriptName]
  : ['run', scriptName, '--workspace', workspaceName]

const result = spawnSync(command, args, {
  cwd: isBun ? resolvedWorkspacePath : repositoryRoot,
  shell: process.platform === 'win32',
  stdio: 'inherit',
})

if (result.error) {
  throw result.error
}

process.exit(result.status ?? 1)
