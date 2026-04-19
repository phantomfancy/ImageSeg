import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const rootDir = fileURLToPath(new URL('.', import.meta.url))

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@contracts': resolve(rootDir, '../contracts/src/index.ts'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: Number(process.env.PORT ?? 5173),
    fs: {
      allow: [resolve(rootDir, '..')],
    },
  },
})
