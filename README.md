# 4CImageSeg

`4C-ai装备识别工具` 使用 `React 19 + TypeScript + ONNX Runtime Web/WebGPU + Aspire TypeScript AppHost` 架构，具有使用ultralytics格式或者hf格式模型进行图像识别推理的功能，并且开发了针对装备识别的额外功能。整个项目具有技术架构新和创新功能多的特点，同时应用功能完善、代码格式扎实。

## 当前结构

- `apphost.ts`：根目录 Aspire TS AppHost
- `web-app/contracts`：纯 TypeScript 模型契约包
- `web-app/frontend`：React 19 + TS/TSX 前端
- `pytorch-training/4CImageSeg.Training`：训练脚本、数据与导出模型

## 当前目标

`Contracts` 负责统一同为 ONNX 但输出格式不同的检测模型，当前优先支持：

- `ultralytics-yolo-detect`
- `ultralytics-rtdetr`
- `hf-detr-like`

成功标志：

- `pytorch-training/4CImageSeg.Training/training_result` 下的所有 ONNX 模型都能被前端导入逻辑识别出契约 family

## 常用命令

安装依赖：

```powershell
npm install
```

`aspire run` 会直接执行根目录 `apphost.ts`，因此根目录 `node_modules` 必须包含 AppHost 运行时依赖；如果出现 `Cannot find package 'vscode-jsonrpc' imported from .modules\\transport.ts`，先重新执行一次 `npm install`。

生成 Aspire TS SDK：

```powershell
aspire restore
```

当 `aspire.config.json` 的 `packages` 发生变化时，也需要重新执行一次 `aspire restore`，以更新 `.modules/` 中的 TypeScript AppHost SDK。

启动开发环境：

```powershell
npm run dev
```

上面的 `npm run dev` 会通过Aspire命令`aspire run`启动整个编排环境，而不是直接启动 Vite。

如果只想单独启动前端 Vite 开发服务器，可使用以下任一方式：

```powershell
cd .\web-app\frontend
npm run dev
```

```powershell
npm run dev --workspace @4cimageseg/frontend
```

前端 Vite 默认监听 `0.0.0.0:5173`，本机访问地址通常为 `http://localhost:5173`。如需修改端口，可先设置 `PORT` 环境变量，例如：

```powershell
$env:PORT=3000
npm run dev --workspace @4cimageseg/frontend
```

发布部署产物：

```powershell
npm run publish
```

`npm run publish` 会在仓库根目录生成 `aspire-output/`，其中至少包含：

- `docker-compose.yaml`
- `.env`

如需按环境生成部署参数，可使用：

```powershell
aspire do prepare-env --environment Staging
```

执行本地部署：

```powershell
npm run deploy
```

停止并清理本地 Docker Compose 部署：

```powershell
aspire do docker-compose-down-env
```

部署前提：

- 已安装并启动 Docker Desktop
- 已在仓库根目录执行 `npm install`
- 已执行 `aspire restore`

关于 Podman：

- 当前仓库内建并验证的官方路径是 Docker Compose。
- 如果你本地主要使用 Podman，建议先执行 `npm run publish`，再手动消费 `aspire-output/` 中的 Compose 产物；本仓库当前不再承诺 `npm run deploy` 对 Podman 的自动兼容。

构建：

```powershell
npm run build
```

验证：

```powershell
npm run test
```

注意：

- `npm run test` 中的 `npm run test:frontend` 与 `npm run verify:training-models` 依赖 `pytorch-training/training_result` 下的测试模型文件集。
- 如果当前机器没有准备 `pytorch-training/training_result`，这两项检查会因找不到 `.onnx` / `config.json` / `preprocessor_config.json` 而失败，这属于预期行为，不代表前端类型检查、`tsgo` 迁移或构建链本身存在问题。
- `npm run publish` / `npm run deploy` 依赖 `apphost.ts` 中已经声明 Docker Compose 部署环境，并依赖 `aspire.config.json` 中已包含 `Aspire.Hosting.Docker`。
- 在未准备该目录时，可先使用下面的命令验证不依赖训练产物的部分：

```powershell
npm run typecheck
npm run build
npm run test:contracts
```
