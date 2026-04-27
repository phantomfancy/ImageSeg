# ImageSeg

`ImageSeg-ai装备识别工具` 使用 `React 19 + TypeScript + ONNX Runtime Web/WebGPU + Aspire TypeScript AppHost` 架构，具有使用ultralytics格式或者hf格式模型进行图像识别推理的功能，并且开发了针对装备识别的额外功能。整个项目具有技术架构新和创新功能多的特点，同时应用功能完善、代码格式扎实。

## 当前结构

- `apphost.ts`：根目录 Aspire TS AppHost
- `web-app/contracts`：纯 TypeScript 模型契约包
- `web-app/frontend`：React 19 + TS/TSX 前端
- `pytorch-training/`：训练脚本、数据与导出模型

## 当前目标

`Contracts` 负责统一同为 ONNX 但输出格式不同的检测模型，当前优先支持：

- `ultralytics-yolo-detect`
- `ultralytics-rtdetr`
- `hf-detr-like`

成功标志：

- `pytorch-training/training_result` 下的所有 ONNX 模型都能被前端导入逻辑识别出契约 family

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

## 静态站点容器部署（Podman）

当前 `apphost.ts` 中的 `addViteApp(...)` 仍主要面向开发期编排；如果目标是先把前端静态站点单独部署起来，优先使用 `web-app/frontend/Dockerfile` + Nginx 容器。

前提：

- `podman machine` 已启动且 `podman info` 可正常返回
- 已在仓库根目录执行 `npm install`

如 Podman 仍处于 `Currently starting` 或 socket 拒绝连接，可先执行：

```powershell
wsl --terminate podman-machine-default
podman machine start
podman info
```

构建静态镜像：

```powershell
npm run static:image
```

启动本地静态站点容器：

```powershell
npm run static:run
```

默认访问地址：

- `http://localhost:8080/`

校验 ONNX Runtime 静态资源是否就绪：

- `http://localhost:8080/ort/ort-wasm-simd-threaded.asyncify.wasm`

停止容器：

```powershell
npm run static:stop
```

说明：

- 当前静态容器部署按根路径 `/` 提供服务。
- ONNX 模型文件仍由浏览器本地导入，不会打包进镜像。
- 如果需要通过 HTTPS 或反向代理对外发布，可在此基础上继续扩展 Nginx 配置。

构建：

```powershell
npm run build
```

验证：

```powershell
npm run test
```

注意：

- `web-app/frontend/public/ort` 是由 `npm run prepare:ort` 从 `onnxruntime-web` 复制生成的前端运行时资产目录，不需要提交到版本库。
- `npm run test` 中的 `npm run test:frontend` 与 `npm run verify:training-models` 依赖 `pytorch-training/training_result` 下的测试模型文件集。
- 如果当前机器没有准备 `pytorch-training/training_result`，这两项检查会因找不到 `.onnx` / `config.json` / `preprocessor_config.json` 而失败，这属于预期行为，不代表前端类型检查、`tsgo` 迁移或构建链本身存在问题。
- `npm run publish` / `npm run deploy` 依赖 `apphost.ts` 中已经声明 Docker Compose 部署环境，并依赖 `aspire.config.json` 中已包含 `Aspire.Hosting.Docker`。
- 在未准备该目录时，可先使用下面的命令验证不依赖训练产物的部分：

```powershell
npm run typecheck
npm run build
npm run test:contracts
```

## WebGPU 下 fp16 模型兼容处理

部分 `hf-detr-like` 的 `fp16` ONNX 模型在 `onnxruntime-web + WebGPU` 下可能出现 `Cast(13)` 与 `tensor(int64)` 相关报错，典型错误如下：

```text
Failed to find kernel for Cast(13) ... the node in the model has the following type (tensor(int64))
```

该问题可参考 ONNX Runtime 官方 issue：

- [microsoft/onnxruntime#25125](https://github.com/microsoft/onnxruntime/issues/25125)

当前已验证可用的处理方式如下：

1. 使用`pytorch_training`中`remove_back_to_back_cast.py`脚本, 来源于 [remove_back_to_back_cast.py](https://github.com/guschmue/ort-web-perf/blob/master/remove_back_to_back_cast.py)
2. 使用脚本转换原始模型，例如：

```powershell
python .\remove_back_to_back_cast.py .\rtdetrv2_original_style_hugginface\model_fp16.onnx .\rtdetrv2_original_style_hugginface\model_fp16_encoded.onnx
```

3. 在 Web App 中导入转换后的 `model_fp16_encoded.onnx` 进行推理
