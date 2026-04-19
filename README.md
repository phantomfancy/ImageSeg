# 4CImageSeg

`4C-ai装备识别工具` 已切换到 `React 19 + TypeScript + ONNX Runtime Web/WebGPU + Aspire TypeScript AppHost` 架构。

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

启动开发环境：

```powershell
npm run dev
```

构建：

```powershell
npm run build
```

验证：

```powershell
npm run test
```
