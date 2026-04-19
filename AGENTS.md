# 4CImageSeg

## 1. 项目基础信息

- 项目名称：`4C-ai装备识别工具`
- 技术栈：`ONNX + React 19 + TypeScript + ONNX Runtime Web/WebGPU + Aspire TS AppHost`
- 项目目标：基于图像的装备识别，优先实现纯前端单图与批量识别，并通过 `Contracts` 统一不同 ONNX 模型输出格式。

## 2. 环境搭建

参考[ENVIRONMENT.md](./ENVIRONMENT.md)；

环境搭建完成后，至少确认：

- 已安装 `Node.js 20+`
- 已安装 `Aspire CLI`
- 已安装 `VS Code`

## 3. 构建与验证流程

对任何生成的代码，换行符采用"crlf"。

首次进入仓库后，先执行：

```powershell
npm install
aspire restore
```

运行 Aspire 编排宿主：

```powershell
npm run dev
```

运行测试：

```powershell
npm run test
```

提交前建议检查：

```powershell
git status --short
npm run build
npm run test
```
