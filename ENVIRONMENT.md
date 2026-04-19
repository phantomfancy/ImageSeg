# 开发环境准备说明

当前仓库已改为 `TypeScript + React 19 + Aspire TypeScript AppHost`。

## 1. Node.js

安装 `Node.js 20+` 并验证：

```powershell
node --version
npm --version
```

## 2. Aspire CLI

安装 Aspire CLI，并确认版本为 `13.2+`，因为 `apphost.ts` 依赖 TypeScript AppHost 支持：

```powershell
irm https://aspire.dev/install.ps1 | iex
aspire --version
```

## 3. 安装与初始化

在仓库根目录执行：

```powershell
npm install
aspire restore
```

`aspire restore` 会根据根目录 `aspire.config.json` 生成 `.modules/`。
如果随后执行 `aspire run` 时提示 `.modules\\transport.ts` 缺少 `vscode-jsonrpc`，说明根目录依赖未安装完整，重新在仓库根目录执行 `npm install` 即可。

## 4. 启动与验证

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
