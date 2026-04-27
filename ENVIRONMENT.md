# 开发环境准备说明

当前仓库已改为 `TypeScript + React 19 + Aspire TypeScript AppHost`。

## 1. Node.js

安装 `Node.js 20+` 并验证：

```powershell
node --version
npm --version
```

## 2. Aspire CLI

安装 Aspire CLI，并确认版本为 `13.2+`，因为 `imageseg-webapp/apphost.ts` 依赖 TypeScript AppHost 支持：

```powershell
irm https://aspire.dev/install.ps1 | iex
aspire --version
```

## 3. 容器引擎（部署时需要）

如果需要执行 `aspire deploy`，当前仓库默认并验证的路径是 Docker Compose，因此请优先准备 Docker Desktop。

至少确认 Docker 命令可用：

```powershell
docker version
```

如果你使用 Podman，可先执行 `npm run publish` 生成 `aspire-output/`，再手动消费其中的 Compose 产物；当前仓库不保证 `npm run deploy` 对 Podman 的自动兼容。

## 4. 安装与初始化

在仓库根目录执行：

```powershell
npm install
aspire restore
```

`aspire restore` 会根据根目录 `aspire.config.json` 生成 `.modules/`，并由该配置指向 `imageseg-webapp/apphost.ts`。
当 `aspire.config.json` 中新增或升级 Aspire 包时，也要重新执行一次 `aspire restore`。
如果随后执行 `aspire run` 时提示 `.modules\\transport.ts` 缺少 `vscode-jsonrpc`，说明根目录依赖未安装完整，重新在仓库根目录执行 `npm install` 即可。

## 5. 启动、部署与验证

启动开发环境：

```powershell
npm run dev
```

发布部署产物：

```powershell
npm run publish
```

该命令会在仓库根目录生成 `aspire-output/`，其中至少包含 `docker-compose.yaml` 与 `.env`。

部署到本地 Docker：

```powershell
npm run deploy
```

清理本地 Docker Compose 部署：

```powershell
aspire do docker-compose-down-env
```

如果需要优先跑通前端静态站点部署，可改用 Podman + Nginx 容器：

```powershell
podman info
npm run static:image
npm run static:run
```

默认访问地址为 `http://localhost:8080/`。
如果 `podman info` 失败，先尝试：

```powershell
wsl --terminate podman-machine-default
podman machine start
podman info
```

停止静态站点容器：

```powershell
npm run static:stop
```

构建：

```powershell
npm run build
```

验证：

```powershell
npm run test
```
