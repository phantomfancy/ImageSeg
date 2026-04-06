# 开发环境准备说明

以下配置流程默认适用于Windows 10及以上系统。项目使用 .NET SDK 10.x 和 Aspire 13.x，使用dotnet CLI/VSCode/Visual Studio进行开发。

推荐使用winget安装，首先配置winget源：

```powershell
winget source add winget https://mirrors.ustc.edu.cn/winget-source --trust-level trusted
```

## 1. 配置 .NET SDK 和 dotnet CLI

### 1.1 安装 .NET 10 SDK

在 https://dotnet.microsoft.com/zh-cn/download 下载或者使用 `winget`：

```powershell
winget install Microsoft.DotNet.SDK.10
```

### 1.2 验证 dotnet CLI

安装完成后，重新打开 PowerShell，依次执行：

```powershell
dotnet --info
dotnet --list-sdks
dotnet --version
```

如有输出则说明已经成功安装dotnet SDK。为满足本项目的开发，建议至少确认输出中包含：

- `10.0.2xx`

### 1.3 配置本地 HTTPS 开发证书

Blazor Web App / Aspire 本地调试通常需要本地 HTTPS 证书。执行：

```powershell
dotnet dev-certs https --trust
```

执行后按系统提示信任证书。

### 1.4 验证本仓库 CLI 构建

在仓库根目录执行：

```powershell
dotnet restore .\4CImageSeg.slnx
dotnet build .\4CImageSeg.slnx
```
如果能生成成功，说明dotnet环境已正确配置并可用于本项目开发。

## 2. 配置 Aspire 13 开发环境

### 2.1 安装Aspire CLI

参照 https://aspire.dev/zh-cn/get-started/install-cli/ 安装，执行：

```powershell
irm https://aspire.dev/install.ps1 | iex
```

安装完成后，使用`aspire --version`验证安装是否成功。在仓库根目录执行以下命令，验证本仓库能正确使用aspire运行和调试：

```powershell
dotnet restore .\4CImageSeg.slnx
dotnet build .\4CImageSeg.slnx
dotnet run --project .\4CImageSeg.AppHost\4CImageSeg.AppHost.csproj
```

## 3. 使用 VS Code 配置本项目开发环境

### 3.1 安装 VS Code

在 https://code.visualstudio.com/ 下载或者使用 `winget`：

```powershell
winget install Microsoft.VisualStudioCode
```

### 3.2 安装 C# Dev Kit

推荐安装以下扩展：

- `.NET Install Tool`
- `C# Dev Kit`
- `C#`
- `Aspire`

### 3.3 首次打开后的建议动作

首次打开仓库后，建议依次确认：

1. 选择 `Trust Workspace`
2. 等待 C# Dev Kit 加载解决方案
3. 如有提示，执行还原依赖
4. 在终端确认：

```powershell
dotnet restore .\4CImageSeg.slnx
dotnet build .\4CImageSeg.slnx
```

## 4. 配置 Podman 环境

### 4.1 安装 Podman

**推荐：如果不是马上需要部署，可以直接跳过本节！！！**
本仓库部署目标包含 `Podman`，建议先安装wsl2环境，然后安装 `Podman Desktop`。

推荐使用 `winget`：

```powershell
winget install RedHat.Podman-Desktop
```

官方 Windows 安装文档：

- https://podman-desktop.io/docs/installation/windows-install


安装完成后，重新打开powershell执行以下脚本验证安装是否成功：

```powershell
podman info
podman --version
```

### 4.2 为 Aspire 指定 Podman

Aspire 官方文档说明可以通过环境变量指定 OCI runtime。  
如果本机同时安装了 Docker 和 Podman，建议显式指定本仓库使用 `Podman`：

```powershell
[System.Environment]::SetEnvironmentVariable("ASPIRE_CONTAINER_RUNTIME", "podman", "User")
```

设置完成后，重开 PowerShell，再检查：

```powershell
$env:ASPIRE_CONTAINER_RUNTIME
```

期望输出：

```text
podman
```
