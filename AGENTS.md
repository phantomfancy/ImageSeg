# 4CImageSeg

## 1. 项目基础信息

- 项目名称：`4C-ai装备识别工具`
- 技术栈：`ONNX + ASP.NET Core 10 + Blazor Web App Auto + Aspire 13`
- 项目目标：基于图像的装备识别，支持单图和批量图片输入，输出识别结果和可视化结果图，支持本地端和服务器端两种识别模式。

## 2. 环境搭建

参考[environment.md](./environment.md)；

环境搭建完成后，至少确认：

- 已安装 `.NET SDK 10.0.201`
- `dotnet --version` 可用
- 已安装 `Aspire CLI`
- 已安装 `VS Code + C# Dev Kit`

## 3. 构建与验证流程

首次进入仓库后，先执行：

```powershell
dotnet restore .\4CImageSeg.slnx
dotnet build .\4CImageSeg.slnx
```

运行 Aspire 编排宿主：

```powershell
dotnet run --project .\4CImageSeg.AppHost\4CImageSeg.AppHost.csproj
```

运行测试：

```powershell
dotnet test .\4CImageSeg.Tests\4CImageSeg.Tests.csproj
```

提交前建议检查：

```powershell
git status --short
dotnet build .\4CImageSeg.slnx
dotnet test .\4CImageSeg.Tests\4CImageSeg.Tests.csproj
```
