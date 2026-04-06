# 4CImageSeg

`4C-ai装备识别工具` 的源码仓库。

本项目目标是构建一个基于 `ASP.NET Core 10 + Blazor Web App + Interactive Auto + Aspire 13` 的渐进式 Web 应用，用于图片、视频和摄像头画面中的装备识别，并支持结果查看与导出。

## 项目功能

当前目标功能包括：

- 导入单张或多张图片/视频
- 调用摄像头读取视频流，以及摄像头校准
- 执行装备识别，计划支持本地识别，并预留服务器识别能力
- 导出结果图像/视频

当前前端界面已按以下流程组织：

1. 概览
2. 导入与采集
3. 执行检测
4. 结果与导出

当前识别能力规划：

- 浏览器端本地识别：后续实现
- 服务器端识别：后续实现
- `TensorFlow.NET`：当前计划用于训练探索与服务器侧推理探索

## 项目组成

- 解决方案文件：`4CImageSeg.slnx`
- SDK 约束：`global.json` 当前锁定 `.NET SDK 10.0.201`
- 编排入口：`4CImageSeg.AppHost`
- API：`4CImageSeg.ApiService`
- 前端宿主：`4CImageSeg.Web`
- 前端客户端：`4CImageSeg.Web.Client`
- 测试项目：`4CImageSeg.Tests`
- 公共默认配置：`4CImageSeg.ServiceDefaults`

## 构建与部署

- 本机命令行构建：

```powershell
dotnet build .\4CImageSeg.slnx
```

- 在解决方案中构建：
如需运行 Aspire 编排宿主，可从 `4CImageSeg.AppHost` 启动。

- 部署到服务器
如需部署，建议使用Aspire生成Docker镜像和Docker Compose配置文件后，通过Docker或Podman部署。

## 当前状态

已完成：

- 基本项目结构搭建与解决方案配置
- Aspire 示例骨架已替换为业务应用骨架
- 已完成全局布局、页面结构、侧边导航、底栏和UI文案编写
- 已实现导入图片/视频，实现读取摄像头视频流和摄像头校准功能
- 已完成识别 API 骨架与基本任务提交流程

未完成：

- 真实图片上传与媒体持久化
- 真实模型推理接入
- 本地识别闭环
- 服务器识别闭环
- 检测结果叠加显示
- 图像导出与视频导出闭环
- 视频逐帧识别与摄像头实时识别
- 数据集、训练、模型版本管理与部署完善
- 