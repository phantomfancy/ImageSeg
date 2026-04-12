using Microsoft.AspNetCore.Components.Forms;
using _4CImageSeg.Contracts;

namespace _4CImageSeg.Web.Client.Services;

public sealed class RecognitionWorkbenchState(RecognitionApiClient apiClient)
{
    private const long MaxImageBytes = 15 * 1024 * 1024;
    private const long MaxModelBytes = 256L * 1024 * 1024;
    private const string DefaultLocalModelUrl = "/models/yolo26s/model_int8.onnx";
    private const string DefaultLocalModelVersion = "builtin:/models/yolo26s/model_int8.onnx";
    private const string DefaultLocalModelDisplayName = "内置模型 yolo26s / model_int8.onnx";

    private readonly List<FileDescriptor> _images = [];
    private readonly List<FileDescriptor> _videos = [];
    private IReadOnlyList<RecognitionModeDto> _recognitionModes =
    [
        new(RecognitionModes.Auto, "自动选择", true),
        new(RecognitionModes.Local, "本地识别", true),
        new(RecognitionModes.Server, "服务器识别", true)
    ];

    private string _selectedMode = RecognitionModes.Auto;
    private string _localModelSourceType = LocalModelSourceTypes.Default;
    private string _customLocalModelUrl = string.Empty;
    private ImportedLocalModel? _importedLocalModel;
    private string _localModelNotice = "当前使用内置 ONNX 模型。";
    private bool _initialized;

    public event Action? Changed;

    public IReadOnlyList<FileDescriptor> Images => _images;
    public IReadOnlyList<FileDescriptor> Videos => _videos;
    public IReadOnlyList<RecognitionModeDto> AvailableRecognitionModes => _recognitionModes;
    public RecognitionCapabilitiesDto? Capabilities { get; private set; }
    public SelectedImageAsset? CurrentImage { get; private set; }
    public RecognitionJobResponse? LastJob { get; private set; }
    public RecognitionResultDto? LastResult { get; private set; }
    public string? AnnotatedImageDataUrl { get; private set; }
    public ImportedLocalModel? ImportedLocalModel => _importedLocalModel;
    public string LocalModelNotice => _localModelNotice;
    public string StatusMessage { get; private set; } = "请先准备输入源，再开始检测。";
    public bool Processing { get; private set; }

    public string SelectedMode
    {
        get => _selectedMode;
        set
        {
            if (_selectedMode == value)
            {
                return;
            }

            _selectedMode = value;
            NotifyStateChanged();
        }
    }

    public string LocalModelSourceType
    {
        get => _localModelSourceType;
        set => SetLocalModelSourceType(value);
    }

    public string CustomLocalModelUrl
    {
        get => _customLocalModelUrl;
        set => SetCustomLocalModelUrl(value);
    }

    public int ImageCount => _images.Count;
    public int VideoCount => _videos.Count;
    public long TotalImageBytes => _images.Sum(item => item.Size);
    public long TotalVideoBytes => _videos.Sum(item => item.Size);
    public string SubmitButtonText => Processing ? "处理中..." : "执行检测";
    public bool HasDetectionResult => LastResult is not null && !string.IsNullOrWhiteSpace(AnnotatedImageDataUrl);
    public string? LocalModelValidationMessage => GetLocalModelValidationMessage();
    public bool IsLocalModelSelectionReady => string.IsNullOrWhiteSpace(LocalModelValidationMessage);
    public string CurrentLocalModelDisplayName => BuildCurrentLocalModelDisplayName();

    public string EffectiveMode
    {
        get
        {
            var requested = RecognitionModes.Normalize(SelectedMode);

            if (requested == RecognitionModes.Auto)
            {
                if (Capabilities?.SupportsLocalInference is true)
                {
                    return RecognitionModes.Local;
                }

                if (Capabilities?.SupportsServerInference is true)
                {
                    return RecognitionModes.Server;
                }
            }

            return requested;
        }
    }

    public async Task InitializeAsync()
    {
        if (_initialized)
        {
            return;
        }

        try
        {
            Capabilities = await apiClient.GetCapabilitiesAsync();
            var modes = await apiClient.GetModesAsync();

            if (modes.Count > 0)
            {
                _recognitionModes = modes;
            }

            if (!string.IsNullOrWhiteSpace(Capabilities?.DefaultMode))
            {
                _selectedMode = Capabilities.DefaultMode;
            }
        }
        catch (Exception ex)
        {
            StatusMessage = $"读取识别能力失败：{ex.Message}";
        }
        finally
        {
            _initialized = true;
            NotifyStateChanged();
        }
    }

    public void SetLocalModelSourceType(string? value)
    {
        var normalized = LocalModelSourceTypes.Normalize(value);
        if (_localModelSourceType == normalized)
        {
            return;
        }

        _localModelSourceType = normalized;
        UpdateLocalModelNotice();
        ClearRecognitionOutputCore();
        NotifyStateChanged();
    }

    public void SetCustomLocalModelUrl(string? value)
    {
        var normalized = value?.Trim() ?? string.Empty;
        if (string.Equals(_customLocalModelUrl, normalized, StringComparison.Ordinal))
        {
            return;
        }

        _customLocalModelUrl = normalized;
        UpdateLocalModelNotice();
        ClearRecognitionOutputCore();
        NotifyStateChanged();
    }

    public async Task HandleImagesSelectedAsync(InputFileChangeEventArgs args)
    {
        _images.Clear();

        var files = args.GetMultipleFiles(32);
        _images.AddRange(files.Select(file => new FileDescriptor(file.Name, file.ContentType, file.Size)));

        CurrentImage = null;
        ClearRecognitionOutputCore();

        if (files.Count == 0)
        {
            StatusMessage = "尚未选择图片。";
            NotifyStateChanged();
            return;
        }

        var firstImage = files[0];

        try
        {
            await using var stream = firstImage.OpenReadStream(MaxImageBytes);
            using var memory = new MemoryStream();
            await stream.CopyToAsync(memory);
            var bytes = memory.ToArray();
            var dataUrl = $"data:{firstImage.ContentType};base64,{Convert.ToBase64String(bytes)}";

            CurrentImage = new SelectedImageAsset(firstImage.Name, firstImage.ContentType, firstImage.Size, dataUrl);
            StatusMessage = files.Count > 1
                ? $"已选择 {files.Count} 张图片。当前阶段先使用第 1 张图片进行本地识别。"
                : "已选择 1 张图片，可直接开始本地识别。";
        }
        catch (Exception ex)
        {
            CurrentImage = null;
            StatusMessage = $"读取图片失败：{ex.Message}";
        }

        NotifyStateChanged();
    }

    public async Task HandleLocalModelSelectedAsync(InputFileChangeEventArgs args)
    {
        var modelFile = args.GetMultipleFiles(1).FirstOrDefault();

        if (modelFile is null)
        {
            _importedLocalModel = null;
            UpdateLocalModelNotice();
            ClearRecognitionOutputCore();
            NotifyStateChanged();
            return;
        }

        try
        {
            var extension = Path.GetExtension(modelFile.Name);
            if (!string.Equals(extension, ".onnx", StringComparison.OrdinalIgnoreCase) &&
                !string.Equals(extension, ".ort", StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException("仅支持导入 .onnx 或 .ort 模型文件。");
            }

            await using var stream = modelFile.OpenReadStream(MaxModelBytes);
            using var memory = new MemoryStream();
            await stream.CopyToAsync(memory);

            _importedLocalModel = new ImportedLocalModel(
                modelFile.Name,
                modelFile.ContentType,
                modelFile.Size,
                memory.ToArray(),
                Guid.NewGuid().ToString("N"));

            UpdateLocalModelNotice();
        }
        catch (Exception ex)
        {
            _importedLocalModel = null;
            _localModelNotice = $"导入本地模型失败：{ex.Message}";
        }

        ClearRecognitionOutputCore();
        NotifyStateChanged();
    }

    public void HandleVideosSelected(InputFileChangeEventArgs args)
    {
        _videos.Clear();
        _videos.AddRange(args.GetMultipleFiles(16).Select(file => new FileDescriptor(file.Name, file.ContentType, file.Size)));
        ClearRecognitionOutputCore();

        StatusMessage = _videos.Count == 0
            ? "尚未选择视频。"
            : $"已选择 {_videos.Count} 个视频。当前阶段暂未提供视频识别，仅保留输入信息。";

        NotifyStateChanged();
    }

    public async Task SubmitServerRecognitionAsync(CancellationToken cancellationToken = default)
    {
        BeginProcessing("正在提交识别任务，请稍候...");

        try
        {
            var request = new RecognitionJobRequest(
                EffectiveMode,
                ImageCount,
                VideoCount,
                TotalImageBytes + TotalVideoBytes);

            LastJob = await apiClient.SubmitJobAsync(request, cancellationToken);
            StatusMessage = LastJob is null
                ? "识别任务提交完成，但未返回任务数据。"
                : "识别任务已提交，可在结果与导出区域查看任务信息。";
        }
        catch (Exception ex)
        {
            StatusMessage = $"任务提交失败：{ex.Message}";
        }
        finally
        {
            Processing = false;
            NotifyStateChanged();
        }
    }

    public void BeginProcessing(string message)
    {
        Processing = true;
        StatusMessage = message;
        NotifyStateChanged();
    }

    public void CompleteLocalRecognition(RecognitionResultDto result, string annotatedImageDataUrl)
    {
        LastJob = new RecognitionJobResponse(
            $"local-{DateTimeOffset.UtcNow:yyyyMMddHHmmssfff}",
            result.Mode,
            RecognitionJobStatuses.Completed,
            "浏览器端单图识别已完成。",
            ImageCount,
            VideoCount,
            TotalImageBytes + TotalVideoBytes);

        LastResult = result;
        AnnotatedImageDataUrl = annotatedImageDataUrl;
        Processing = false;
        StatusMessage = result.Detections.Count == 0
            ? "识别完成，当前图片未检测到目标。"
            : $"识别完成，共检测到 {result.Detections.Count} 个目标。";

        NotifyStateChanged();
    }

    public void FailProcessing(string message)
    {
        Processing = false;
        StatusMessage = message;
        NotifyStateChanged();
    }

    public static string FormatBytes(long bytes)
    {
        string[] units = ["B", "KB", "MB", "GB"];
        double value = bytes;
        var unitIndex = 0;

        while (value >= 1024 && unitIndex < units.Length - 1)
        {
            value /= 1024;
            unitIndex++;
        }

        return $"{value:0.##} {units[unitIndex]}";
    }

    public bool TryBuildLocalModelRequest(out BrowserLocalModelRequest? request, out string? validationMessage)
    {
        validationMessage = GetLocalModelValidationMessage();
        if (!string.IsNullOrWhiteSpace(validationMessage))
        {
            request = null;
            return false;
        }

        switch (_localModelSourceType)
        {
            case LocalModelSourceTypes.Url:
                request = new BrowserLocalModelRequest(
                    $"url:{_customLocalModelUrl}",
                    LocalModelSourceTypes.Url,
                    _customLocalModelUrl,
                    _customLocalModelUrl,
                    BuildCurrentLocalModelDisplayName(),
                    null);
                return true;

            case LocalModelSourceTypes.File:
                request = new BrowserLocalModelRequest(
                    $"file:{_importedLocalModel!.CacheKey}",
                    LocalModelSourceTypes.File,
                    null,
                    $"本地文件：{_importedLocalModel.Name}",
                    _importedLocalModel.Name,
                    _importedLocalModel.Bytes);
                return true;

            default:
                request = new BrowserLocalModelRequest(
                    "builtin:yolo26s:model_int8",
                    LocalModelSourceTypes.Default,
                    DefaultLocalModelUrl,
                    DefaultLocalModelVersion,
                    DefaultLocalModelDisplayName,
                    null);
                return true;
        }
    }

    private void ClearRecognitionOutputCore()
    {
        LastJob = null;
        LastResult = null;
        AnnotatedImageDataUrl = null;
    }

    private string? GetLocalModelValidationMessage()
    {
        if (_localModelSourceType == LocalModelSourceTypes.Url)
        {
            if (string.IsNullOrWhiteSpace(_customLocalModelUrl))
            {
                return "请输入可直接访问的 ONNX 模型 URL。";
            }

            if (!Uri.TryCreate(_customLocalModelUrl, UriKind.Absolute, out var modelUri) ||
                (modelUri.Scheme != Uri.UriSchemeHttps && modelUri.Scheme != Uri.UriSchemeHttp))
            {
                return "请输入有效的 http/https 模型 URL。";
            }
        }

        if (_localModelSourceType == LocalModelSourceTypes.File && _importedLocalModel is null)
        {
            return "请选择本地 ONNX 模型文件。";
        }

        return null;
    }

    private string BuildCurrentLocalModelDisplayName()
    {
        if (_localModelSourceType == LocalModelSourceTypes.Url && !string.IsNullOrWhiteSpace(_customLocalModelUrl))
        {
            if (!Uri.TryCreate(_customLocalModelUrl, UriKind.Absolute, out var modelUri))
            {
                return _customLocalModelUrl;
            }

            var fileName = Path.GetFileName(modelUri.AbsolutePath);
            return string.IsNullOrWhiteSpace(fileName) ? _customLocalModelUrl : fileName;
        }

        if (_localModelSourceType == LocalModelSourceTypes.File && _importedLocalModel is not null)
        {
            return _importedLocalModel.Name;
        }

        return DefaultLocalModelDisplayName;
    }

    private void UpdateLocalModelNotice()
    {
        _localModelNotice = _localModelSourceType switch
        {
            LocalModelSourceTypes.Url when string.IsNullOrWhiteSpace(_customLocalModelUrl)
                => "请输入可直接访问的 ONNX 模型 URL。URL 导入完全由浏览器直连，目标站点需要允许跨域访问。",
            LocalModelSourceTypes.Url
                => $"当前将通过 URL 加载模型：{_customLocalModelUrl}",
            LocalModelSourceTypes.File when _importedLocalModel is not null
                => $"已导入本地模型：{_importedLocalModel.Name}（{FormatBytes(_importedLocalModel.Size)}）。",
            LocalModelSourceTypes.File
                => "请选择本地 ONNX 模型文件。模型文件仅在当前浏览器会话内有效。",
            _ => "当前使用内置 ONNX 模型。"
        };
    }

    private void NotifyStateChanged() => Changed?.Invoke();
}

public sealed record FileDescriptor(string Name, string ContentType, long Size);

public sealed record SelectedImageAsset(
    string Name,
    string ContentType,
    long Size,
    string DataUrl);

public static class LocalModelSourceTypes
{
    public const string Default = "default";
    public const string Url = "url";
    public const string File = "file";

    public static string Normalize(string? value) =>
        value?.Trim().ToLowerInvariant() switch
        {
            Url => Url,
            File => File,
            _ => Default
        };
}

public sealed record ImportedLocalModel(
    string Name,
    string ContentType,
    long Size,
    byte[] Bytes,
    string CacheKey);

public sealed record BrowserLocalModelRequest(
    string CacheKey,
    string SourceType,
    string? ModelUrl,
    string ModelVersion,
    string DisplayName,
    byte[]? ModelBytes);
