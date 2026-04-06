using System.Net.Http.Json;
using Microsoft.AspNetCore.Components.Forms;

namespace _4CImageSeg.Web.Client.Services;

public sealed class RecognitionWorkbenchState(HttpClient http)
{
    private readonly List<FileDescriptor> _images = [];
    private readonly List<FileDescriptor> _videos = [];
    private readonly IReadOnlyList<RecognitionModeOption> _recognitionModes =
    [
        new("auto", "自动选择"),
        new("local", "本机处理"),
        new("server", "服务器处理")
    ];

    private string _selectedMode = "auto";

    public event Action? Changed;

    public IReadOnlyList<FileDescriptor> Images => _images;
    public IReadOnlyList<FileDescriptor> Videos => _videos;
    public IReadOnlyList<RecognitionModeOption> RecognitionModes => _recognitionModes;
    public string StatusMessage { get; private set; } = "请先准备输入源，再开始检测。";
    public bool Submitting { get; private set; }
    public RecognitionJobResponse? LastJob { get; private set; }

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

    public int ImageCount => _images.Count;
    public int VideoCount => _videos.Count;
    public long TotalImageBytes => _images.Sum(item => item.Size);
    public long TotalVideoBytes => _videos.Sum(item => item.Size);
    public string SubmitButtonText => Submitting ? "提交中..." : "执行检测";

    public void HandleImagesSelected(InputFileChangeEventArgs args)
    {
        _images.Clear();
        _images.AddRange(args.GetMultipleFiles(32).Select(file => new FileDescriptor(file.Name, file.ContentType, file.Size)));
        StatusMessage = $"已选择 {ImageCount} 张图片，可继续选择处理方式并开始检测。";
        NotifyStateChanged();
    }

    public void HandleVideosSelected(InputFileChangeEventArgs args)
    {
        _videos.Clear();
        _videos.AddRange(args.GetMultipleFiles(16).Select(file => new FileDescriptor(file.Name, file.ContentType, file.Size)));
        StatusMessage = $"已选择 {VideoCount} 个视频，可继续选择处理方式并开始检测。";
        NotifyStateChanged();
    }

    public async Task SubmitRecognitionAsync()
    {
        Submitting = true;
        StatusMessage = "正在提交检测任务，请稍候...";
        NotifyStateChanged();

        try
        {
            var request = new RecognitionJobRequest(
                SelectedMode,
                ImageCount,
                VideoCount,
                TotalImageBytes + TotalVideoBytes);

            using var response = await http.PostAsJsonAsync("api/recognition/jobs", request);
            response.EnsureSuccessStatusCode();

            LastJob = await response.Content.ReadFromJsonAsync<RecognitionJobResponse>();
            StatusMessage = "检测任务已提交，可前往结果与导出查看处理状态。";
        }
        catch (Exception ex)
        {
            StatusMessage = $"任务提交失败：{ex.Message}";
        }
        finally
        {
            Submitting = false;
            NotifyStateChanged();
        }
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

    private void NotifyStateChanged() => Changed?.Invoke();
}

public sealed record FileDescriptor(string Name, string ContentType, long Size);
public sealed record RecognitionModeOption(string Value, string Label);
public sealed record RecognitionJobRequest(string Mode, int ImageCount, int VideoCount, long TotalBytes);
public sealed record RecognitionJobResponse(string JobId, string Mode, string Status, string Message);
