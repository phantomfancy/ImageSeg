using Microsoft.Extensions.Options;
using _4CImageSeg.Contracts;
using _4CImageSeg.ApiService;

var builder = WebApplication.CreateBuilder(args);

// Add service defaults & Aspire client integrations.
builder.AddServiceDefaults();

// Add services to the container.
builder.Services.AddProblemDetails();
builder.Services.AddOpenApi();
builder.Services.Configure<RecognitionOptions>(builder.Configuration.GetSection("Recognition"));
builder.Services.Configure<OnnxRecognitionOptions>(builder.Configuration.GetSection("OnnxRecognition"));
builder.Services.AddHttpClient();
builder.Services.AddSingleton<OnnxRecognitionService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
app.UseExceptionHandler();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.MapGet("/", () => Results.Ok(new
{
    service = "4C-ai装备识别工具 API",
    status = "running",
    description = "当前提供基于 ONNX Runtime 的识别能力接口，服务器端由 Microsoft.ML.OnnxRuntime 执行推理。"
}));

app.MapGet("/api/recognition/capabilities", (IOptions<RecognitionOptions> options) =>
{
    var value = options.Value;

    return Results.Ok(new RecognitionCapabilitiesDto(
        value.SupportsLocalInference,
        value.SupportsServerInference,
        value.SupportsImageExport,
        value.SupportsVideoExport,
        value.DefaultMode,
        value.Notes));
});

app.MapGet("/api/recognition/modes", (IOptions<RecognitionOptions> options) =>
{
    var value = options.Value;

    var modes = new List<RecognitionModeDto>
    {
        new(RecognitionModes.Auto, "自动选择", true),
        new(RecognitionModes.Local, "本地识别", value.SupportsLocalInference),
        new(RecognitionModes.Server, "服务器识别", value.SupportsServerInference)
    };

    return Results.Ok(modes);
});

app.MapPost("/api/recognition/jobs", (RecognitionJobRequest request, IOptions<RecognitionOptions> options) =>
{
    var value = options.Value;
    var normalizedMode = RecognitionModes.Normalize(request.Mode);

    var acceptedMode = normalizedMode switch
    {
        RecognitionModes.Local when value.SupportsLocalInference => RecognitionModes.Local,
        RecognitionModes.Server when value.SupportsServerInference => RecognitionModes.Server,
        _ => value.DefaultMode
    };

    var message = acceptedMode switch
    {
        RecognitionModes.Local => "已接受本地识别请求，浏览器端将通过 onnxruntime-web 执行推理。",
        RecognitionModes.Server => "已接受服务器识别请求，ApiService 将通过 Microsoft.ML.OnnxRuntime 执行推理。",
        _ => "已接受 Auto 模式任务，系统将按当前能力自动选择 onnxruntime 或 onnxruntime-web 执行路径。"
    };

    return Results.Accepted($"/api/recognition/jobs/{Guid.NewGuid():N}", new RecognitionJobResponse(
        $"job-{DateTimeOffset.UtcNow:yyyyMMddHHmmssfff}",
        acceptedMode,
        RecognitionJobStatuses.Queued,
        message,
        request.ImageCount,
        request.VideoCount,
        request.TotalBytes));
});

app.MapPost("/api/recognition/server-detect", async (
    IFormFile image,
    OnnxRecognitionService inferenceService,
    IOptions<OnnxRecognitionOptions> options,
    CancellationToken cancellationToken) =>
{
    var settings = options.Value;

    if (image.Length == 0)
    {
        return Results.BadRequest("图片文件不能为空。");
    }

    if (image.Length > settings.MaxUploadBytes)
    {
        return Results.BadRequest($"图片大小不能超过 {settings.MaxUploadBytes / 1024 / 1024} MB。");
    }

    if (string.IsNullOrWhiteSpace(image.ContentType) || !image.ContentType.StartsWith("image/", StringComparison.OrdinalIgnoreCase))
    {
        return Results.BadRequest("仅支持图片文件。");
    }

    await using var stream = image.OpenReadStream();
    var result = await inferenceService.RunSingleImageAsync(stream, image.FileName, cancellationToken);
    return Results.Ok(result);
})
.DisableAntiforgery();

app.MapDefaultEndpoints();

app.Run();

sealed class RecognitionOptions
{
    public bool SupportsLocalInference { get; set; } = true;
    public bool SupportsServerInference { get; set; } = true;
    public bool SupportsImageExport { get; set; } = true;
    public bool SupportsVideoExport { get; set; } = false;
    public string DefaultMode { get; set; } = "auto";
    public string[] Notes { get; set; } =
    [
        "服务器端单图识别由 ASP.NET Core + Microsoft.ML.OnnxRuntime 执行。",
        "浏览器本地识别由 onnxruntime-web 执行，优先使用 WebGPU，失败后回退到 WASM。",
        "当前前后端默认模型均为 onnx-community/yolo26s-ONNX 的 onnx/model_int8.onnx。"
    ];
}
