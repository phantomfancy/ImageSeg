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
    description = "当前阶段提供识别能力和任务提交的占位接口。"
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
        RecognitionModes.Local => "已接受本地识别任务占位请求，等待浏览器端推理接入。",
        RecognitionModes.Server => "已接受服务器识别任务占位请求，等待基于 ONNX 的服务器端推理服务接入。",
        _ => "已接受 Auto 模式任务，占位逻辑将按能力自动选择执行路径。"
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
        "当前接口为骨架占位实现。",
        "后续将接入基于 ONNX 的服务器端推理链路。",
        "浏览器本地推理能力将通过前端可运行引擎补齐。"
    ];
}
