using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);

// Add service defaults & Aspire client integrations.
builder.AddServiceDefaults();

// Add services to the container.
builder.Services.AddProblemDetails();
builder.Services.AddOpenApi();
builder.Services.Configure<RecognitionOptions>(builder.Configuration.GetSection("Recognition"));

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

    return Results.Ok(new RecognitionCapabilitiesResponse(
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

    var modes = new List<RecognitionModeResponse>
    {
        new("auto", "Auto", true),
        new("local", "本地识别", value.SupportsLocalInference),
        new("server", "服务器识别", value.SupportsServerInference)
    };

    return Results.Ok(modes);
});

app.MapPost("/api/recognition/jobs", (RecognitionJobRequest request, IOptions<RecognitionOptions> options) =>
{
    var value = options.Value;
    var normalizedMode = string.IsNullOrWhiteSpace(request.Mode) ? value.DefaultMode : request.Mode.Trim().ToLowerInvariant();

    var acceptedMode = normalizedMode switch
    {
        "local" when value.SupportsLocalInference => "local",
        "server" when value.SupportsServerInference => "server",
        _ => value.DefaultMode
    };

    var message = acceptedMode switch
    {
        "local" => "已接受本地识别任务占位请求，等待浏览器端推理接入。",
        "server" => "已接受服务器识别任务占位请求，等待 TensorFlow.NET 推理服务接入。",
        _ => "已接受 Auto 模式任务，占位逻辑将按能力自动选择执行路径。"
    };

    return Results.Accepted($"/api/recognition/jobs/{Guid.NewGuid():N}", new RecognitionJobResponse(
        $"job-{DateTimeOffset.UtcNow:yyyyMMddHHmmssfff}",
        acceptedMode,
        "queued",
        message,
        request.ImageCount,
        request.VideoCount,
        request.TotalBytes));
});

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
        "后续将接入 TensorFlow.NET 训练和服务器推理。",
        "浏览器本地推理能力将通过前端可运行引擎补齐。"
    ];
}

sealed record RecognitionCapabilitiesResponse(
    bool SupportsLocalInference,
    bool SupportsServerInference,
    bool SupportsImageExport,
    bool SupportsVideoExport,
    string DefaultMode,
    string[] Notes);

sealed record RecognitionModeResponse(string Value, string Label, bool Enabled);

sealed record RecognitionJobRequest(
    string Mode,
    int ImageCount,
    int VideoCount,
    long TotalBytes);

sealed record RecognitionJobResponse(
    string JobId,
    string Mode,
    string Status,
    string Message,
    int ImageCount,
    int VideoCount,
    long TotalBytes);
