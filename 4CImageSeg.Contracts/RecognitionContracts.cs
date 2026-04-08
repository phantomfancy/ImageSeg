namespace _4CImageSeg.Contracts;

public static class RecognitionModes
{
    public const string Auto = "auto";
    public const string Local = "local";
    public const string Server = "server";

    public static string Normalize(string? value) =>
        string.IsNullOrWhiteSpace(value) ? Auto : value.Trim().ToLowerInvariant();
}

public static class RecognitionJobStatuses
{
    public const string Queued = "queued";
    public const string Completed = "completed";
    public const string Failed = "failed";
}

public sealed record RecognitionCapabilitiesDto(
    bool SupportsLocalInference,
    bool SupportsServerInference,
    bool SupportsImageExport,
    bool SupportsVideoExport,
    string DefaultMode,
    IReadOnlyList<string> Notes);

public sealed record RecognitionModeDto(string Value, string Label, bool Enabled);

public sealed record RecognitionJobRequest(
    string Mode,
    int ImageCount,
    int VideoCount,
    long TotalBytes);

public sealed record RecognitionJobResponse(
    string JobId,
    string Mode,
    string Status,
    string Message,
    int ImageCount,
    int VideoCount,
    long TotalBytes);

public sealed record DetectionBoxDto(
    float X,
    float Y,
    float Width,
    float Height);

public sealed record DetectionDto(
    string Label,
    float Confidence,
    DetectionBoxDto Box);

public sealed record RecognitionResultDto(
    string InputSource,
    string Mode,
    string ModelVersion,
    DateTimeOffset DetectedAtUtc,
    IReadOnlyList<DetectionDto> Detections);

public sealed record RecognitionExecutionResultDto(
    RecognitionResultDto Result,
    string? AnnotatedImageDataUrl);
