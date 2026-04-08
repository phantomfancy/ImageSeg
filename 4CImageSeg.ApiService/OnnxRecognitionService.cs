using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Options;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using _4CImageSeg.Contracts;

namespace _4CImageSeg.ApiService;

sealed class OnnxRecognitionService(
    IHttpClientFactory httpClientFactory,
    IWebHostEnvironment environment,
    IOptions<OnnxRecognitionOptions> options,
    ILogger<OnnxRecognitionService> logger)
{
    private readonly SemaphoreSlim _initializationLock = new(1, 1);
    private InferenceSession? _session;
    private string? _inputName;
    private string? _logitsOutputName;
    private string? _boxesOutputName;
    private readonly ConcurrentDictionary<int, string> _labelMap = new();

    private OnnxRecognitionOptions Settings => options.Value;

    public async Task<RecognitionExecutionResultDto> RunSingleImageAsync(
        Stream imageStream,
        string inputSource,
        CancellationToken cancellationToken = default)
    {
        var session = await EnsureSessionAsync(cancellationToken);

        using var image = await Image.LoadAsync<Rgb24>(imageStream, cancellationToken);
        var originalWidth = image.Width;
        var originalHeight = image.Height;

        using var resized = image.Clone(ctx => ctx.Resize(Settings.InputWidth, Settings.InputHeight));
        var inputTensor = CreateInputTensor(resized);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName!, inputTensor)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);
        var logits = results.First(item => item.Name == _logitsOutputName).AsTensor<float>();
        var boxes = results.First(item => item.Name == _boxesOutputName).AsTensor<float>();

        var detections = DecodeDetections(logits, boxes, originalWidth, originalHeight);
        var result = new RecognitionResultDto(
            inputSource,
            RecognitionModes.Server,
            Settings.ModelVersion,
            DateTimeOffset.UtcNow,
            detections);

        return new RecognitionExecutionResultDto(result, null);
    }

    private async Task<InferenceSession> EnsureSessionAsync(CancellationToken cancellationToken)
    {
        if (_session is not null)
        {
            return _session;
        }

        await _initializationLock.WaitAsync(cancellationToken);

        try
        {
            if (_session is not null)
            {
                return _session;
            }

            var modelPath = await ResolveModelPathAsync(cancellationToken);
            var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions();
            _session = new InferenceSession(modelPath, sessionOptions);

            _inputName = _session.InputMetadata.Keys.First();
            _logitsOutputName = ResolveOutputName(_session.OutputMetadata.Keys, "logits");
            _boxesOutputName = ResolveOutputName(_session.OutputMetadata.Keys, "pred_boxes");

            logger.LogInformation("已加载 ONNX 模型：{ModelPath}", modelPath);
            return _session;
        }
        finally
        {
            _initializationLock.Release();
        }
    }

    private async Task<string> ResolveModelPathAsync(CancellationToken cancellationToken)
    {
        if (!string.IsNullOrWhiteSpace(Settings.ModelPath))
        {
            var candidate = Path.IsPathRooted(Settings.ModelPath)
                ? Settings.ModelPath
                : Path.GetFullPath(Path.Combine(environment.ContentRootPath, Settings.ModelPath));

            if (File.Exists(candidate))
            {
                return candidate;
            }

            throw new FileNotFoundException($"未找到配置的 ONNX 模型文件：{candidate}");
        }

        if (string.IsNullOrWhiteSpace(Settings.ModelUri))
        {
            throw new InvalidOperationException("未配置 ONNX 模型路径或模型下载地址。");
        }

        var cachePath = Path.GetFullPath(Path.Combine(environment.ContentRootPath, Settings.ModelCachePath));
        if (File.Exists(cachePath))
        {
            return cachePath;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);

        using var client = httpClientFactory.CreateClient();
        using var response = await client.GetAsync(Settings.ModelUri, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        await using var networkStream = await response.Content.ReadAsStreamAsync(cancellationToken);
        await using var fileStream = File.Create(cachePath);
        await networkStream.CopyToAsync(fileStream, cancellationToken);

        return cachePath;
    }

    private IReadOnlyList<DetectionDto> DecodeDetections(
        Tensor<float> logits,
        Tensor<float> boxes,
        int imageWidth,
        int imageHeight)
    {
        var queryCount = logits.Dimensions[^2];
        var classCountWithNoObject = logits.Dimensions[^1];
        var classCount = classCountWithNoObject - 1;
        var detections = new List<DetectionDto>();

        for (var queryIndex = 0; queryIndex < queryCount; queryIndex++)
        {
            var bestClassIndex = 0;
            var bestScore = 0f;
            var expValues = new float[classCountWithNoObject];
            var maxLogit = float.NegativeInfinity;

            for (var classIndex = 0; classIndex < classCountWithNoObject; classIndex++)
            {
                var current = logits[0, queryIndex, classIndex];
                if (current > maxLogit)
                {
                    maxLogit = current;
                }
            }

            float sum = 0f;
            for (var classIndex = 0; classIndex < classCountWithNoObject; classIndex++)
            {
                var value = MathF.Exp(logits[0, queryIndex, classIndex] - maxLogit);
                expValues[classIndex] = value;
                sum += value;
            }

            for (var classIndex = 0; classIndex < classCount; classIndex++)
            {
                var probability = expValues[classIndex] / sum;
                if (probability > bestScore)
                {
                    bestScore = probability;
                    bestClassIndex = classIndex;
                }
            }

            if (bestScore < Settings.ScoreThreshold)
            {
                continue;
            }

            var cx = boxes[0, queryIndex, 0];
            var cy = boxes[0, queryIndex, 1];
            var width = boxes[0, queryIndex, 2];
            var height = boxes[0, queryIndex, 3];

            var left = Clamp((cx - width / 2f) * imageWidth, 0f, imageWidth);
            var top = Clamp((cy - height / 2f) * imageHeight, 0f, imageHeight);
            var right = Clamp((cx + width / 2f) * imageWidth, 0f, imageWidth);
            var bottom = Clamp((cy + height / 2f) * imageHeight, 0f, imageHeight);

            detections.Add(new DetectionDto(
                ResolveLabel(bestClassIndex),
                bestScore,
                new DetectionBoxDto(
                    left,
                    top,
                    MathF.Max(0f, right - left),
                    MathF.Max(0f, bottom - top))));
        }

        return detections
            .OrderByDescending(item => item.Confidence)
            .Take(Settings.MaxDetections)
            .ToArray();
    }

    private DenseTensor<float> CreateInputTensor(Image<Rgb24> image)
    {
        var tensor = new DenseTensor<float>([1, 3, Settings.InputHeight, Settings.InputWidth]);

        for (var y = 0; y < Settings.InputHeight; y++)
        {
            for (var x = 0; x < Settings.InputWidth; x++)
            {
                var pixel = image[x, y];
                tensor[0, 0, y, x] = pixel.R / 255f;
                tensor[0, 1, y, x] = pixel.G / 255f;
                tensor[0, 2, y, x] = pixel.B / 255f;
            }
        }

        return tensor;
    }

    private string ResolveOutputName(IEnumerable<string> candidates, string expected)
    {
        var exact = candidates.FirstOrDefault(item => string.Equals(item, expected, StringComparison.OrdinalIgnoreCase));
        if (!string.IsNullOrWhiteSpace(exact))
        {
            return exact;
        }

        var partial = candidates.FirstOrDefault(item => item.Contains(expected, StringComparison.OrdinalIgnoreCase));
        return partial
            ?? throw new InvalidOperationException($"模型输出中未找到 {expected}。");
    }

    private string ResolveLabel(int classId)
    {
        return _labelMap.GetOrAdd(classId, static key => key switch
        {
            0 => "person",
            1 => "bicycle",
            2 => "car",
            3 => "motorcycle",
            4 => "airplane",
            5 => "bus",
            6 => "train",
            7 => "truck",
            8 => "boat",
            9 => "traffic light",
            10 => "fire hydrant",
            11 => "stop sign",
            12 => "parking meter",
            13 => "bench",
            14 => "bird",
            15 => "cat",
            16 => "dog",
            17 => "horse",
            18 => "sheep",
            19 => "cow",
            20 => "elephant",
            21 => "bear",
            22 => "zebra",
            23 => "giraffe",
            24 => "backpack",
            25 => "umbrella",
            26 => "handbag",
            27 => "tie",
            28 => "suitcase",
            29 => "frisbee",
            30 => "skis",
            31 => "snowboard",
            32 => "sports ball",
            33 => "kite",
            34 => "baseball bat",
            35 => "baseball glove",
            36 => "skateboard",
            37 => "surfboard",
            38 => "tennis racket",
            39 => "bottle",
            40 => "wine glass",
            41 => "cup",
            42 => "fork",
            43 => "knife",
            44 => "spoon",
            45 => "bowl",
            46 => "banana",
            47 => "apple",
            48 => "sandwich",
            49 => "orange",
            50 => "broccoli",
            51 => "carrot",
            52 => "hot dog",
            53 => "pizza",
            54 => "donut",
            55 => "cake",
            56 => "chair",
            57 => "couch",
            58 => "potted plant",
            59 => "bed",
            60 => "dining table",
            61 => "toilet",
            62 => "tv",
            63 => "laptop",
            64 => "mouse",
            65 => "remote",
            66 => "keyboard",
            67 => "cell phone",
            68 => "microwave",
            69 => "oven",
            70 => "toaster",
            71 => "sink",
            72 => "refrigerator",
            73 => "book",
            74 => "clock",
            75 => "vase",
            76 => "scissors",
            77 => "teddy bear",
            78 => "hair drier",
            79 => "toothbrush",
            _ => $"class-{key}"
        });
    }

    private static float Clamp(float value, float min, float max) => MathF.Min(MathF.Max(value, min), max);
}
