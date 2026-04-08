namespace _4CImageSeg.ApiService;

sealed class OnnxRecognitionOptions
{
    public string? ModelPath { get; set; }
    public string? ModelUri { get; set; }
    public string ModelCachePath { get; set; } = "App_Data/Models/yolo26s/model.onnx";
    public string ModelVersion { get; set; } = "onnx-community/yolo26s-ONNX";
    public int InputWidth { get; set; } = 640;
    public int InputHeight { get; set; } = 640;
    public float ScoreThreshold { get; set; } = 0.35f;
    public int MaxDetections { get; set; } = 20;
    public long MaxUploadBytes { get; set; } = 15 * 1024 * 1024;
}
