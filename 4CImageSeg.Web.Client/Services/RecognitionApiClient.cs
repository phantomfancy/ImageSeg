using System.Net.Http.Json;
using _4CImageSeg.Contracts;

namespace _4CImageSeg.Web.Client.Services;

public sealed class RecognitionApiClient(HttpClient http)
{
    public const string ClientName = "RecognitionApi";

    public async Task<RecognitionCapabilitiesDto?> GetCapabilitiesAsync(CancellationToken cancellationToken = default) =>
        await http.GetFromJsonAsync<RecognitionCapabilitiesDto>("api/recognition/capabilities", cancellationToken);

    public async Task<IReadOnlyList<RecognitionModeDto>> GetModesAsync(CancellationToken cancellationToken = default) =>
        await http.GetFromJsonAsync<IReadOnlyList<RecognitionModeDto>>("api/recognition/modes", cancellationToken)
        ?? [];

    public async Task<RecognitionJobResponse?> SubmitJobAsync(RecognitionJobRequest request, CancellationToken cancellationToken = default)
    {
        using var response = await http.PostAsJsonAsync("api/recognition/jobs", request, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<RecognitionJobResponse>(cancellationToken: cancellationToken);
    }
}
