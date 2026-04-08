using Microsoft.AspNetCore.Components;
using System.Net;
using System.Net.Http.Json;
using _4CImageSeg.Web.Client;
using _4CImageSeg.Web.Client.Services;
using _4CImageSeg.Web.Components;
using _4CImageSeg.Contracts;

var builder = WebApplication.CreateBuilder(args);

// Add service defaults & Aspire client integrations.
builder.AddServiceDefaults();

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents()
    .AddInteractiveWebAssemblyComponents();

builder.Services.AddOutputCache();
builder.Services.AddHttpClient(RecognitionApiClient.ClientName, client =>
{
    client.BaseAddress = new Uri("https+http://apiservice");
});
builder.Services.AddScoped(sp =>
{
    var navigationManager = sp.GetRequiredService<NavigationManager>();
    return new HttpClient { BaseAddress = new Uri(navigationManager.BaseUri) };
});
builder.Services.AddScoped<SpaNavigationState>();
builder.Services.AddScoped<RecognitionApiClient>();
builder.Services.AddScoped<RecognitionWorkbenchState>();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseWebAssemblyDebugging();
}
else
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    app.UseHsts();
}

app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);
app.UseHttpsRedirection();

app.UseAntiforgery();

app.UseOutputCache();

app.MapStaticAssets();

var recognitionApi = app.MapGroup("/api/recognition");

recognitionApi.MapGet("/capabilities", async (IHttpClientFactory clientFactory, CancellationToken cancellationToken) =>
{
    var client = clientFactory.CreateClient(RecognitionApiClient.ClientName);
    using var response = await client.GetAsync("/api/recognition/capabilities", cancellationToken);

    if (!response.IsSuccessStatusCode)
    {
        return Results.StatusCode((int)response.StatusCode);
    }

    var payload = await response.Content.ReadFromJsonAsync<RecognitionCapabilitiesDto>(cancellationToken: cancellationToken);
    return payload is null
        ? Results.Problem("无法解析识别能力响应。", statusCode: (int)HttpStatusCode.BadGateway)
        : Results.Ok(payload);
});

recognitionApi.MapGet("/modes", async (IHttpClientFactory clientFactory, CancellationToken cancellationToken) =>
{
    var client = clientFactory.CreateClient(RecognitionApiClient.ClientName);
    using var response = await client.GetAsync("/api/recognition/modes", cancellationToken);

    if (!response.IsSuccessStatusCode)
    {
        return Results.StatusCode((int)response.StatusCode);
    }

    var payload = await response.Content.ReadFromJsonAsync<IReadOnlyList<RecognitionModeDto>>(cancellationToken: cancellationToken);
    return payload is null
        ? Results.Problem("无法解析识别模式响应。", statusCode: (int)HttpStatusCode.BadGateway)
        : Results.Ok(payload);
});

recognitionApi.MapPost("/jobs", async (RecognitionJobRequest request, IHttpClientFactory clientFactory, CancellationToken cancellationToken) =>
{
    var client = clientFactory.CreateClient(RecognitionApiClient.ClientName);
    using var response = await client.PostAsJsonAsync("/api/recognition/jobs", request, cancellationToken);

    if (!response.IsSuccessStatusCode)
    {
        return Results.StatusCode((int)response.StatusCode);
    }

    var payload = await response.Content.ReadFromJsonAsync<RecognitionJobResponse>(cancellationToken: cancellationToken);
    return payload is null
        ? Results.Problem("无法解析识别任务响应。", statusCode: (int)HttpStatusCode.BadGateway)
        : Results.Accepted($"/api/recognition/jobs/{payload.JobId}", payload);
});

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode()
    .AddInteractiveWebAssemblyRenderMode()
    .AddAdditionalAssemblies(typeof(_4CImageSeg.Web.Client._Imports).Assembly);

app.MapDefaultEndpoints();

app.Run();
