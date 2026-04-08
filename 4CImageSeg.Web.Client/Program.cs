using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using _4CImageSeg.Web.Client.Services;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddScoped(_ => new HttpClient
{
    BaseAddress = new Uri(builder.HostEnvironment.BaseAddress)
});
builder.Services.AddScoped<SpaNavigationState>();
builder.Services.AddScoped<RecognitionApiClient>();
builder.Services.AddScoped<RecognitionWorkbenchState>();

await builder.Build().RunAsync();
