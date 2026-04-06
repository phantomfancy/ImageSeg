var builder = DistributedApplication.CreateBuilder(args);

var apiService = builder.AddProject<Projects._4CImageSeg_ApiService>("apiservice")
    .WithHttpHealthCheck("/health");

builder.AddProject<Projects._4CImageSeg_Web>("webfrontend")
    .WithExternalHttpEndpoints()
    .WithHttpHealthCheck("/health")
    .WithReference(apiService)
    .WaitFor(apiService);

builder.Build().Run();
