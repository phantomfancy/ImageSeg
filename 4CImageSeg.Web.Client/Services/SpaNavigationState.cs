namespace _4CImageSeg.Web.Client.Services;

public sealed class SpaNavigationState
{
    public event Action? Changed;

    public AppView CurrentView { get; private set; } = AppView.Overview;

    public void NavigateTo(AppView view)
    {
        if (CurrentView == view)
        {
            return;
        }

        CurrentView = view;
        Changed?.Invoke();
    }
}

public enum AppView
{
    Overview,
    Intake,
    Detect,
    Results
}
