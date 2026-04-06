let activeStream;

export async function listVideoDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();

    return devices
        .filter(device => device.kind === "videoinput")
        .map((device, index) => ({
            deviceId: device.deviceId,
            label: device.label || `摄像头 ${index + 1}`
        }));
}

export async function startPreview(videoElement, deviceId) {
    await stopPreview(videoElement);

    const constraints = deviceId
        ? { video: { deviceId: { exact: deviceId } }, audio: false }
        : { video: true, audio: false };

    activeStream = await navigator.mediaDevices.getUserMedia(constraints);
    videoElement.srcObject = activeStream;
    await videoElement.play();
}

export async function stopPreview(videoElement) {
    if (activeStream) {
        activeStream.getTracks().forEach(track => track.stop());
        activeStream = null;
    }

    if (videoElement) {
        videoElement.pause();
        videoElement.srcObject = null;
    }
}
