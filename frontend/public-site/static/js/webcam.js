// The ONLY JavaScript file in this project. Keep it small and vanilla — no libraries.
// Responsibility: capture ~4s of webcam video for liveness, submit to the backend,
// then immediately release the camera. Any feature beyond this should be HTMX.

async function startLivenessCapture(targetUrl, onComplete, options = {}) {
    const fieldName = options.fieldName || 'liveness_video';
    const targetId = options.targetId || 'liveness-result';
    const durationMs = options.durationMs || 4000;
    const video = document.getElementById('webcam-preview');
    const resultTarget = document.getElementById(targetId);

    if (resultTarget) {
        resultTarget.innerHTML = '<div class="alert"><span>Recording challenge video...</span></div>';
    }

    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 },
        audio: false,
    });
    video.srcObject = stream;

    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    const chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
        // IMPORTANT: stop the stream immediately after capture.
        stream.getTracks().forEach((t) => t.stop());

        const blob = new Blob(chunks, { type: 'video/webm' });
        const formData = new FormData();
        formData.append(fieldName, blob, 'liveness.webm');

        const response = await fetch(targetUrl, { method: 'POST', body: formData });
        const redirectUrl = response.headers.get('HX-Redirect');
        if (redirectUrl) {
            window.location.href = redirectUrl;
            return;
        }

        const html = await response.text();
        const target = document.getElementById(targetId);
        target.innerHTML = html;
        // Let HTMX process any hx-* attributes in the returned fragment.
        if (window.htmx) window.htmx.process(target);

        if (onComplete) onComplete(response.ok);
    };

    recorder.start();
    setTimeout(() => recorder.stop(), durationMs);
}

window.startLivenessCapture = startLivenessCapture;
