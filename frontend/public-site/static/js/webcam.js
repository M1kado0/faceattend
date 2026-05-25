// The ONLY JavaScript file in this project. Keep it small and vanilla.
// Responsibility: guide webcam liveness, record video, submit to backend, then release camera.

const USER_CAMERA_CONSTRAINTS = {
    facingMode: 'user',
    width: { ideal: 1280 },
    height: { ideal: 720 },
};

const FACE_LANDMARKER_MODEL_URL =
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const TASKS_VISION_WASM_URL =
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const TASKS_VISION_MODULE_URL =
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest';
const BLINK_CLOSED_THRESHOLD = 0.45;
const BLINK_OPEN_THRESHOLD = 0.25;
const MIN_FACE_HEIGHT_RATIO = 0.14;
const CHALLENGE_COUNTDOWN_SECONDS = 3;
const NO_FACE_MESSAGE = 'Position your face in the oval';
const MOVE_CLOSER_MESSAGE = 'Move closer';

let faceLandmarkerPromise = null;

async function startLivenessCapture(targetUrl, onComplete, options = {}) {
    const fieldName = options.fieldName || 'liveness_video';
    const targetId = options.targetId || 'liveness-result';
    const durationMs = options.durationMs || 6000;
    const waitForFaceMs = options.waitForFaceMs || 10000;
    const sessionSelectId = options.sessionSelectId || null;
    const video = document.getElementById('webcam-preview');
    const resultTarget = document.getElementById(targetId);
    const guidance = createLivenessGuidance();
    let stopped = false;
    let animationFrameId = null;
    let recorder = null;
    let recordingTimeoutId = null;
    let recordingStarted = false;
    let countdownTimerId = null;
    let countdownActive = false;
    const chunks = [];

    if (sessionSelectId) {
        const sessionSelect = document.getElementById(sessionSelectId);
        if (!sessionSelect || !sessionSelect.value) {
            setResult(resultTarget, 'Choose an attendance session first.');
            return;
        }
    }

    guidance.reset();
    guidance.setPhase('face', 'active');
    guidance.show(NO_FACE_MESSAGE, 'info');
    setResult(resultTarget, 'Preparing camera...');

    const stream = await navigator.mediaDevices.getUserMedia({
        video: USER_CAMERA_CONSTRAINTS,
        audio: false,
    });
    video.srcObject = stream;
    await video.play();

    let faceLandmarker = null;
    try {
        setResult(resultTarget, 'Preparing live guidance...');
        faceLandmarker = await getFaceLandmarker();
    } catch (error) {
        console.warn('MediaPipe live guidance unavailable; backend verification still runs.', error);
        guidance.setPhase('face', 'done');
        guidance.setPhase('blink', 'active');
        startRecording();
        guidance.show('Blink twice', 'active');
        return;
    }

    setResult(resultTarget, 'Waiting for face...');
    const waitStartedAt = performance.now();
    const blinkState = createBlinkState();

    const analyzeFrame = () => {
        if (stopped) return;

        const result = faceLandmarker.detectForVideo(video, performance.now());
        const faceLandmarks = result.faceLandmarks || [];
        const blendshapes = result.faceBlendshapes || [];

        if (faceLandmarks.length !== 1) {
            cancelCountdown();
            guidance.setPhase('face', 'active');
            guidance.show(NO_FACE_MESSAGE, 'info');
            checkFaceWaitTimeout();
            animationFrameId = requestAnimationFrame(analyzeFrame);
            return;
        }

        const faceHeightRatio = getFaceHeightRatio(faceLandmarks[0]);
        if (faceHeightRatio < MIN_FACE_HEIGHT_RATIO) {
            cancelCountdown();
            guidance.setPhase('face', 'active');
            guidance.show(MOVE_CLOSER_MESSAGE, 'info');
            checkFaceWaitTimeout();
            animationFrameId = requestAnimationFrame(analyzeFrame);
            return;
        }

        if (!recordingStarted) {
            if (!countdownActive) startCountdown();
            animationFrameId = requestAnimationFrame(analyzeFrame);
            return;
        }

        const scores = getBlendshapeScores(blendshapes[0]);
        const blinkScore = Math.min(
            scores.eyeBlinkLeft || 0,
            scores.eyeBlinkRight || 0,
        );
        const blinkCount = observeBlink(blinkState, blinkScore, performance.now());
        if (blinkCount === 1) {
            guidance.show('Blink 1 detected', 'success');
            setTimeout(() => {
                if (!stopped && blinkState.count === 1) guidance.show('Blink once more', 'active');
            }, 500);
        }
        if (blinkCount >= 2) {
            guidance.setPhase('blink', 'done');
            guidance.show('Blink 2 detected', 'success');
            setTimeout(() => {
                if (!stopped) guidance.show('Challenge complete', 'success');
                stopCapture();
            }, 400);
            return;
        }

        animationFrameId = requestAnimationFrame(analyzeFrame);
    };
    animationFrameId = requestAnimationFrame(analyzeFrame);

    function checkFaceWaitTimeout() {
        if (performance.now() - waitStartedAt < waitForFaceMs) return;
        cancelCountdown();
        guidance.show('Could not find a usable face. Try again.', 'error');
        setResult(resultTarget, 'Position your face in the oval and try again.');
        stopCapture({ upload: false });
    }

    function startCountdown() {
        countdownActive = true;
        guidance.setPhase('face', 'done');
        guidance.show('Face detected', 'success');
        setResult(resultTarget, 'Face detected. Get ready to blink twice.');

        let secondsLeft = CHALLENGE_COUNTDOWN_SECONDS;
        setTimeout(() => {
            if (!countdownActive || stopped || recordingStarted) return;
            guidance.show('Get ready to blink twice', 'active');
        }, 500);

        countdownTimerId = setInterval(() => {
            if (!countdownActive || stopped || recordingStarted) return;

            if (secondsLeft > 0) {
                guidance.show(`Blink twice in ${secondsLeft}`, 'active');
                secondsLeft -= 1;
                return;
            }

            cancelCountdown({ keepMessage: true });
            guidance.setPhase('blink', 'active');
            startRecording();
            guidance.show('Blink twice', 'active');
        }, 1000);
    }

    function cancelCountdown(options = {}) {
        if (!countdownActive) return;
        countdownActive = false;
        if (countdownTimerId !== null) {
            clearInterval(countdownTimerId);
            countdownTimerId = null;
        }
        if (!options.keepMessage && !recordingStarted && !stopped) {
            setResult(resultTarget, 'Waiting for face...');
        }
    }

    function startRecording() {
        if (recordingStarted) return;
        recordingStarted = true;
        recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
        recorder.ondataavailable = (event) => chunks.push(event.data);
        recorder.onstop = async () => {
            if (stopped && chunks.length === 0) return;
            await uploadRecording();
        };
        recorder.start();
        recordingTimeoutId = setTimeout(stopCapture, durationMs);
        setResult(resultTarget, 'Recording challenge video...');
    }

    async function uploadRecording() {
        stream.getTracks().forEach((track) => track.stop());
        guidance.setPhase('upload', 'active');
        guidance.show('Upload starts', 'active');

        const blob = new Blob(chunks, { type: 'video/webm' });
        const formData = new FormData();
        formData.append(fieldName, blob, 'liveness.webm');
        if (sessionSelectId) {
            const sessionSelect = document.getElementById(sessionSelectId);
            if (sessionSelect && sessionSelect.value) {
                formData.append('session_id', sessionSelect.value);
            }
        }

        const response = await fetch(targetUrl, { method: 'POST', body: formData });
        const redirectUrl = response.headers.get('HX-Redirect');
        if (redirectUrl) {
            window.location.href = redirectUrl;
            return;
        }

        const html = await response.text();
        const target = document.getElementById(targetId);
        target.innerHTML = html;
        if (window.htmx) window.htmx.process(target);

        if (onComplete) onComplete(response.ok);
    }

    function stopCapture(options = {}) {
        if (stopped) return;
        stopped = true;
        cancelCountdown({ keepMessage: true });
        if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
        if (recordingTimeoutId !== null) clearTimeout(recordingTimeoutId);

        if (options.upload === false || !recordingStarted || !recorder) {
            stream.getTracks().forEach((track) => track.stop());
            return;
        }

        if (recorder.state !== 'inactive') recorder.stop();
    }
}

window.startLivenessCapture = startLivenessCapture;

function createLivenessGuidance() {
    const root = document.getElementById('liveness-guidance');
    const phases = document.querySelectorAll('[data-liveness-phase]');

    function setFaceGuide(state) {
        const faceGuide = document.getElementById('liveness-face-guide');
        if (!faceGuide) return;
        const classByState = {
            active: 'h-[58%] w-[70%] rounded-[50%] border-4 border-primary shadow-[0_0_0_9999px_rgba(0,0,0,0.42)]',
            done: 'h-[58%] w-[70%] rounded-[50%] border-4 border-success shadow-[0_0_0_9999px_rgba(0,0,0,0.42)]',
            idle: 'h-[58%] w-[70%] rounded-[50%] border-4 border-white/85 shadow-[0_0_0_9999px_rgba(0,0,0,0.42)]',
        };
        faceGuide.className = classByState[state] || classByState.idle;
    }

    return {
        reset() {
            phases.forEach((phase) => {
                phase.className = 'rounded-full bg-white/20 px-3 py-2 text-white/70 backdrop-blur';
            });
            setFaceGuide('idle');
        },
        show(message, state = 'info') {
            if (!root) return;
            const classByState = {
                active: 'rounded-lg bg-primary px-4 py-3 text-center text-sm font-semibold text-primary-content shadow-lg',
                error: 'rounded-lg bg-error px-4 py-3 text-center text-sm font-semibold text-error-content shadow-lg',
                info: 'rounded-lg bg-black/65 px-4 py-3 text-center text-sm font-semibold text-white backdrop-blur',
                success: 'rounded-lg bg-success px-4 py-3 text-center text-sm font-semibold text-success-content shadow-lg',
            };
            root.className = classByState[state] || classByState.info;
            root.textContent = message;
        },
        setPhase(name, state = 'idle') {
            const phase = document.querySelector(`[data-liveness-phase="${name}"]`);
            if (!phase) return;
            const classByState = {
                active: 'rounded-full bg-primary px-3 py-2 text-primary-content shadow',
                done: 'rounded-full bg-success px-3 py-2 text-success-content shadow',
                idle: 'rounded-full bg-white/20 px-3 py-2 text-white/70 backdrop-blur',
            };
            phase.className = classByState[state] || classByState.idle;
            if (name === 'face') setFaceGuide(state);
        },
    };
}

function setResult(target, message) {
    if (!target) return;
    target.innerHTML = `<div class="alert"><span>${message}</span></div>`;
}

async function getFaceLandmarker() {
    if (!faceLandmarkerPromise) {
        faceLandmarkerPromise = (async () => {
            const vision = await import(TASKS_VISION_MODULE_URL);
            const filesetResolver = await vision.FilesetResolver.forVisionTasks(
                TASKS_VISION_WASM_URL,
            );
            return vision.FaceLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: FACE_LANDMARKER_MODEL_URL,
                    delegate: 'GPU',
                },
                runningMode: 'VIDEO',
                numFaces: 1,
                outputFaceBlendshapes: true,
            });
        })();
    }
    return faceLandmarkerPromise;
}

function createBlinkState() {
    return {
        state: 'open',
        count: 0,
        lastBlinkAt: 0,
    };
}

function observeBlink(state, blinkScore, timestampMs) {
    if (state.state === 'open' && blinkScore >= BLINK_CLOSED_THRESHOLD) {
        state.state = 'closed';
        return state.count;
    }

    if (state.state === 'closed' && blinkScore <= BLINK_OPEN_THRESHOLD) {
        if (timestampMs - state.lastBlinkAt > 250) {
            state.count += 1;
            state.lastBlinkAt = timestampMs;
        }
        state.state = 'open';
    }

    return state.count;
}

function getBlendshapeScores(faceBlendshapes) {
    const scores = {};
    const categories = faceBlendshapes ? faceBlendshapes.categories || [] : [];
    categories.forEach((category) => {
        scores[category.categoryName] = category.score;
    });
    return scores;
}

function getFaceHeightRatio(landmarks) {
    if (!landmarks.length) return 0;
    const ys = landmarks.map((landmark) => landmark.y);
    return Math.max(...ys) - Math.min(...ys);
}
