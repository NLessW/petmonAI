// ONNX Runtime 로그 레벨 설정 (경고 숨기기)
setTimeout(() => {
    if (typeof ort !== 'undefined') {
        ort.env.logLevel = 'error'; // 'verbose', 'info', 'warning', 'error', 'fatal'
        console.log('ONNX Runtime 로그 레벨: error로 설정됨');
    }
}, 100);

// CPU vendor 경고 메시지 필터링
const originalWarn = console.warn;
console.warn = function (...args) {
    const message = args.join(' ');
    if (message.includes('Unknown CPU vendor') || message.includes('cpuinfo_vendor')) {
        return; // CPU vendor 경고 무시
    }
    originalWarn.apply(console, args);
};

let model = null,
    metadata = null,
    webcamStream = null,
    captures = [],
    results = [],
    evaluations = [],
    chart = null,
    analyzedCaptures = [];
let metadataFile = null,
    modelFile = null,
    weightsFile = null;
let classesFile = null,
    onnxModelFile = null;
let onnxDataFile = null;
let modelType = 'teachable'; // 'teachable' or 'onnx'
let onnxSession = null;
let onnxInputSize = 640; // ONNX 모델의 입력 크기 (동적으로 설정됨)
let currentAspectRatio = '1:1';
let currentResolution = 720;

// ONNX 입력 크기 선택 처리
document.getElementById('onnx-input-size')?.addEventListener('change', (e) => {
    onnxInputSize = parseInt(e.target.value);
    console.log(`ONNX 입력 크기 설정: ${onnxInputSize}x${onnxInputSize}`);
});

function getResolutionDimensions() {
    const aspectRatio = currentAspectRatio;
    const resolution = currentResolution;

    let width, height;

    if (aspectRatio === '1:1') {
        width = height = resolution;
    } else if (aspectRatio === '4:3') {
        height = resolution;
        width = Math.round((resolution * 4) / 3);
    } else if (aspectRatio === '16:9') {
        height = resolution;
        width = Math.round((resolution * 16) / 9);
    }

    return { width, height };
}

function updateWebcamStyle() {
    const webcam = document.getElementById('webcam');
    const dims = getResolutionDimensions();
    const ratio = dims.width / dims.height;
    webcam.style.aspectRatio = ratio.toFixed(4);
    webcam.style.maxWidth = dims.width + 'px';
}

function updateCaptureStyle() {
    const dims = getResolutionDimensions();
    const ratio = dims.width / dims.height;
    const style = document.createElement('style');
    style.id = 'dynamic-aspect-ratio';
    const existingStyle = document.getElementById('dynamic-aspect-ratio');
    if (existingStyle) existingStyle.remove();

    style.textContent = `
        .capture-item { aspect-ratio: ${ratio.toFixed(4)}; }
        .eval-card img { aspect-ratio: ${ratio.toFixed(4)}; }
    `;
    document.head.appendChild(style);
}

document.getElementById('aspect-ratio').addEventListener('change', (e) => {
    currentAspectRatio = e.target.value;
    updateWebcamStyle();
    updateCaptureStyle();

    // 웹캠이 실행 중이면 재시작
    if (webcamStream) {
        restartWebcam();
    }
});

document.getElementById('resolution').addEventListener('change', (e) => {
    currentResolution = parseInt(e.target.value);
    updateWebcamStyle();
    updateCaptureStyle();

    // 웹캠이 실행 중이면 재시작
    if (webcamStream) {
        restartWebcam();
    }
});

async function restartWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
        webcamStream = null;
    }

    try {
        const dims = getResolutionDimensions();
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: dims.width, height: dims.height, facingMode: 'environment' },
        });
        document.getElementById('webcam').srcObject = webcamStream;
    } catch (error) {
        alert('웹캠 재시작 오류: ' + error.message);
        document.getElementById('start-webcam-btn').disabled = false;
        document.getElementById('start-webcam-btn').textContent = '웹캠 시작';
    }
}

// 초기 스타일 설정
updateWebcamStyle();
updateCaptureStyle();

document.getElementById('model-type').addEventListener('change', (e) => {
    modelType = e.target.value;
    const teachableInputs = document.getElementById('teachable-inputs');
    const onnxInputs = document.getElementById('onnx-inputs');

    if (modelType === 'teachable') {
        teachableInputs.style.display = 'flex';
        onnxInputs.style.display = 'none';
    } else {
        teachableInputs.style.display = 'none';
        onnxInputs.style.display = 'flex';
    }

    // 파일 초기화
    metadataFile = null;
    modelFile = null;
    weightsFile = null;
    classesFile = null;
    onnxModelFile = null;
    onnxDataFile = null;
    model = null;
    onnxSession = null;
    metadata = null;

    // 상태 표시 초기화
    document.getElementById('metadata-status').textContent = '❌';
    document.getElementById('model-status').textContent = '❌';
    document.getElementById('weights-status').textContent = '❌';
    document.getElementById('classes-status').textContent = '❌';
    document.getElementById('onnx-status').textContent = '❌';
    document.getElementById('onnx-data-status').textContent = '⚪';
    document.getElementById('model-status-msg').textContent = '';

    checkFilesReady();
});

document.getElementById('metadata').addEventListener('change', (e) => {
    metadataFile = e.target.files[0];
    document.getElementById('metadata-status').textContent = metadataFile ? '✅' : '❌';
    checkFilesReady();
});
document.getElementById('model').addEventListener('change', (e) => {
    modelFile = e.target.files[0];
    document.getElementById('model-status').textContent = modelFile ? '✅' : '❌';
    checkFilesReady();
});
document.getElementById('weights').addEventListener('change', (e) => {
    weightsFile = e.target.files[0];
    document.getElementById('weights-status').textContent = weightsFile ? '✅' : '❌';
    checkFilesReady();
});
document.getElementById('classes').addEventListener('change', (e) => {
    classesFile = e.target.files[0];
    document.getElementById('classes-status').textContent = classesFile ? '✅' : '❌';
    checkFilesReady();
});
document.getElementById('onnx-model').addEventListener('change', (e) => {
    onnxModelFile = e.target.files[0];
    document.getElementById('onnx-status').textContent = onnxModelFile ? '✅' : '❌';
    checkFilesReady();
});
document.getElementById('onnx-data').addEventListener('change', (e) => {
    onnxDataFile = e.target.files[0];
    document.getElementById('onnx-data-status').textContent = onnxDataFile ? '✅' : '⚪';
});

function checkFilesReady() {
    if (modelType === 'teachable') {
        document.getElementById('load-model-btn').disabled = !(metadataFile && modelFile && weightsFile);
    } else {
        document.getElementById('load-model-btn').disabled = !(classesFile && onnxModelFile);
    }
}

document.getElementById('load-model-btn').addEventListener('click', async () => {
    const statusMsg = document.getElementById('model-status-msg');
    statusMsg.textContent = '모델 로딩 중...';
    try {
        if (modelType === 'teachable') {
            await loadTeachableModel(statusMsg);
        } else {
            await loadOnnxModel(statusMsg);
        }
        document.getElementById('start-webcam-btn').disabled = false;
        displayClassLabels();
    } catch (error) {
        console.error(error);
        statusMsg.textContent = '❌ 모델 로드 실패: ' + error.message;
    }
});

async function loadTeachableModel(statusMsg) {
    metadata = JSON.parse(await metadataFile.text());
    const modelJson = JSON.parse(await modelFile.text());
    const customIOHandler = {
        load: async () => ({
            modelTopology: modelJson.modelTopology,
            weightSpecs: modelJson.weightsManifest[0].weights,
            weightData: await weightsFile.arrayBuffer(),
            format: modelJson.format,
            generatedBy: modelJson.generatedBy,
            convertedBy: modelJson.convertedBy,
        }),
    };
    model = await tf.loadLayersModel(customIOHandler);
    statusMsg.textContent = '✅ Teachable Machine 모델 로드 완료! 클래스: ' + metadata.labels.join(', ');
}

async function loadOnnxModel(statusMsg) {
    // classes.txt 읽기
    const classesText = await classesFile.text();
    const labels = classesText
        .trim()
        .split('\n')
        .map((l) => l.trim())
        .filter((l) => l);
    metadata = { labels: labels };

    // ONNX 모델 로드
    const arrayBuffer = await onnxModelFile.arrayBuffer();

    // 세션 옵션 (로그 레벨 설정)
    const sessionOptions = {
        logSeverityLevel: 3, // 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
    };

    // 외부 데이터 파일이 있는 경우
    if (onnxDataFile) {
        statusMsg.textContent = 'ONNX 모델 + 외부 데이터 로딩 중...';
        const dataArrayBuffer = await onnxDataFile.arrayBuffer();

        // 외부 데이터를 처리하기 위한 옵션 설정
        onnxSession = await ort.InferenceSession.create(arrayBuffer, {
            ...sessionOptions,
            externalData: [
                {
                    data: new Uint8Array(dataArrayBuffer),
                    path: onnxDataFile.name,
                },
            ],
        });
    } else {
        onnxSession = await ort.InferenceSession.create(arrayBuffer, sessionOptions);
    }

    // 모델의 입력 크기 감지 (안전하게 처리)
    try {
        const inputNames = onnxSession.inputNames;
        if (inputNames && inputNames.length > 0) {
            const inputName = inputNames[0];
            // ONNX Runtime Web에서 입력 shape 정보 가져오기
            const inputMetadata = onnxSession._inputs || onnxSession.inputs;
            if (inputMetadata && inputMetadata[inputName]) {
                const inputShape = inputMetadata[inputName].dims || inputMetadata[inputName].shape;
                // 입력 형식: [batch, channel, height, width] 가정
                if (inputShape && inputShape.length >= 4) {
                    onnxInputSize = inputShape[2]; // height
                    console.log(`ONNX 모델 입력 크기 감지: ${onnxInputSize}x${inputShape[3]}`);
                }
            }
        }
    } catch (e) {
        console.warn('입력 크기를 자동 감지할 수 없습니다. 기본값 사용:', onnxInputSize, e);
    }

    statusMsg.textContent = `✅ ONNX 모델 로드 완료! 입력: ${onnxInputSize}x${onnxInputSize}, 클래스: ${labels.join(', ')}`;
}

function displayClassLabels() {
    const container = document.getElementById('class-labels');
    container.innerHTML = '';
    metadata.labels.forEach((label) => {
        const item = document.createElement('span');
        item.className = 'class-label-item';
        item.textContent = label;
        container.appendChild(item);
    });
}

document.getElementById('start-webcam-btn').addEventListener('click', async () => {
    try {
        const dims = getResolutionDimensions();
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: dims.width, height: dims.height, facingMode: 'environment' },
        });
        document.getElementById('webcam').srcObject = webcamStream;
        document.getElementById('start-test-btn').disabled = false;
        document.getElementById('start-webcam-btn').textContent = '웹캠 실행중';
        document.getElementById('start-webcam-btn').disabled = true;
    } catch (error) {
        alert('웹캠 오류: ' + error.message);
    }
});

let isCapturing = false;
let captureInterval = null;

document.getElementById('start-test-btn').addEventListener('click', async () => {
    if (isCapturing) return;
    if (captures.length >= 10) {
        runAnalysis();
        return;
    }

    isCapturing = true;
    const btn = document.getElementById('start-test-btn');
    btn.textContent = '📷 촬영 중...';

    captureInterval = setInterval(() => {
        if (captures.length >= 10) {
            clearInterval(captureInterval);
            isCapturing = false;
            btn.textContent = '🔍 분석 시작';
            return;
        }
        capturePhoto();
    }, 500);
});

function capturePhoto() {
    const canvas = document.getElementById('capture-canvas'),
        webcam = document.getElementById('webcam');

    // 웹캠 전체 화면을 캡처 (정사각형)
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    canvas.getContext('2d').drawImage(webcam, 0, 0);
    captures.push(canvas.toDataURL('image/jpeg'));
    updateCapturesGrid();
    updateTestButton();
}

function updateTestButton() {
    const btn = document.getElementById('start-test-btn');
    if (captures.length >= 10 && !isCapturing) {
        btn.textContent = '🔍 분석 시작';
    } else if (isCapturing) {
        btn.textContent = '📷 촬영 중... (' + captures.length + '/10)';
    } else {
        btn.textContent = '🔍 판독 시작 (' + captures.length + '/10)';
    }
}

async function runAnalysis() {
    if ((modelType === 'teachable' && !model) || (modelType === 'onnx' && !onnxSession) || captures.length === 0)
        return;
    const btn = document.getElementById('start-test-btn');
    btn.textContent = '분석 중...';
    btn.disabled = true;
    results = [];
    evaluations = new Array(captures.length).fill(null);
    analyzedCaptures = [...captures];
    for (let i = 0; i < captures.length; i++) {
        if (modelType === 'teachable') {
            results.push(await predictImageTeachable(captures[i]));
        } else {
            results.push(await predictImageOnnx(captures[i]));
        }
    }
    displayResults();

    // 분석 완료 후 초기화하여 다시 테스트 가능하게
    captures = [];
    updateCapturesGrid();
    btn.textContent = '🔍 판독 시작 (0/10)';
    btn.disabled = false;
}

function updateCapturesGrid() {
    const grid = document.getElementById('captures-grid');
    grid.innerHTML = '';
    captures.forEach((img, i) => {
        const item = document.createElement('div');
        item.className = 'capture-item';
        item.innerHTML =
            '<img src="' + img + '"><button class="remove-btn" onclick="removeCapture(' + i + ')">×</button>';
        grid.appendChild(item);
    });
}
function removeCapture(i) {
    captures.splice(i, 1);
    updateCapturesGrid();
    updateTestButton();
}

async function predictImageTeachable(imageData) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = async () => {
            const canvas = document.createElement('canvas');
            canvas.width = 224;
            canvas.height = 224;
            const ctx = canvas.getContext('2d');

            // 검은 배경으로 채우기
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, 224, 224);

            // 비율을 유지하며 letterbox 처리
            const scale = Math.min(224 / img.width, 224 / img.height);
            const scaledWidth = img.width * scale;
            const scaledHeight = img.height * scale;
            const x = (224 - scaledWidth) / 2;
            const y = (224 - scaledHeight) / 2;

            ctx.drawImage(img, x, y, scaledWidth, scaledHeight);

            const tensor = tf.browser.fromPixels(canvas).toFloat().div(127.5).sub(1).expandDims();
            const predictions = await model.predict(tensor).data();
            tensor.dispose();
            const maxIndex = predictions.indexOf(Math.max(...predictions));
            const confidence = predictions[maxIndex];
            const label = metadata.labels[maxIndex] || `Unknown-${maxIndex}`;

            console.log(
                `Teachable 예측 - Index: ${maxIndex}, Label: ${label}, Confidence: ${confidence.toFixed(3)}, Total Classes: ${metadata.labels.length}`,
            );

            resolve({
                label: label,
                confidence: confidence,
                allPredictions: Array.from(predictions),
            });
        };
        img.src = imageData;
    });
}

async function predictImageOnnx(imageData) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = async () => {
            const size = onnxInputSize;
            const canvas = document.createElement('canvas');
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext('2d');

            // 검은 배경으로 채우기
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, size, size);

            // 비율을 유지하며 letterbox 처리
            const scale = Math.min(size / img.width, size / img.height);
            const scaledWidth = img.width * scale;
            const scaledHeight = img.height * scale;
            const x = (size - scaledWidth) / 2;
            const y = (size - scaledHeight) / 2;

            ctx.drawImage(img, x, y, scaledWidth, scaledHeight);

            // ImageData를 가져와서 ONNX 입력 형식으로 변환
            const imageData = ctx.getImageData(0, 0, size, size);
            const inputTensor = preprocessImageForOnnx(imageData);

            try {
                // ONNX 추론 - 입력 이름을 동적으로 가져오기
                const inputName = onnxSession.inputNames[0];
                const feeds = {};
                feeds[inputName] = inputTensor;

                const results = await onnxSession.run(feeds);

                // 출력 이름을 동적으로 가져오기
                const outputName = onnxSession.outputNames[0];
                const output = results[outputName];
                const predictions = output.data;

                console.log(`원본 출력 형태:`, output.dims);
                console.log(`원본 출력 크기:`, predictions.length);
                console.log(`클래스 수:`, metadata.labels.length);

                // YOLO 모델인지 확인 (출력이 [1, num_boxes, num_features] 또는 [1, num_features, num_boxes] 형태)
                let probabilities;

                if (output.dims.length === 3) {
                    // YOLO 객체 감지 모델 (예: [1, 84, 8400] 또는 [1, 8400, 85])
                    console.log('YOLO 객체 감지 모델 감지됨');
                    const [batch, dim1, dim2] = output.dims;

                    // YOLOv8 형식: [1, 84, 8400] - 클래스가 앞에
                    // YOLOv5 형식: [1, 8400, 85] - 클래스가 뒤에
                    let numClasses = metadata.labels.length;
                    let isYoloV8 = dim1 === 4 + numClasses; // [1, 84, 8400]
                    let isYoloV5 = dim2 === 5 + numClasses; // [1, 8400, 85]

                    if (isYoloV8) {
                        console.log('YOLOv8 형식 처리');
                        // [1, 84, 8400] -> 각 8400개 박스에서 최고 신뢰도 찾기
                        const numBoxes = dim2;
                        const classScores = new Array(numClasses).fill(0);

                        for (let box = 0; box < numBoxes; box++) {
                            // 각 박스의 클래스 점수 (인덱스 4부터 시작)
                            for (let cls = 0; cls < numClasses; cls++) {
                                const idx = (4 + cls) * numBoxes + box;
                                classScores[cls] = Math.max(classScores[cls], predictions[idx]);
                            }
                        }

                        probabilities = classScores;
                    } else if (isYoloV5) {
                        console.log('YOLOv5 형식 처리');
                        // [1, 8400, 85] -> 각 박스에서 최고 신뢰도 찾기
                        const numBoxes = dim1;
                        const numFeatures = dim2;
                        const classScores = new Array(numClasses).fill(0);

                        for (let box = 0; box < numBoxes; box++) {
                            const objectness = predictions[box * numFeatures + 4];
                            for (let cls = 0; cls < numClasses; cls++) {
                                const score = predictions[box * numFeatures + 5 + cls] * objectness;
                                classScores[cls] = Math.max(classScores[cls], score);
                            }
                        }

                        probabilities = classScores;
                    } else {
                        console.warn('알 수 없는 YOLO 형식, 원본 데이터 사용');
                        probabilities = Array.from(predictions).slice(0, numClasses);
                    }

                    // 정규화 (합이 1이 되도록)
                    const sum = probabilities.reduce((a, b) => a + b, 0);
                    if (sum > 0) {
                        probabilities = probabilities.map((p) => p / sum);
                    }
                } else {
                    // 일반 분류 모델
                    console.log('일반 분류 모델로 처리');
                    // Softmax 적용
                    const expScores = Array.from(predictions)
                        .slice(0, metadata.labels.length)
                        .map((x) => Math.exp(x));
                    const sumExp = expScores.reduce((a, b) => a + b, 0);
                    probabilities = expScores.map((x) => x / sumExp);
                }

                const maxIndex = probabilities.indexOf(Math.max(...probabilities));
                const confidence = probabilities[maxIndex];
                const label = metadata.labels[maxIndex] || `Unknown-${maxIndex}`;

                console.log(`ONNX 예측 - Index: ${maxIndex}, Label: ${label}, Confidence: ${confidence.toFixed(3)}`);
                console.log(
                    `모든 확률:`,
                    probabilities.map((p, i) => `${metadata.labels[i]}: ${(p * 100).toFixed(1)}%`).join(', '),
                );

                resolve({
                    label: label,
                    confidence: confidence,
                    allPredictions: probabilities,
                });
            } catch (error) {
                // 오류 메시지에서 필요한 입력 크기를 파싱
                const errorMsg = error.message || error.toString();
                const match = errorMsg.match(/Expected: (\d+)/);
                if (match) {
                    const expectedSize = parseInt(match[1]);
                    console.warn(`입력 크기 오류 감지. ${onnxInputSize} → ${expectedSize}로 변경 필요`);
                    alert(
                        `모델 입력 크기가 ${expectedSize}x${expectedSize}여야 합니다. 입력 크기를 ${expectedSize}로 변경하고 다시 시도하세요.`,
                    );
                }
                throw error;
            }
        };
        img.src = imageData;
    });
}

function preprocessImageForOnnx(imageData) {
    // ImageData를 [1, 3, size, size] 형식의 Float32Array로 변환
    // 정규화: (pixel / 255.0 - mean) / std
    // ImageNet 기준: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    const size = onnxInputSize;
    const data = imageData.data;
    const float32Data = new Float32Array(1 * 3 * size * size);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const idx = (i * size + j) * 4;
            // R, G, B 채널 분리 및 정규화
            float32Data[0 * size * size + i * size + j] = (data[idx] / 255.0 - mean[0]) / std[0];
            float32Data[1 * size * size + i * size + j] = (data[idx + 1] / 255.0 - mean[1]) / std[1];
            float32Data[2 * size * size + i * size + j] = (data[idx + 2] / 255.0 - mean[2]) / std[2];
        }
    }

    return new ort.Tensor('float32', float32Data, [1, 3, size, size]);
}

function displayResults() {
    document.getElementById('results-section').style.display = 'block';
    const evalContainer = document.getElementById('evaluation-container');
    evalContainer.innerHTML = '';
    results.forEach((r, i) => {
        const card = document.createElement('div');
        card.className = 'eval-card';
        card.innerHTML =
            '<img src="' +
            analyzedCaptures[i] +
            '"><div class="prediction">예측: ' +
            r.label +
            '</div><div class="confidence">신뢰도: ' +
            (r.confidence * 100).toFixed(1) +
            '%</div><div class="eval-buttons"><button class="correct-btn" onclick="markResult(' +
            i +
            ',true)">✅ 맞음</button><button class="wrong-btn" onclick="markResult(' +
            i +
            ',false)">❌ 틀림</button><button class="normal-btn" onclick="markResult(' +
            i +
            ",'normal')\">⚪ 해당없음</button></div>";
        evalContainer.appendChild(card);
    });
    updateTable();
    updateChart();
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

function markResult(i, result) {
    if (result === false) {
        showClassSelector(i);
        return;
    }
    evaluations[i] = result;
    const cards = document.querySelectorAll('.eval-card'),
        btns = cards[i].querySelectorAll('.eval-buttons button');
    btns.forEach((b) => b.classList.remove('selected'));
    if (result === true) btns[0].classList.add('selected');
    else if (result === 'normal') btns[2].classList.add('selected');
    hideClassSelector(i);
    updateTable();
    updateChart();
    updateFinalAccuracy();
}

function showClassSelector(i) {
    const cards = document.querySelectorAll('.eval-card');
    const card = cards[i];

    // 이미 선택기가 있으면 제거
    const existing = card.querySelector('.class-selector');
    if (existing) existing.remove();

    const r = results[i];
    let selectorHtml = '<div class="class-selector"><div class="selector-title">실제 클래스 선택:</div>';

    metadata.labels.forEach((label, idx) => {
        // 예측 결과 배열 크기보다 작은 인덱스만 표시
        if (idx < r.allPredictions.length) {
            const conf = (r.allPredictions[idx] * 100).toFixed(1);
            selectorHtml +=
                '<div class="class-option" onclick="selectActualClass(' +
                i +
                ",'" +
                label +
                '\')">' +
                '<span class="class-name">' +
                label +
                '</span>' +
                '<div class="conf-bar-bg"><div class="conf-bar" style="width:' +
                conf +
                '%"></div></div>' +
                '<span class="conf-value">' +
                conf +
                '%</span></div>';
        }
    });

    selectorHtml += '</div>';

    card.insertAdjacentHTML('beforeend', selectorHtml);

    // 틀림 버튼 선택 표시
    const btns = card.querySelectorAll('.eval-buttons button');
    btns.forEach((b) => b.classList.remove('selected'));
    btns[1].classList.add('selected');
}

function hideClassSelector(i) {
    const cards = document.querySelectorAll('.eval-card');
    const existing = cards[i].querySelector('.class-selector');
    if (existing) existing.remove();
}

function selectActualClass(i, actualClass) {
    evaluations[i] = { correct: false, actualClass: actualClass };
    hideClassSelector(i);

    // 선택된 클래스 표시
    const cards = document.querySelectorAll('.eval-card');
    const card = cards[i];
    let actualDisplay = card.querySelector('.actual-class');
    if (!actualDisplay) {
        card.insertAdjacentHTML('beforeend', '<div class="actual-class">실제: ' + actualClass + '</div>');
    } else {
        actualDisplay.textContent = '실제: ' + actualClass;
    }

    updateTable();
    updateChart();
    updateFinalAccuracy();
}

window.selectActualClass = selectActualClass;

function updateTable() {
    const tbody = document.querySelector('#results-table tbody');
    tbody.innerHTML = '';
    results.forEach((r, i) => {
        const evalText =
            evaluations[i] === null
                ? '-'
                : evaluations[i] === true
                  ? '<span class="eval-correct">✅ 맞음</span>'
                  : evaluations[i] === 'normal'
                    ? '<span class="eval-normal">⚪ 해당없음</span>'
                    : '<span class="eval-wrong">❌ 틀림 → ' + evaluations[i].actualClass + '</span>';
        const row = document.createElement('tr');
        row.innerHTML =
            '<td><img src="' +
            analyzedCaptures[i] +
            '" class="table-img"></td><td>' +
            r.label +
            '</td><td>' +
            (r.confidence * 100).toFixed(1) +
            '%</td><td>' +
            evalText +
            '</td>';
        tbody.appendChild(row);
    });
}

function updateChart() {
    const ctx = document.getElementById('accuracy-chart').getContext('2d');
    const classStats = {};
    metadata.labels.forEach((l) => (classStats[l] = { total: 0, correct: 0 }));

    results.forEach((r, i) => {
        // 레이블이 없으면 생성
        if (!classStats[r.label]) {
            classStats[r.label] = { total: 0, correct: 0 };
        }
        classStats[r.label].total++;
        if (evaluations[i] === true) classStats[r.label].correct++;
    });

    // 실제 사용된 레이블만 차트에 표시
    const usedLabels = Object.keys(classStats).filter((l) => classStats[l].total > 0);
    const accuracyData = usedLabels.map((l) =>
            classStats[l].total === 0 ? 0 : ((classStats[l].correct / classStats[l].total) * 100).toFixed(1),
        ),
        countData = usedLabels.map((l) => classStats[l].total);
    if (chart) chart.destroy();
    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: usedLabels,
            datasets: [
                {
                    label: '정확도 (%)',
                    data: accuracyData,
                    backgroundColor: 'rgba(0,255,100,0.7)',
                    borderColor: 'rgba(0,255,100,1)',
                    borderWidth: 1,
                },
                {
                    label: '예측 횟수',
                    data: countData,
                    backgroundColor: 'rgba(100,150,255,0.7)',
                    borderColor: 'rgba(100,150,255,1)',
                    borderWidth: 1,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                title: { display: true, text: '클래스별 정확도 및 예측 분포', color: '#ccc' },
                legend: { labels: { color: '#ccc' } },
            },
            scales: {
                y: { beginAtZero: true, max: 100, ticks: { color: '#ccc' }, grid: { color: '#444' } },
                x: { ticks: { color: '#ccc' }, grid: { color: '#444' } },
            },
        },
    });
}

function updateFinalAccuracy() {
    const evaluated = evaluations.filter((e) => e !== null),
        correct = evaluations.filter((e) => e === true).length,
        total = evaluated.length;
    document.getElementById('evaluated-count').textContent = total;
    document.getElementById('final-accuracy-value').textContent =
        total > 0 ? ((correct / total) * 100).toFixed(1) + '% (' + correct + '/' + total + ')' : '-';
}

window.removeCapture = removeCapture;
window.markResult = markResult;
