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

function checkFilesReady() {
    document.getElementById('load-model-btn').disabled = !(metadataFile && modelFile && weightsFile);
}

document.getElementById('load-model-btn').addEventListener('click', async () => {
    const statusMsg = document.getElementById('model-status-msg');
    statusMsg.textContent = '모델 로딩 중...';
    try {
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
        statusMsg.textContent = '✅ 모델 로드 완료! 클래스: ' + metadata.labels.join(', ');
        document.getElementById('start-webcam-btn').disabled = false;
        displayClassLabels();
    } catch (error) {
        console.error(error);
        statusMsg.textContent = '❌ 모델 로드 실패: ' + error.message;
    }
});

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
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 180, facingMode: 'environment' },
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

    // 웹캠 전체 화면을 캡처 (640x180)
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
    if (!model || captures.length === 0) return;
    const btn = document.getElementById('start-test-btn');
    btn.textContent = '분석 중...';
    btn.disabled = true;
    results = [];

    // 각 캡처 이미지를 8등분하여 분석 (결과를 이미지별로 그룹화)
    for (let i = 0; i < captures.length; i++) {
        const slices = await sliceImage(captures[i]);
        const imageResults = [];
        for (let j = 0; j < slices.length; j++) {
            const result = await predictImage(slices[j]);
            imageResults.push(result);
        }
        results.push(imageResults);
    }

    evaluations = new Array(captures.length).fill(null).map(() => new Array(8).fill(null));
    analyzedCaptures = [...captures];
    displayResults();

    // 분석 완료 후 초기화하여 다시 테스트 가능하게
    captures = [];
    updateCapturesGrid();
    btn.textContent = '🔍 판독 시작 (0/10)';
    btn.disabled = false;
}

async function sliceImage(imageData) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const slices = [];
            const sliceWidth = 80; // 가로 80px씩
            const numSlices = 8;

            for (let i = 0; i < numSlices; i++) {
                const canvas = document.createElement('canvas');
                canvas.width = sliceWidth;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, i * sliceWidth, 0, sliceWidth, img.height, 0, 0, sliceWidth, img.height);
                slices.push(canvas.toDataURL('image/jpeg'));
            }
            resolve(slices);
        };
        img.src = imageData;
    });
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

async function predictImage(imageData) {
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
            const label = confidence < 0.5 ? 'ok_normal' : metadata.labels[maxIndex];
            resolve({
                label: label,
                confidence: confidence,
                allPredictions: Array.from(predictions),
            });
        };
        img.src = imageData;
    });
}

function displayResults() {
    document.getElementById('results-section').style.display = 'block';
    const evalContainer = document.getElementById('evaluation-container');
    evalContainer.innerHTML = '';

    results.forEach((imageResults, imgIdx) => {
        const card = document.createElement('div');
        card.className = 'eval-card';

        // 이미지 컨테이너 생성
        const imgContainer = document.createElement('div');
        imgContainer.className = 'image-container';
        imgContainer.style.position = 'relative';

        const img = document.createElement('img');
        img.src = analyzedCaptures[imgIdx];
        img.style.width = '100%';
        img.style.display = 'block';
        imgContainer.appendChild(img);

        // 8개 영역 오버레이 생성
        for (let i = 0; i < 8; i++) {
            const overlay = document.createElement('div');
            overlay.className = 'slice-overlay';
            overlay.style.position = 'absolute';
            overlay.style.left = i * 12.5 + '%';
            overlay.style.top = '0';
            overlay.style.width = '12.5%';
            overlay.style.height = '100%';
            overlay.style.cursor = 'pointer';
            overlay.style.transition = 'background 0.2s';

            const r = imageResults[i];
            const tooltip = document.createElement('div');
            tooltip.className = 'slice-tooltip';
            tooltip.innerHTML = '<strong>' + r.label + '</strong><br>' + (r.confidence * 100).toFixed(1) + '%';
            tooltip.style.display = 'none';
            overlay.appendChild(tooltip);

            overlay.addEventListener('mouseenter', function () {
                this.style.background = 'rgba(0, 255, 100, 0.3)';
                tooltip.style.display = 'block';
            });
            overlay.addEventListener('mouseleave', function () {
                this.style.background = 'transparent';
                tooltip.style.display = 'none';
            });

            imgContainer.appendChild(overlay);
        }

        card.appendChild(imgContainer);

        // 전체 평가 정보 및 버튼
        const infoDiv = document.createElement('div');
        infoDiv.className = 'eval-info';

        // 각 조각의 예측 요약
        const summary = {};
        imageResults.forEach((r) => {
            summary[r.label] = (summary[r.label] || 0) + 1;
        });
        const summaryText = Object.entries(summary)
            .map(([label, count]) => label + '×' + count)
            .join(', ');

        infoDiv.innerHTML =
            '<div class="prediction">예측: ' +
            summaryText +
            '</div>' +
            '<div class="eval-buttons">' +
            '<button class="correct-btn" onclick="markImageResult(' +
            imgIdx +
            ',true)">✅ 전체 맞음</button>' +
            '<button class="wrong-btn" onclick="markImageResult(' +
            imgIdx +
            ',false)">❌ 전체 틀림</button>' +
            '</div>';

        card.appendChild(infoDiv);
        evalContainer.appendChild(card);
    });

    updateTable();
    updateChart();
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

function markImageResult(imgIdx, isCorrect) {
    for (let i = 0; i < 8; i++) {
        evaluations[imgIdx][i] = isCorrect;
    }

    const cards = document.querySelectorAll('.eval-card');
    const btns = cards[imgIdx].querySelectorAll('.eval-buttons button');
    btns.forEach((b) => b.classList.remove('selected'));
    if (isCorrect) btns[0].classList.add('selected');
    else btns[1].classList.add('selected');

    updateTable();
    updateChart();
    updateFinalAccuracy();
}

window.markImageResult = markImageResult;

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

    results.forEach((imageResults, imgIdx) => {
        imageResults.forEach((r, sliceIdx) => {
            const evalStatus = evaluations[imgIdx][sliceIdx];
            const evalText =
                evalStatus === null
                    ? '-'
                    : evalStatus === true
                      ? '<span class="eval-correct">✅ 맞음</span>'
                      : '<span class="eval-wrong">❌ 틀림</span>';
            const row = document.createElement('tr');
            row.innerHTML =
                '<td>이미지' +
                (imgIdx + 1) +
                '-' +
                (sliceIdx + 1) +
                '</td><td>' +
                r.label +
                '</td><td>' +
                (r.confidence * 100).toFixed(1) +
                '%</td><td>' +
                evalText +
                '</td>';
            tbody.appendChild(row);
        });
    });
}

function updateChart() {
    const ctx = document.getElementById('accuracy-chart').getContext('2d');
    const classStats = {};
    metadata.labels.forEach((l) => (classStats[l] = { total: 0, correct: 0 }));

    results.forEach((imageResults, imgIdx) => {
        imageResults.forEach((r, sliceIdx) => {
            classStats[r.label].total++;
            if (evaluations[imgIdx][sliceIdx] === true) classStats[r.label].correct++;
        });
    });

    const labels = metadata.labels,
        accuracyData = labels.map((l) =>
            classStats[l].total === 0 ? 0 : ((classStats[l].correct / classStats[l].total) * 100).toFixed(1),
        ),
        countData = labels.map((l) => classStats[l].total);
    if (chart) chart.destroy();
    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
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
    let total = 0;
    let correct = 0;

    evaluations.forEach((imageEvals) => {
        imageEvals.forEach((e) => {
            if (e !== null) {
                total++;
                if (e === true) correct++;
            }
        });
    });

    document.getElementById('evaluated-count').textContent = total;
    document.getElementById('final-accuracy-value').textContent =
        total > 0 ? ((correct / total) * 100).toFixed(1) + '% (' + correct + '/' + total + ')' : '-';
}

window.removeCapture = removeCapture;
window.markResult = markResult;
