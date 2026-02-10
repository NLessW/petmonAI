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
    metadata.labels.forEach(label => {
        const item = document.createElement('span');
        item.className = 'class-label-item';
        item.textContent = label;
        container.appendChild(item);
    });
}

document.getElementById('start-webcam-btn').addEventListener('click', async () => {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'environment' },
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
    evaluations = new Array(captures.length).fill(null);
    analyzedCaptures = [...captures];
    for (let i = 0; i < captures.length; i++) results.push(await predictImage(captures[i]));
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

async function predictImage(imageData) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = async () => {
            const canvas = document.createElement('canvas');
            canvas.width = 224;
            canvas.height = 224;
            canvas.getContext('2d').drawImage(img, 0, 0, 224, 224);
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
            ',\'normal\')">⚪ 해당없음</button></div>';
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
        const conf = (r.allPredictions[idx] * 100).toFixed(1);
        selectorHtml += '<div class="class-option" onclick="selectActualClass(' + i + ',\'' + label + '\')">' +
            '<span class="class-name">' + label + '</span>' +
            '<div class="conf-bar-bg"><div class="conf-bar" style="width:' + conf + '%"></div></div>' +
            '<span class="conf-value">' + conf + '%</span></div>';
    });
    
    // ok_normal 옵션도 추가
    selectorHtml += '<div class="class-option" onclick="selectActualClass(' + i + ',\'ok_normal\')">' +
        '<span class="class-name">ok_normal (해당없음)</span>' +
        '<div class="conf-bar-bg"><div class="conf-bar" style="width:0%"></div></div>' +
        '<span class="conf-value">-</span></div>';
    
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
                    ? '<span class="eval-normal">⚪ ok_normal</span>'
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
        classStats[r.label].total++;
        if (evaluations[i] === true) classStats[r.label].correct++;
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
                legend: { labels: { color: '#ccc' } }
            },
            scales: { 
                y: { beginAtZero: true, max: 100, ticks: { color: '#ccc' }, grid: { color: '#444' } },
                x: { ticks: { color: '#ccc' }, grid: { color: '#444' } }
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
