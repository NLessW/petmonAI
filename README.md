# PETMON AI 테스트 (ONNX Runtime Web)

PyTorch 모델을 웹 브라우저에서 실행하는 AI 테스트 플랫폼입니다.

## 🚀 빠른 시작

### 1. PyTorch 모델을 ONNX로 변환

#### YOLOv8 모델 변환 (권장)

YOLOv8으로 학습한 `best.pt` 파일을 변환하는 경우:

```bash
# Ultralytics 설치
pip install ultralytics

# ai 폴더에 best.pt 파일 배치 후 실행
python convert_yolov8_to_onnx.py
```

✨ **변환 스크립트가 자동으로 생성하는 파일:**

- `ai/best.onnx` - ONNX 모델 파일
- `ai/classes.txt` - 클래스 이름 목록 (**자동 생성 - 별도 준비 불필요!**)

💡 **참고**: YOLOv8 학습 결과물은 `best.pt`와 `last.pt` 2개만 나오지만,  
변환 스크립트가 모델에서 클래스 정보를 추출해서 `classes.txt`를 자동으로 만들어줍니다.

#### 일반 PyTorch 모델 변환

```bash
pip install torch onnx
python convert_to_onnx.py
```

**중요:** `convert_to_onnx.py` 파일을 열어 다음을 수정하세요:

- `YourModel` 클래스를 실제 모델 구조에 맞게 수정
- `NUM_CLASSES`: 클래스 개수
- `MODEL_PATH`: .pth 파일 경로
- `OUTPUT_PATH`: 출력할 .onnx 파일 경로

### 2. 클래스 이름 파일 준비 (일반 PyTorch 모델만 해당)

**YOLOv8을 사용하는 경우 이 단계는 건너뛰세요!** (자동으로 생성됨)

일반 PyTorch 모델을 사용하는 경우, `ai/classes.txt` 파일에 클래스 이름을 한 줄에 하나씩 작성:

```
ok_normal
defect_crack
defect_scratch
...
```

### 3. GitHub Pages에 배포

1. GitHub 저장소 생성
2. Settings > Pages에서 배포 브랜치 설정
3. 파일 업로드 후 `https://username.github.io/repository-name/` 접속

### 4. 웹 사이트에서 사용

1. **모델 파일 업로드**
    - 모델 타입: `PyTorch/ONNX` 선택
    - `classes.txt` 선택
    - `best.onnx` (또는 `my_model.onnx`) 선택
    - ONNX 입력 크기: YOLOv8인 경우 `640x640` 선택 (일반 모델은 `224x224`)
    - "모델 로드" 버튼 클릭

2. **웹캠 설정**
    - 화면 비율 선택 (1:1, 4:3, 16:9)
    - 해상도 선택 (예: 1920x1080)

3. **웹캠 테스트**
    - "웹캠 시작" 클릭
    - "판독 시작" 클릭하여 이미지 10장 자동 촬영

4. **결과 분석**
    - 각 이미지가 자동으로 분석됨
    - 예측 결과를 평가하고 정확도 확인
    - `no_object` 클래스는 자동으로 무시됨

## 📁 파일 구조

```
ai_test/
├── index.html                    # 메인 HTML
├── css/
│   └── styles.css                # 스타일시트
├── src/
│   └── index.js                  # ONNX Runtime Web 로직
├── ai/
│   ├── best.pt                   # 원본 YOLOv8 모델
│   ├── best.onnx                 # 변환된 ONNX 모델
│   ├── my_model.pth              # 또는 일반 PyTorch 모델
│   └── classes.txt               # 클래스 이름 목록
├── convert_yolov8_to_onnx.py     # YOLOv8 변환 스크립트
├── convert_to_onnx.py            # 일반 PyTorch 변환 스크립트
└── merge_onnx_data.py            # ONNX 데이터 병합 스크립트
```

## 🔧 기술 스택

- **ONNX Runtime Web**: 브라우저에서 ONNX 모델 실행
- **Chart.js**: 정확도 시각화
- **Vanilla JavaScript**: 순수 자바스크립트

## 📝 모델 요구사항

### YOLOv8 모델

- **입력**: `[1, 3, 640, 640]` (배치, RGB, 높이, 너비)
- **전처리**: 0-255 픽셀값을 0-1로 정규화
- **출력**: YOLOv8 표준 출력 형식

### 일반 분류 모델

- **입력**: `[1, 3, 224, 224]` (배치, RGB, 높이, 너비)
- **전처리**: ImageNet 정규화 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **출력**: `[1, num_classes]` (클래스별 확률)

## ⚠️ 주의사항

1. **모델 구조**: `convert_to_onnx.py`의 `YourModel` 클래스를 실제 모델에 맞게 수정
2. **입력 크기**: 224x224가 아닌 경우 코드 수정 필요
3. **정규화**: 모델 학습 시 사용한 정규화 방식과 동일하게 설정
4. **브라우저 호환성**: 최신 Chrome, Edge, Firefox 권장

## 🐛 문제 해결

### ONNX 외부 데이터 파일 오류

**오류 메시지**: `Failed to load external data file ".onnx.data"`

ONNX 모델이 외부 데이터 파일(.data)을 사용하는 경우, 브라우저에서 로드할 수 없습니다.

**해결 방법**:

```bash
pip install onnx
python merge_onnx_data.py
```

`merge_onnx_data.py` 파일에서 파일 경로를 확인하고 실행하면, 외부 데이터를 포함한 단일 ONNX 파일이 생성됩니다.

### 모델 로드 실패

- ONNX 모델이 올바르게 변환되었는지 확인
- 브라우저 콘솔에서 에러 메시지 확인

### 예측 결과가 이상함

- 전처리(정규화) 방식이 학습 시와 동일한지 확인
- 입력 이미지 크기 확인
- 클래스 이름 순서가 올바른지 확인

### 웹캠이 작동하지 않음

- HTTPS 환경에서만 작동 (GitHub Pages는 자동으로 HTTPS)
- 브라우저 권한 설정 확인
