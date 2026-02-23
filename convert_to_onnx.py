"""
PyTorch 모델을 ONNX로 변환하는 스크립트
사용법: python convert_to_onnx.py
"""

import torch
import torch.nn as nn

# 모델 구조 정의 (실제 모델 구조에 맞게 수정하세요)
class YourModel(nn.Module):
    def __init__(self, num_classes):
        super(YourModel, self).__init__()
        # 예시: ResNet 기반 모델
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # ... 추가 레이어들
        )
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 설정
NUM_CLASSES = 10  # 클래스 개수에 맞게 수정
MODEL_PATH = 'ai/my_model.pth'
OUTPUT_PATH = 'ai/my_model.onnx'

def convert_to_onnx():
    # 모델 로드
    model = YourModel(num_classes=NUM_CLASSES)
    
    # .pth 파일 로드
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # state_dict 로드 (저장 방식에 따라 다를 수 있음)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 더미 입력 생성 (배치=1, 채널=3, 높이=224, 너비=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # ONNX로 변환
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ 모델이 성공적으로 변환되었습니다: {OUTPUT_PATH}")
    
    # 외부 데이터 파일을 모델에 포함시키기 (웹 브라우저 호환성을 위해)
    import onnx
    onnx_model = onnx.load(OUTPUT_PATH)
    
    # 외부 데이터가 있으면 모델 내부로 포함
    onnx.save(onnx_model, OUTPUT_PATH, save_as_external_data=False)
    
    # 변환 확인
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX 모델 검증 완료")
    print("💡 모델이 단일 파일로 저장되어 웹 브라우저에서 사용 가능합니다.")

if __name__ == '__main__':
    convert_to_onnx()
