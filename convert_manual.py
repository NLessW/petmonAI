"""
수동 ONNX 변환 스크립트 - PyTorch export 직접 사용
"""
import torch
from ultralytics import YOLO
import os

print("=" * 60)
print("YOLOv8 수동 변환 (직접 torch.onnx.export 사용)")
print("=" * 60)

# 모델 로드
print("\n1. 모델 로딩...")
model = YOLO('ai/best.pt')
print(f"   클래스: {list(model.names.values())}")

# 모델을 eval 모드로 설정
print("\n2. 모델 준비 중...")
pytorch_model = model.model
pytorch_model.eval()

# 더미 입력 생성
print("\n3. 더미 입력 생성...")
dummy_input = torch.randn(1, 3, 640, 640)

# ONNX로 변환
print("\n4. ONNX 변환 중... (잠시만 기다려주세요)")
output_path = 'ai/best.onnx'

try:
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch'},
            'output': {0: 'batch'}
        }
    )
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✅ 변환 성공!")
        print(f"   파일: {output_path}")
        print(f"   크기: {file_size:.2f} MB")
    else:
        print("\n❌ 오류: 파일이 생성되지 않았습니다")
        
except Exception as e:
    print(f"\n❌ 변환 실패: {e}")
    import traceback
    traceback.print_exc()

# classes.txt 생성
print("\n5. classes.txt 생성...")
classes_path = 'ai/classes.txt'
with open(classes_path, 'w', encoding='utf-8') as f:
    for name in model.names.values():
        f.write(f"{name}\n")

print(f"   파일: {classes_path}")
print("\n" + "=" * 60)
print("✅ 완료!")
print("=" * 60)
