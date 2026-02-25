"""
YOLOv8 간단 변환 스크립트 (문제 발생 시 사용)
"""

from ultralytics import YOLO

# 모델 로드
print("모델 로딩...")
model = YOLO('ai/best.pt')

print(f"클래스: {list(model.names.values())}")

# 간단한 변환 (최소 옵션)
print("\nONNX 변환 시작...")
model.export(format='onnx')

print("\n✅ 완료! ai/best.onnx 파일을 확인하세요.")

# classes.txt 생성
with open('ai/classes.txt', 'w', encoding='utf-8') as f:
    for name in model.names.values():
        f.write(f"{name}\n")

print("✅ ai/classes.txt 파일도 생성했습니다.")
