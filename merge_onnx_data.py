"""
외부 데이터 파일(.data)을 포함한 ONNX 모델을 단일 파일로 병합하는 스크립트
사용법: python merge_onnx_data.py
"""

import onnx
import os

# 설정
INPUT_MODEL = 'ai/yolox_best.onnx'  # 입력 ONNX 모델 경로
OUTPUT_MODEL = 'ai/yolox_best_merged.onnx'  # 출력 ONNX 모델 경로

def merge_onnx_external_data():
    """외부 데이터 파일을 ONNX 모델 내부로 병합"""
    
    if not os.path.exists(INPUT_MODEL):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {INPUT_MODEL}")
        return
    
    print(f"📂 모델 로딩 중: {INPUT_MODEL}")
    
    # ONNX 모델 로드 (외부 데이터 포함)
    onnx_model = onnx.load(INPUT_MODEL)
    
    print("🔄 외부 데이터를 모델 내부로 병합 중...")
    
    # 외부 데이터를 모델 내부로 포함하여 저장
    onnx.save(
        onnx_model,
        OUTPUT_MODEL,
        save_as_external_data=False  # 외부 데이터를 사용하지 않음
    )
    
    print(f"✅ 병합 완료: {OUTPUT_MODEL}")
    
    # 파일 크기 확인
    input_size = os.path.getsize(INPUT_MODEL) / (1024 * 1024)
    output_size = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
    
    print(f"📊 원본 크기: {input_size:.2f} MB")
    print(f"📊 병합 후 크기: {output_size:.2f} MB")
    
    # 모델 검증
    print("🔍 모델 검증 중...")
    onnx.checker.check_model(OUTPUT_MODEL)
    print("✅ 모델 검증 완료!")
    print("💡 이제 웹 브라우저에서 사용할 수 있습니다.")

if __name__ == '__main__':
    try:
        merge_onnx_external_data()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("\n💡 확인 사항:")
        print("1. ONNX 모델 파일과 .data 파일이 같은 폴더에 있는지 확인")
        print("2. onnx 패키지가 설치되어 있는지 확인 (pip install onnx)")
