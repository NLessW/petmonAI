"""
YOLOv8 모델(.pt)을 ONNX로 변환하는 스크립트

사용법: 
1. ai 폴더에 best.pt 파일 배치
2. python convert_yolov8_to_onnx.py 실행

생성되는 파일:
- ai/best.onnx (ONNX 모델)
- ai/classes.txt (클래스 이름 - 자동 생성!)

참고: YOLOv8 학습 시 classes.txt는 생성되지 않지만,
      이 스크립트가 모델에서 클래스 정보를 추출해서 자동으로 생성합니다.
"""

from ultralytics import YOLO
import os

# 설정
MODEL_PATH = 'ai/best.pt'  # YOLOv8 모델 경로
OUTPUT_DIR = 'ai'
IMGSZ = 640  # 입력 이미지 크기 (640, 320, 416 등)

def convert_yolov8_to_onnx():
    print("=" * 50)
    print("YOLOv8 모델을 ONNX로 변환합니다")
    print("=" * 50)
    
    # 모델 파일 확인
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print(f"💡 팁: {MODEL_PATH} 경로에 best.pt 파일을 배치하세요")
        return
    
    try:
        # YOLOv8 모델 로드
        print(f"\n📦 모델 로딩 중: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        
        # 모델 정보 출력
        print(f"✅ 모델 로드 완료")
        print(f"   클래스 개수: {len(model.names)}")
        print(f"   클래스 이름: {list(model.names.values())}")
        
        # ONNX로 변환
        print(f"\n🔄 ONNX 변환 중...")
        print(f"   입력 크기: {IMGSZ}x{IMGSZ}")
        print(f"   잠시만 기다려주세요... (1-2분 소요될 수 있습니다)\n")
        
        # 변환된 파일 경로 (미리 계산)
        onnx_path = MODEL_PATH.replace('.pt', '.onnx')
        
        # YOLOv8의 export 메소드 사용
        try:
            export_result = model.export(
                format='onnx',
                imgsz=IMGSZ,
                simplify=False,  # simplify를 False로 변경 (안정성 향상)
                opset=11,  # ONNX opset 버전 (11로 낮춤)
                dynamic=False,  # 동적 배치 비활성화
            )
            print(f"\n✅ Export 함수 실행 완료")
        except Exception as export_error:
            print(f"\n❌ Export 중 오류 발생: {export_error}")
            raise
        
        # 파일이 실제로 생성되었는지 확인
        if not os.path.exists(onnx_path):
            print(f"❌ 오류: ONNX 파일이 생성되지 않았습니다: {onnx_path}")
            print(f"💡 export() 결과: {export_result}")
            return
        
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        print(f"\n✅ 변환 완료: {onnx_path}")
        print(f"   파일 크기: {file_size:.2f} MB")
        
        # classes.txt 파일 생성
        classes_path = os.path.join(OUTPUT_DIR, 'classes.txt')
        with open(classes_path, 'w', encoding='utf-8') as f:
            for class_name in model.names.values():
                f.write(f"{class_name}\n")
        
        print(f"✅ 클래스 파일 자동 생성: {classes_path}")
        print(f"   (모델에서 클래스 정보를 추출했습니다)")
        print(f"\n{'=' * 50}")
        print("🎉 모든 작업 완료!")
        print(f"{'=' * 50}")
        print("\n📋 생성된 파일:")
        print(f"   • ONNX 모델: {onnx_path}")
        print(f"   • 클래스 파일: {classes_path} ⭐ 자동 생성!")
        print(f"\n💡 참고:")
        print(f"   YOLOv8 학습 결과물(best.pt, last.pt)에는 classes.txt가 없지만,")
        print(f"   이 스크립트가 모델 내부 정보를 읽어서 자동으로 생성했습니다.")
        print(f"\n💡 다음 단계:")
        print(f"   1. 웹사이트를 열고 '모델 타입'을 'PyTorch/ONNX'로 선택")
        print(f"   2. '{os.path.basename(classes_path)}' 업로드 ⭐")
        print(f"   3. '{os.path.basename(onnx_path)}' 업로드")
        print(f"   4. 'ONNX 입력 크기'를 {IMGSZ}로 설정")
        print(f"   5. '모델 로드' 버튼 클릭")
        print(f"\n✨ classes.txt가 없어도 걱정하지 마세요 - 방금 생성했습니다!")
        
    except ImportError:
        print("❌ 오류: ultralytics 패키지가 설치되지 않았습니다")
        print("💡 해결 방법: pip install ultralytics")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    convert_yolov8_to_onnx()
