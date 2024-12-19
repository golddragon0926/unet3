import os
import cv2
import numpy as np
import onnxruntime as ort

# 모델 로드 (ONNX 형식)
onnx_model_path = "model_unet3plus.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# 입력 및 출력 폴더 설정
input_folder = "data/test/images"
output_folder = "onnx_python_result"
os.makedirs(output_folder, exist_ok=True)

# 색상 매핑 딕셔너리 (BGR 순서로 변경)
color_map = {
    0: [0, 255, 0],  # 초록색 (BGR 순서)
    1: [124, 10, 4],  # 파란색 (BGR 순서)
}

# 이미지 처리 함수
def process_image(image_file):
    # 이미지 읽기 (Grayscale -> RGB 변환)
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load {image_file}")
        return

    # Grayscale -> RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # ONNX 모델 입력 형태로 변환
    input_image = cv2.resize(image_rgb, (512, 512)).astype(np.float32) / 255.0
    input_image = np.expand_dims(input_image, axis=0)  # (1, H, W, C)

    # ONNX 추론
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    predicted_output = ort_session.run([output_name], {input_name: input_image})[0][0]

    # 디버깅: 추론 결과 확인
    print(f"Predicted Output Shape: {predicted_output.shape}")
    print(f"Predicted Output Unique Values: {np.unique(predicted_output)}")

    # 클래스 맵 변환
    predicted_output = np.argmax(predicted_output, axis=-1)  # 클래스 맵으로 변환
    print(f"Class Map Unique Values: {np.unique(predicted_output)}")

    # 출력 이미지를 RGB로 변환
    output_image = np.zeros((predicted_output.shape[0], predicted_output.shape[1], 3), dtype=np.uint8)
    for value, color in color_map.items():
        output_image[predicted_output == value] = color

    # 원본 크기로 복원 (선택 사항)
    output_image = cv2.resize(output_image, (image.shape[1], image.shape[0]))

    # 원본 이미지와 예측 결과 합치기
    combined_image = cv2.addWeighted(image_rgb, 0.3, output_image, 0.7, 0)

    # 결과 저장 (예측된 이미지)
    original_name = os.path.basename(image_file)
    predicted_image_path = os.path.join(output_folder, f"predicted_{original_name}")
    if not cv2.imwrite(predicted_image_path, output_image):
        print(f"Failed to save predicted image: {predicted_image_path}")
    else:
        print(f"Saved predicted image: {predicted_image_path}")

    # 결과 저장 (원본 이미지와 합성된 이미지)
    combined_image_path = os.path.join(output_folder, f"combined_{original_name}")
    if not cv2.imwrite(combined_image_path, combined_image):
        print(f"Failed to save combined image: {combined_image_path}")
    else:
        print(f"Saved combined image: {combined_image_path}")


# 데이터 폴더 내 모든 이미지 처리
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith((".png", ".jpg", ".jpeg"))]
for image_file in image_files:
    process_image(image_file)
