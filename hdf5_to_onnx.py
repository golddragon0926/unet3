import tf2onnx
import onnx
from tensorflow.keras.models import load_model
from losses.unet_loss import unet3p_hybrid_loss  # 손실 함수 임포트
from losses.loss import dice_coef  # dice_coef 메트릭 임포트

# 모델 로드 시 커스텀 손실 함수와 메트릭 함수 등록
model = load_model(
    'model_unet3plus.hdf5', 
    custom_objects={'unet3p_hybrid_loss': unet3p_hybrid_loss, 'dice_coef': dice_coef}
)

# 모델을 ONNX 형식으로 변환 (Opset 버전 14로 설정)
onnx_model, _ = tf2onnx.convert.from_keras(
    model, 
    opset=14  # Opset 14로 설정
)

# 변환된 모델을 ONNX 형식으로 저장
onnx.save(onnx_model, 'model_unet3plus.onnx')

# ONNX 모델 로드
onnx_model = onnx.load('model_unet3plus.onnx')

# 모델의 Opset과 IR 버전 확인
print(f"Opset Version: {onnx_model.opset_import[0].version}")
print(f"IR Version: {onnx_model.ir_version}")

# 모델의 출력 형태 확인
for output in onnx_model.graph.output:
    print(f"Output Name: {output.name}, Shape: {output.type.tensor_type.shape}")
