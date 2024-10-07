VAE_CONFIGS = '/home/tiennv/chaos/stable_diffusion_onnx/vae_text_encoder/vae_decoder/config.json'

VAE_ONNX_PATH = '/home/tiennv/chaos/stable_diffusion_onnx/vae_text_encoder/vae_decoder/model.onnx'

UNET_ONNX_PATH = "/home/tiennv/chaos/trash/unet_optimize/model.onnx"

TOKENIZER_PATH = '/home/tiennv/chaos/stable_diffusion_onnx/vae_text_encoder/tokenizer'

TEXT_ENCODER_PATH = '/home/tiennv/chaos/stable_diffusion_onnx/vae_text_encoder/text_encoder/model.onnx'

SCHEDULER_PATH = '/home/tiennv/chaos/stable_diffusion_onnx/vae_text_encoder/scheduler'

CONTROLNEXT_ONNX_PATH = '/home/tiennv/chaos/stable_diffusion_onnx/controlnext/model.onnx'

IMAGE_ENCODER_ONNX_PATH = '/home/tiennv/trang/ControlNeXt/ControlNeXt-SD1.5/onnx_w/image_encoder/model.onnx'

PROJ_ONNX_PATH = '/home/tiennv/chaos/stable_diffusion_onnx/proj/model.onnx'

providers = ['CUDAExecutionProvider']

provider_options = [{'device_id': 1}]
