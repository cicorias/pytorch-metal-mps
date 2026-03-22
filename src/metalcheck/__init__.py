"""metalcheck — PyTorch GPU acceleration utilities (MPS / ROCm / CUDA)."""

__version__ = "0.1.0"

from metalcheck.device import (  # noqa: F401
    device_info,
    get_device,
    is_cuda_available,
    is_mps_available,
    is_rocm_available,
)
from metalcheck.utils import (  # noqa: F401
    benchmark_conv2d,
    benchmark_elementwise,
    benchmark_matmul,
    benchmark_model_inference,
    benchmark_model_training,
)
