"""metalcheck — PyTorch macOS Metal (MPS) utilities."""

__version__ = "0.1.0"

from metalcheck.device import device_info, get_device, is_mps_available  # noqa: F401
from metalcheck.utils import (  # noqa: F401
    benchmark_conv2d,
    benchmark_elementwise,
    benchmark_matmul,
    benchmark_model_inference,
    benchmark_model_training,
)
