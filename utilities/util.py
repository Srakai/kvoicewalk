"""Device utility functions for PyTorch."""

import torch


def get_device() -> str:
    """
    Get the best available device for PyTorch computations.

    Note: MPS (Apple Silicon GPU) is currently disabled due to Kokoro TTS
    incompatibility with torch.repeat_interleave on MPS backend.
    Priority: CUDA (NVIDIA) > CPU

    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    # MPS disabled: Kokoro uses repeat_interleave which isn't supported on MPS
    # if torch.backends.mps.is_available():
    #     return "mps"
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
