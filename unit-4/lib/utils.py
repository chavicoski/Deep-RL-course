"""Implementation of some generic utility functions"""
import torch


def get_device(device_type: str) -> str:
    """Given the selected computing device type, checks if it is available and returns
    the device id of the selected computing device

    Args:
        device_type (str): Type of computing device. Choices: "gpu" and "cpu"

    Returns:
        The id of the selected device
    """
    if device_type == "gpu":
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            print(f"There is no GPU available, going to fallback to CPU")
            return "cpu"
    elif device_type == "cpu":
        return "cpu"
    else:
        raise ValueError(f"Invalid computing device, got '{device_type}'")
