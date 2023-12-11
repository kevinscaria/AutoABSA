import torch


def get_device():
    return torch.device(
        'mps' if torch.backends.mps.is_built() else
        'cuda' if torch.cuda.is_available() else
        'cpu'
    )