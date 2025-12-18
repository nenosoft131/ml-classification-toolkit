import os
import numpy as np
import torch


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def to_numpy(ft: torch.Tensor | None) -> np.ndarray | None:
    if ft is None:
        return None
    if isinstance(ft, np.ndarray):
        return ft
    try:
        return ft.detach().cpu().numpy()
    except AttributeError:
        pass
    return np.array(ft)


def to_image(m):
    img = to_numpy(m)
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0)).copy()
    return img


def unsqueeze(x):
    if isinstance(x, np.ndarray):
        return x[np.newaxis, ...]
    else:
        return x.unsqueeze(dim=0)


def atleast4d(x):
    if x is None:
        return x
    if len(x.shape) == 3:
        return unsqueeze(x)
    return x


def atleast3d(x):
    if x is None:
        return x
    if len(x.shape) == 2:
        return unsqueeze(x)
    return x



