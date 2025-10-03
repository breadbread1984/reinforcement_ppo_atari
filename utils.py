#!/usr/bin/python3

import torch

def move_dynamic_cache_to_device(dynamic_cache, device):
    # 假设dynamic_cache内部可以通过items访问所有缓存tensor
    for key, value in dynamic_cache.items():
        if isinstance(value, torch.Tensor):
            dynamic_cache[key] = value.to(device)
        elif isinstance(value, (list, tuple)):
            dynamic_cache[key] = type(value)(v.to(device) if isinstance(v, torch.Tensor) else v for v in value)
    return dynamic_cache
