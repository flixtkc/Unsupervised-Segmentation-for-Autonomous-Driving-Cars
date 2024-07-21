import torch

def clear_gpu_memory():
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

clear_gpu_memory()
