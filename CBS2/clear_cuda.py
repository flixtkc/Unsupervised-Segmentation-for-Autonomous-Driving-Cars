import torch

def clear_cuda_cache():
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

clear_cuda_cache()
