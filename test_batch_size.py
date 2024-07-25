import time
import torch
from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(3, 224, 224), torch.tensor([idx])  # Simulate data

# Adjust this value to the size of your dataset
dataset_size = 10000
dataset = DummyDataset(dataset_size)

def test_batch_size(batch_size, num_workers):
    try:
        start_time = time.time()
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        for batch in loader:
            pass
        end_time = time.time()
        print(f"Batch size: {batch_size}, Time taken: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Batch size: {batch_size} failed with error: {e}")

# Test different values of batch_size around 128
num_workers = 16  # Optimal value found previously
for batch_size in [96, 112, 128, 144, 160]:
    test_batch_size(batch_size, num_workers)
