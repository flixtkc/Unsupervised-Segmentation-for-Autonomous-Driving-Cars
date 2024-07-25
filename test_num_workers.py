import time
import torch
from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        time.sleep(0.01)  # Simulate data loading time
        return torch.tensor([idx])

# Adjust this value to the size of your dataset
dataset_size = 10000
dataset = DummyDataset(dataset_size)

batch_size = 64

def test_num_workers(num_workers):
    start_time = time.time()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    for batch in loader:
        pass
    end_time = time.time()
    print(f"Num workers: {num_workers}, Time taken: {end_time - start_time:.2f} seconds")

# Test different values of num_workers
for num_workers in [4, 8, 16, 32, 48, 64]:
    test_num_workers(num_workers)
