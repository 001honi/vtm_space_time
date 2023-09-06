import torch
from torch.utils.data import Dataset


class BenchmarkDataset(Dataset):
    def __init__(self, shot, dset_size):
        self.shot = shot
        self.dset_size = dset_size

    def __len__(self):
        return self.dset_size
    
    def __getitem__(self, idx):
        X = torch.rand(1, self.shot, 3, 224, 224)
        Y = torch.rand(1, self.shot, 1, 224, 224)
        M = torch.ones_like(Y)
        t_idx = torch.LongTensor([0])

        return X, Y, M, t_idx