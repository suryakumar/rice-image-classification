import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np



# A custom dataset class to apply one-hot encoding to labels
class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomDataset, self).__init__(root=root, transform=transform)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        
        # Convert target to one-hot encoding
        one_hot = np.zeros(len(self.classes), dtype=np.float32)
        one_hot[target] = 1.0
        target = one_hot
        
        return sample, target