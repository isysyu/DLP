import torch
import numpy as np
import os

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        features = []
        for file in os.listdir(filePath):
            if file.endswith('.npy'):
                feature = np.load(os.path.join(filePath, file))
                features.append(feature)
        return np.concatenate(features, axis=0)

    def _getLabels(self, filePath):
        labels = []
        for file in os.listdir(filePath):
            if file.endswith('.npy'):
                label = np.load(os.path.join(filePath, file))
                labels.append(label)
        return np.concatenate(labels, axis=0)

    def __init__(self, mode):
        assert mode in ['train', 'test', 'finetune']
        if mode == 'train':
            self.features = self._getFeatures(filePath='./dataset/LOSO_train/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_train/labels/')
        elif mode == 'finetune':
            self.features = self._getFeatures(filePath='./dataset/FT/features/')
            self.labels = self._getLabels(filePath='./dataset/FT/labels/')
        elif mode == 'test':
            self.features = self._getFeatures(filePath='./dataset/LOSO_test/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_test/labels/')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return feature, label

def get_dataloader(mode, batch_size=32, shuffle=True):
    dataset = MIBCI2aDataset(mode)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)