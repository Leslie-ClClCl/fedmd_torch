import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data.dataset import Dataset


class dataset_cifar(Dataset):  # 这是一个Dataset子类
    def __init__(self, X, y: np, transform):
        self.X = X
        self.y = torch.LongTensor(y)
        self.transform = transform
        # self.X = np.vstack(self.X).reshape(-1, 3, 32, 32)
        # self.X = self.X.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img = Image.fromarray(self.X[index])
        res_X = self.transform(img)
        res_y = self.y[index]
        return res_X, res_y

    def __len__(self):
        return len(self.y)

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y


class dataset_mnist(Dataset):  # 这是一个Dataset子类
    def __init__(self, X, y: np, transform):
        self.X = X.squeeze(-1)
        self.y = torch.LongTensor(y)
        self.transform = transform
        # self.X = np.vstack(self.X).reshape(-1, 3, 32, 32)
        # self.X = self.X.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img = Image.fromarray(self.X[index])
        res_X = self.transform(img)
        res_y = self.y[index]
        return res_X, res_y

    def __len__(self):
        return len(self.y)

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y


class dataset_logits(Dataset):  # 这是一个Dataset子类
    def __init__(self, X, y: np, transform):
        self.X = X
        self.y = y
        self.transform = transform
        # self.X = np.vstack(self.X).reshape(-1, 3, 32, 32)
        # self.X = self.X.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img = Image.fromarray(self.X[index])
        res_X = self.transform(img)
        res_y = self.y[index]
        return res_X, res_y

    def __len__(self):
        return len(self.y)

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y


def generate_alignment_data(X, y, N_alignment=3000, transform=None):
    split = StratifiedShuffleSplit(n_splits=1, train_size=N_alignment)
    if N_alignment == "all":
        alignment_data = {}
        alignment_data["idx"] = np.arange(y.shape[0])
        alignment_data["X"] = X
        alignment_data["y"] = y
        return alignment_data
    for train_index, _ in split.split(X, y):
        X_alignment = X[train_index]
        y_alignment = y[train_index]
    alignment_set = dataset_cifar(X_alignment, y_alignment, transform=transform)
    return alignment_set


if __name__ == '__main__':

    from torchvision import transforms, datasets
    import torch.utils.data as data

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True)
    train_dataset.targets = np.squeeze(train_dataset.targets)
    test_dataset.targets = np.squeeze(test_dataset.targets)

    train_dataset = dataset_cifar(train_dataset.data, train_dataset.targets, transform=train_transform)
    train_loader = data.DataLoader(train_dataset, 128)

    for idx, (img, label) in enumerate(train_loader):
        pass
