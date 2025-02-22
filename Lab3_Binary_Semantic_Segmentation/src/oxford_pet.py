import os
import torch
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):
        assert mode in {"train", "valid", "test"}
        self.root = root
        self.mode = mode
        self.transform = transform
        self.images_directory = os.path.join(self.root, "oxford-iiit-pet", "images")
        self.masks_directory = os.path.join(self.root, "oxford-iiit-pet", "annotations", "trimaps")
        self.filenames = self._read_split()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = Image.open(image_path).convert("RGB")
        trimap = Image.open(mask_path)


        image = image.resize((256, 256), Image.BILINEAR)
        trimap = trimap.resize((256, 256), Image.NEAREST)

        mask = self._preprocess_mask(np.array(trimap))

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(mask).long()
        return image, mask

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "oxford-iiit-pet", "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):
        target_root = os.path.join(root, "oxford-iiit-pet")

        filepath = os.path.join(target_root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath, target_root)

        filepath = os.path.join(target_root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath, target_root)


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(filepath)) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath, extract_dir):
    if not os.path.exists(os.path.join(extract_dir, os.path.basename(filepath).split('.')[0])):
        shutil.unpack_archive(filepath, extract_dir)


def get_transform(mode):
    if mode == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def load_dataset(data_path, mode, batch_size=2):
    transform = get_transform(mode)
    dataset = OxfordPetDataset(root=data_path, mode=mode, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"))
    return dataloader


import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#if __name__ == "__main__":
#    data_path = "../dataset"

#    if not os.path.exists(os.path.join(data_path, "oxford-iiit-pet", "images")):
#       OxfordPetDataset.download(data_path)


#    train_loader = load_dataset(data_path, "train")
#    val_loader = load_dataset(data_path, "valid")
#    test_loader = load_dataset(data_path, "test")

#    print(f"Train dataset size: {len(train_loader.dataset)}")
#    print(f"Validation dataset size: {len(val_loader.dataset)}")
#    print(f"Test dataset size: {len(test_loader.dataset)}")

#    for images, masks in train_loader:
#        print("Image batch shape:", images.shape)
#        print("Mask batch shape:", masks.shape)
#        print("Unique values in mask:", torch.unique(masks))
#        break