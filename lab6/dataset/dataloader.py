import os
from tqdm import tqdm
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "iclevr")

class IclevrDataset(Dataset):

    def __init__(
        self, mode="train", json_root=DATA_DIR, image_root=IMAGE_DIR, num_cpus=8
    ):
        super().__init__()

        assert mode in ["train", "test", "new_test"], "IclevrDataset mode error"

        self.mode = mode
        self.image_root = image_root
        self.json_root = json_root

        self.image_transformation = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.labels = None
        json_path = os.path.join(self.json_root, f"{self.mode}.json")
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

        if self.mode == "train":
            self.image_paths = list(json_data.keys())
            self.labels = list(json_data.values())
        elif self.mode == "test" or self.mode == "new_test":
            self.labels = list(json_data)
            self.image_paths = [f"{self.mode}_{i:06d}.png" for i in range(len(self.labels))]


        objects_path = os.path.join(self.json_root, "objects.json")
        with open(objects_path, "r") as json_file:
            self.objects_dict = json.load(json_file)


        one_hot_labels = []
        for label in self.labels:
            one_hot_label = torch.zeros(len(self.objects_dict), dtype=torch.long)
            for label_name in label:
                one_hot_label[self.objects_dict[label_name]] = 1
            one_hot_labels.append(one_hot_label)
        self.labels = torch.stack(one_hot_labels)

        self.images = None
        if self.mode == "train":
            with ThreadPoolExecutor(max_workers=num_cpus) as executor:
                self.images = list(
                    tqdm(
                        executor.map(self._load_image, self.image_paths),
                        total=len(self.image_paths),
                        desc=f"Loading {self.mode} images",
                    )
                )
            self.images = torch.stack(self.images)

    def _load_image(self, image_path):
        try:
            full_path = os.path.join(self.image_root, image_path)
            image = Image.open(full_path).convert("RGB")
            image = self.image_transformation(image)
            return image
        except IOError:
            print(f"Error loading image: {full_path}")
            return torch.zeros(3, 64, 64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.mode == "train" and self.images is not None:
            return self.images[index], self.labels[index]
        elif self.mode == "test" or self.mode == "new_test":
            return self.labels[index]  # 只返回標籤
        else:
            image = self._load_image(self.image_paths[index])
            return image, self.labels[index]


if __name__ == "__main__":
    print("Train dataset")
    train_dataset = IclevrDataset(mode="train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape)
        break

    print("Test dataset")
    test_dataset = IclevrDataset(mode="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    for batch in test_loader:
        print(batch[0].shape, batch[1].shape)
        break

    print("New Test dataset")
    new_test_dataset = IclevrDataset(mode="new_test")
    new_test_loader = DataLoader(new_test_dataset, batch_size=32, shuffle=False)
    for batch in new_test_loader:
        print(batch[0].shape, batch[1].shape)
        break