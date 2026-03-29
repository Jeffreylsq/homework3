import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, split="train", transform_pipeline="default"):
        self.transform = self.get_transform(split, transform_pipeline)
        self.data = []

        with open(Path(dataset_path, "labels.csv"), newline="") as f:
            for fname, label, _ in csv.reader(f):
                if label in LABEL_NAMES:
                    img_path = Path(dataset_path, fname)
                    label_id = LABEL_NAMES.index(label)
                    self.data.append((img_path, label_id))

    def get_transform(self, split, transform_pipeline="default"):
        if split == "train":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Pairs of images and labels (int) for classification
        """
        img_path, label_id = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        data = (self.transform(img), label_id)

        return data


def load_data(
    dataset_path,
    transform_pipeline="default",
    return_dataloader=True,
    num_workers=2,
    batch_size=128,
    shuffle=False,
):
    split = "train" if shuffle else "val"

    dataset = SuperTuxDataset(
        dataset_path,
        split=split,
        transform_pipeline=transform_pipeline
    )

    if not return_dataloader:
        return dataset

    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
