import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GTSRBDataset(Dataset):

    def __init__(self, root_dir, split="train", transform=None):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == "train":
            csv_file = f"{root_dir}/Train.csv"

        elif split == "test":
            csv_file = f"{root_dir}/Test.csv"

        else:
            raise ValueError("split must be 'train' or 'test'")

        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = f"{self.root_dir}/{row['Path']}"
        label = int(row["ClassId"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label