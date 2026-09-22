from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from .preprocessing import get_train_transform, get_eval_transform


class GTSRBDataset(Dataset):

    def __init__(self, data, root_dir, transform=None):

        self.data = data.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"GTSRB dataset directory not found: {self.root_dir}"
            )

        required_columns = {"ClassId", "Path"}
        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = self.root_dir / row["Path"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")
        label = int(row["ClassId"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_gtsrb_csv(csv_file):

    csv_file = Path(csv_file)

    if not csv_file.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_file}"
        )

    data = pd.read_csv(csv_file)

    required_columns = {"ClassId", "Path"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {csv_file}: {missing_columns}"
        )

    return data


def create_gtsrb_datasets(
    root_dir,
    validation_size=0.2,
    random_state=42,
    train_transform=None,
    eval_transform=None,
):

    if train_transform is None:
        train_transform = get_train_transform()

    if eval_transform is None:
        eval_transform = get_eval_transform()

    root_dir = Path(root_dir)

    train_csv = root_dir / "Train.csv"
    test_csv = root_dir / "Test.csv"

    full_train_data = load_gtsrb_csv(train_csv)
    test_data = load_gtsrb_csv(test_csv)

    train_data, validation_data = train_test_split(
        full_train_data,
        test_size=validation_size,
        random_state=random_state,
        stratify=full_train_data["ClassId"],
    )

    train_dataset = GTSRBDataset(
        data=train_data,
        root_dir=root_dir,
        transform=train_transform,
    )

    validation_dataset = GTSRBDataset(
        data=validation_data,
        root_dir=root_dir,
        transform=eval_transform,
    )

    test_dataset = GTSRBDataset(
        data=test_data,
        root_dir=root_dir,
        transform=eval_transform,
    )

    return train_dataset, validation_dataset, test_dataset


def create_gtsrb_dataloaders(
    root_dir,
    batch_size=64,
    num_workers=0,
    validation_size=0.2,
    random_state=42,
    train_transform=None,
    eval_transform=None,
):

    train_dataset, validation_dataset, test_dataset = create_gtsrb_datasets(
        root_dir=root_dir,
        validation_size=validation_size,
        random_state=random_state,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader, test_loader