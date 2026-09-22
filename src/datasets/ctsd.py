from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from .preprocessing import get_train_transform, get_eval_transform


REQUIRED_COLUMNS = {
    "file_name",
    "width",
    "height",
    "x1",
    "y1",
    "x2",
    "y2",
    "category",
}


class CTSDSampleDataset(Dataset):

    def __init__(self, data, image_dir, transform=None):
        self.data = data.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"CTSD image directory not found: {self.image_dir}"
            )

        missing_columns = REQUIRED_COLUMNS - set(self.data.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required CTSD annotation columns: {missing_columns}"
            )

        if self.data.empty:
            raise ValueError("CTSD dataset contains no samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = self.image_dir / str(row["file_name"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")
        label = int(row["category"])

        if self.transform is not None:
            image = self.transform(image)

        metadata = {
            "file_name": str(row["file_name"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "x1": int(row["x1"]),
            "y1": int(row["y1"]),
            "x2": int(row["x2"]),
            "y2": int(row["y2"]),
        }

        return image, label, metadata


def load_ctsd_annotations(csv_file):
    
    csv_file = Path(csv_file)

    if not csv_file.exists():
        raise FileNotFoundError(
            f"CTSD annotation file not found: {csv_file}"
        )

    data = pd.read_csv(csv_file)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required CTSD annotation columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError(
            f"CTSD annotation file is empty: {csv_file}"
        )

    if data["file_name"].isna().any():
        raise ValueError(
            "Missing file names found in CTSD annotations."
        )

    if data["category"].isna().any():
        raise ValueError(
            "Missing category labels found in CTSD annotations."
        )

    data["category"] = data["category"].astype(int)

    if (data["category"] < 0).any():
        raise ValueError(
            "CTSD category labels must be non-negative."
        )

    
    category_counts = data.groupby("file_name")["category"].nunique()
    conflicting_files = category_counts[category_counts > 1]

    if not conflicting_files.empty:
        raise ValueError(
            "Some CTSD images have conflicting category labels: "
            f"{list(conflicting_files.index)}"
        )

    data = data.drop_duplicates(subset="file_name", keep="first")
    data = data.reset_index(drop=True)

    return data


def load_ctsd_data(root_dir):
    
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(
            f"CTSD root directory not found: {root_dir}"
        )

    annotation_file = root_dir / "annotations.csv"
    image_dir = root_dir / "images"

    data = load_ctsd_annotations(annotation_file)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"CTSD image directory not found: {image_dir}"
        )

    return data, image_dir


def create_ctsd_datasets(
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

    data, image_dir = load_ctsd_data(root_dir)

    train_data, validation_data = train_test_split(
        data,
        test_size=validation_size,
        random_state=random_state,
        stratify=data["category"],
    )

    train_dataset = CTSDSampleDataset(
        data=train_data,
        image_dir=image_dir,
        transform=train_transform,
    )

    validation_dataset = CTSDSampleDataset(
        data=validation_data,
        image_dir=image_dir,
        transform=eval_transform,
    )

    return train_dataset, validation_dataset


def create_ctsd_dataloaders(
    root_dir,
    batch_size=64,
    num_workers=0,
    validation_size=0.2,
    random_state=42,
    train_transform=None,
    eval_transform=None,
):

    train_dataset, validation_dataset = create_ctsd_datasets(
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

    return train_loader, validation_loader
