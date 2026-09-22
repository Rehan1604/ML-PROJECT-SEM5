from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from .preprocessing import get_train_transform, get_eval_transform


NUM_CLASSES = 62
IMAGE_EXTENSIONS = {".ppm", ".png", ".jpg", ".jpeg"}


class BTSDSampleDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

        if not self.samples:
            raise ValueError("BTSD dataset contains no image samples.")

        for image_path, label in self.samples:
            if not Path(image_path).exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )
            if not 0 <= int(label) < NUM_CLASSES:
                raise ValueError(
                    f"Invalid BTSD class label: {label}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, int(label)


def _collect_samples(split_dir):

    split_dir = Path(split_dir)

    if not split_dir.exists():
        raise FileNotFoundError(
            f"BTSD split directory not found: {split_dir}"
        )

    if not split_dir.is_dir():
        raise NotADirectoryError(
            f"BTSD split path is not a directory: {split_dir}"
        )

    samples = []
    class_dirs = sorted(
        path for path in split_dir.iterdir()
        if path.is_dir()
    )

    if not class_dirs:
        raise ValueError(
            f"No class directories found in: {split_dir}"
        )

    for class_dir in class_dirs:
        try:
            class_id = int(class_dir.name)
        except ValueError:
            # Ignore non-class directories.
            continue

        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(
                f"Invalid BTSD class directory: {class_dir.name}"
            )

        image_files = sorted(
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_files:
            samples.append((image_path, class_id))

    if not samples:
        raise ValueError(
            f"No BTSD images found in: {split_dir}"
        )

    return samples


def load_btsd_samples(root_dir):
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(
            f"BTSD root directory not found: {root_dir}"
        )

    training_dir = root_dir / "BelgiumTSC_Training" / "Training"
    testing_dir = root_dir / "BelgiumTSC_Testing" / "Testing"

    train_samples = _collect_samples(training_dir)
    test_samples = _collect_samples(testing_dir)

    return train_samples, test_samples


def create_btsd_datasets(
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

    train_samples, test_samples = load_btsd_samples(root_dir)

    train_data, validation_data = train_test_split(
        train_samples,
        test_size=validation_size,
        random_state=random_state,
        stratify=[label for _, label in train_samples],
    )

    train_dataset = BTSDSampleDataset(
        samples=train_data,
        transform=train_transform,
    )

    validation_dataset = BTSDSampleDataset(
        samples=validation_data,
        transform=eval_transform,
    )

    test_dataset = BTSDSampleDataset(
        samples=test_samples,
        transform=eval_transform,
    )

    return train_dataset, validation_dataset, test_dataset


def create_btsd_dataloaders(
    root_dir,
    batch_size=64,
    num_workers=0,
    validation_size=0.2,
    random_state=42,
    train_transform=None,
    eval_transform=None,
):

    train_dataset, validation_dataset, test_dataset = create_btsd_datasets(
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
