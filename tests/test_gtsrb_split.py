from pathlib import Path

from torchvision import transforms

from src.datasets.gtsrb import create_gtsrb_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "GTSRB"


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


train_dataset, validation_dataset, test_dataset = create_gtsrb_datasets(
    root_dir=DATA_DIR,
    validation_size=0.2,
    random_state=42,
    train_transform=transform,
    eval_transform=transform,
)


def get_labels(dataset):
    return set(dataset.data["ClassId"].astype(int))


train_classes = get_labels(train_dataset)
validation_classes = get_labels(validation_dataset)
test_classes = get_labels(test_dataset)

expected_classes = set(range(43))


print("Train samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))
print("Test samples:", len(test_dataset))

print("\nNumber of classes:")
print("Train:", len(train_classes))
print("Validation:", len(validation_classes))
print("Test:", len(test_classes))

print("\nMissing classes:")
print("Train:", expected_classes - train_classes)
print("Validation:", expected_classes - validation_classes)
print("Test:", expected_classes - test_classes)

assert train_classes == expected_classes
assert validation_classes == expected_classes
assert test_classes == expected_classes

print("\nAll 43 classes are present in all three splits.")