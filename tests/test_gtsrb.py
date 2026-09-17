from pathlib import Path

from src.datasets.gtsrb import create_gtsrb_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "GTSRB"


train_loader, validation_loader, test_loader = create_gtsrb_dataloaders(
    root_dir=DATA_DIR,
    batch_size=64,
    num_workers=0,
    validation_size=0.2,
    random_state=42,
)


print("Train samples:", len(train_loader.dataset))
print("Validation samples:", len(validation_loader.dataset))
print("Test samples:", len(test_loader.dataset))


images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)

print("Pixel minimum:", images.min().item())
print("Pixel maximum:", images.max().item())

print("First 10 labels:", labels[:10])