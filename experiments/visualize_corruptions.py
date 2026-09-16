import os

import cv2
import matplotlib.pyplot as plt

from src.augmentations import (
    apply_corruption,
    CORRUPTIONS,
)


IMAGE_PATH = "sample.jpg"

OUTPUT_DIR = "experiments/corruption_samples"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load {IMAGE_PATH}"
    )

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


for corruption in CORRUPTIONS:

    fig, axes = plt.subplots(
        1,
        6,
        figsize=(18, 4)
    )

    axes[0].imshow(image)
    axes[0].set_title("Clean")
    axes[0].axis("off")

    for severity in range(1, 6):

        corrupted = apply_corruption(
            image,
            corruption,
            severity
        )

        axes[severity].imshow(
            corrupted
        )

        axes[severity].set_title(
            f"Severity {severity}"
        )

        axes[severity].axis("off")

    fig.suptitle(
        corruption.replace("_", " ").title()
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{corruption}.pdf"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )