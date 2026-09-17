import torch
from PIL import Image

from src.datasets.preprocessing import (
    IMAGE_SIZE,
    get_eval_transform,
    get_train_transform,
)


def test_image_size():
    assert IMAGE_SIZE == (32, 32)


def test_train_transform():
    transform = get_train_transform()

    image = Image.new("RGB", (64, 64))
    output = transform(image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 32, 32)


def test_eval_transform():
    transform = get_eval_transform()

    image = Image.new("RGB", (64, 64))
    output = transform(image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 32, 32)


if __name__ == "__main__":
    test_image_size()
    test_train_transform()
    test_eval_transform()

    print("All preprocessing tests passed.")