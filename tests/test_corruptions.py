import numpy as np
import pytest

from src.augmentations import (
    apply_corruption,
    CORRUPTIONS,
)


@pytest.fixture
def sample_image():
    image = np.random.randint(
        0,
        256,
        (64, 64, 3),
        dtype=np.uint8
    )

    return image


def test_all_corruptions_exist():
    assert len(CORRUPTIONS) == 14


@pytest.mark.parametrize(
    "corruption",
    CORRUPTIONS
)
def test_corruption_severity_1(
    sample_image,
    corruption
):

    result = apply_corruption(
        sample_image,
        corruption,
        1
    )

    assert isinstance(
        result,
        np.ndarray
    )

    assert result.shape == sample_image.shape

    assert result.dtype == np.uint8


@pytest.mark.parametrize(
    "corruption",
    CORRUPTIONS
)
def test_corruption_severity_5(
    sample_image,
    corruption
):

    result = apply_corruption(
        sample_image,
        corruption,
        5
    )

    assert result.shape == sample_image.shape

    assert result.dtype == np.uint8

    assert np.min(result) >= 0
    assert np.max(result) <= 255


def test_invalid_severity(sample_image):

    with pytest.raises(ValueError):

        apply_corruption(
            sample_image,
            "gaussian",
            6
        )


def test_invalid_corruption(sample_image):

    with pytest.raises(ValueError):

        apply_corruption(
            sample_image,
            "random_noise",
            3
        )