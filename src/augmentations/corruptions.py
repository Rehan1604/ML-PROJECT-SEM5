# src/augmentations/corruptions.py

import io
import cv2
import numpy as np
from PIL import Image

from .severity import (
    CORRUPTIONS,
    get_parameters,
    validate_corruption,
)


def _clip_image(image):
    """
    Ensure image is uint8 in range [0, 255].
    """
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


# ---------------------------------------------------------
# 1. GAUSSIAN NOISE
# ---------------------------------------------------------

def gaussian_noise(image, severity):
    params = get_parameters("gaussian", severity)

    sigma = params["sigma"] * 255.0

    noise = np.random.normal(
        0,
        sigma,
        image.shape
    )

    return _clip_image(image.astype(np.float32) + noise)


# ---------------------------------------------------------
# 2. IMPULSE / SALT-AND-PEPPER NOISE
# ---------------------------------------------------------

def impulse_noise(image, severity):
    params = get_parameters("impulse", severity)

    amount = params["amount"]

    result = image.copy()

    random_matrix = np.random.random(image.shape[:2])

    salt = random_matrix < amount / 2
    pepper = random_matrix > (1 - amount / 2)

    result[salt] = 255
    result[pepper] = 0

    return result


# ---------------------------------------------------------
# 3. SHOT NOISE
# ---------------------------------------------------------

def shot_noise(image, severity):
    params = get_parameters("shot", severity)

    scale = params["scale"]

    image_float = image.astype(np.float32) / 255.0

    noisy = np.random.poisson(
        image_float * (1.0 / scale)
    ) * scale

    noisy = noisy * 255.0

    return _clip_image(noisy)


# ---------------------------------------------------------
# 4. SNOW
# ---------------------------------------------------------

def snow(image, severity):
    params = get_parameters("snow", severity)

    intensity = params["intensity"]

    result = image.copy().astype(np.float32)

    h, w = image.shape[:2]

    num_flakes = int(h * w * intensity * 0.08)

    ys = np.random.randint(0, h, num_flakes)
    xs = np.random.randint(0, w, num_flakes)

    for y, x in zip(ys, xs):

        radius = np.random.randint(1, 4)

        cv2.circle(
            result,
            (x, y),
            radius,
            (255, 255, 255),
            -1
        )

    return _clip_image(result)


# ---------------------------------------------------------
# 5. FOG
# ---------------------------------------------------------

def fog(image, severity):
    params = get_parameters("fog", severity)

    strength = params["strength"]

    h, w = image.shape[:2]

    overlay = np.ones_like(image, dtype=np.float32) * 255

    # Smooth spatial mask
    noise = np.random.rand(h, w).astype(np.float32)

    noise = cv2.GaussianBlur(
        noise,
        (0, 0),
        sigmaX=max(h, w) / 8
    )

    noise = cv2.normalize(
        noise,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    alpha = noise[..., None] * strength

    image_float = image.astype(np.float32)

    result = (
        image_float * (1 - alpha)
        + overlay * alpha
    )

    return _clip_image(result)


# ---------------------------------------------------------
# 6. FROST
# ---------------------------------------------------------

def frost(image, severity):
    params = get_parameters("frost", severity)

    strength = params["strength"]

    h, w = image.shape[:2]

    frost_layer = np.random.normal(
        220,
        25,
        (h, w, 3)
    )

    frost_layer = np.clip(
        frost_layer,
        0,
        255
    )

    # Blur frost texture
    frost_layer = cv2.GaussianBlur(
        frost_layer.astype(np.float32),
        (0, 0),
        5
    )

    result = (
        image.astype(np.float32) * (1 - strength)
        + frost_layer * strength
    )

    return _clip_image(result)


# ---------------------------------------------------------
# 7. MOTION BLUR
# ---------------------------------------------------------

def motion_blur(image, severity):
    params = get_parameters("motion_blur", severity)

    kernel_size = params["kernel_size"]

    kernel = np.zeros(
        (kernel_size, kernel_size)
    )

    kernel[kernel_size // 2, :] = 1

    kernel /= kernel_size

    return cv2.filter2D(
        image,
        -1,
        kernel
    )


# ---------------------------------------------------------
# 8. ZOOM BLUR
# ---------------------------------------------------------

def zoom_blur(image, severity):
    params = get_parameters("zoom_blur", severity)

    strength = params["strength"]

    h, w = image.shape[:2]

    result = image.astype(np.float32)

    num_steps = 5

    for i in range(1, num_steps + 1):

        scale = 1.0 + (
            strength * i / num_steps
        )

        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        y1 = (new_h - h) // 2
        x1 = (new_w - w) // 2

        cropped = resized[
            y1:y1 + h,
            x1:x1 + w
        ]

        result += cropped.astype(np.float32)

    result /= (num_steps + 1)

    return _clip_image(result)


# ---------------------------------------------------------
# 9. DEFOCUS BLUR
# ---------------------------------------------------------

def defocus_blur(image, severity):
    params = get_parameters("defocus_blur", severity)

    kernel_size = params["kernel_size"]

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


# ---------------------------------------------------------
# 10. BRIGHTNESS
# ---------------------------------------------------------

def brightness(image, severity):
    params = get_parameters("brightness", severity)

    factor = params["factor"]

    # Increasing severity makes image progressively brighter
    result = image.astype(np.float32)

    result = result + (
        255.0 * factor * 0.5
    )

    return _clip_image(result)


# ---------------------------------------------------------
# 11. CONTRAST
# ---------------------------------------------------------

def contrast(image, severity):
    params = get_parameters("contrast", severity)

    factor = params["factor"]

    result = image.astype(np.float32)

    mean = np.mean(
        result,
        axis=(0, 1),
        keepdims=True
    )

    # Increasing severity reduces contrast
    result = mean + (
        result - mean
    ) * (1.0 - factor)

    return _clip_image(result)


# ---------------------------------------------------------
# 12. PIXELATE
# ---------------------------------------------------------

def pixelate(image, severity):
    params = get_parameters("pixelate", severity)

    scale = params["scale"]

    h, w = image.shape[:2]

    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))

    small = cv2.resize(
        image,
        (small_w, small_h),
        interpolation=cv2.INTER_LINEAR
    )

    result = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return result


# ---------------------------------------------------------
# 13. ELASTIC TRANSFORM
# ---------------------------------------------------------

def elastic_transform(image, severity):
    params = get_parameters(
        "elastic_transform",
        severity
    )

    alpha = params["alpha"]
    sigma = params["sigma"]

    h, w = image.shape[:2]

    dx = np.random.uniform(
        -1,
        1,
        (h, w)
    ).astype(np.float32)

    dy = np.random.uniform(
        -1,
        1,
        (h, w)
    ).astype(np.float32)

    dx = cv2.GaussianBlur(
        dx,
        (0, 0),
        sigma
    )

    dy = cv2.GaussianBlur(
        dy,
        (0, 0),
        sigma
    )

    dx *= alpha
    dy *= alpha

    x, y = np.meshgrid(
        np.arange(w),
        np.arange(h)
    )

    map_x = (
        x.astype(np.float32) + dx
    )

    map_y = (
        y.astype(np.float32) + dy
    )

    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


# ---------------------------------------------------------
# 14. JPEG COMPRESSION
# ---------------------------------------------------------

def jpeg_compression(image, severity):
    params = get_parameters(
        "jpeg_compression",
        severity
    )

    quality = params["quality"]

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality
        ]
    )

    if not success:
        raise RuntimeError(
            "JPEG compression failed."
        )

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )

    return decoded


# ---------------------------------------------------------
# CORRUPTION DISPATCHER
# ---------------------------------------------------------

_CORRUPTION_FUNCTIONS = {
    "gaussian": gaussian_noise,
    "impulse": impulse_noise,
    "shot": shot_noise,
    "snow": snow,
    "fog": fog,
    "frost": frost,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "defocus_blur": defocus_blur,
    "brightness": brightness,
    "contrast": contrast,
    "pixelate": pixelate,
    "elastic_transform": elastic_transform,
    "jpeg_compression": jpeg_compression,
}


def apply_corruption(
    image,
    corruption,
    severity
):
    """
    Apply one corruption at a specified severity.

    Parameters
    ----------
    image : np.ndarray
        RGB or BGR uint8 image.

    corruption : str
        Name of corruption.

    severity : int
        Integer from 1 to 5.

    Returns
    -------
    np.ndarray
        Corrupted image.
    """

    validate_corruption(
        corruption,
        severity
    )

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "image must be a NumPy array."
        )

    if image.dtype != np.uint8:
        image = _clip_image(image)

    corrupted = _CORRUPTION_FUNCTIONS[
        corruption
    ](
        image.copy(),
        severity
    )

    return _clip_image(corrupted)


def list_corruptions():
    """Return all supported corruption names."""
    return list(CORRUPTIONS)