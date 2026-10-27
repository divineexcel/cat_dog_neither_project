"""Inference-time image views shared by the app and command-line classifier."""

from __future__ import annotations

from PIL import Image


def model_views(image: Image.Image, target_size: tuple[int, int]) -> list[Image.Image]:
    """Make square, aspect-preserving views to avoid distorting wide scenes."""
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)

    if width > height * 1.15:
        max_offset = width - side
        offsets = (0, max_offset // 2, max_offset)
        crops = [image.crop((left, 0, left + side, side)) for left in offsets]
    elif height > width * 1.15:
        max_offset = height - side
        offsets = (0, max_offset // 2, max_offset)
        crops = [image.crop((0, top, side, top + side)) for top in offsets]
    else:
        crops = [image]

    return [crop.resize(target_size, Image.Resampling.BILINEAR) for crop in crops]
