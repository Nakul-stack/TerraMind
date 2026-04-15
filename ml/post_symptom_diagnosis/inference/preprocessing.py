"""
Image preprocessing for the Post-Symptom Diagnosis model.

Mirrors the training-time transform pipeline (without augmentation).
Uses the image size from class_metadata_fast.json.
"""

import io
import logging
from PIL import Image
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet normalisation statistics used during training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_inference_transform(img_size: int) -> transforms.Compose:
    """
    Build the deterministic inference transform pipeline.

    Parameters
    ----------
    img_size : int
        Target spatial resolution (e.g. 192 from the metadata).

    Returns
    -------
    torchvision.transforms.Compose
        A composed transform ready for inference (no random augmentation).
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image_bytes: bytes, img_size: int) -> torch.Tensor:
    """
    Convert raw image bytes into a model-ready tensor.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image file.
    img_size : int
        Target spatial resolution from model metadata.

    Returns
    -------
    torch.Tensor
        A tensor of shape (1, 3, img_size, img_size), ready for inference.

    Raises
    ------
    ValueError
        If the image cannot be opened or is corrupt.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.error("Failed to open image: %s", exc)
        raise ValueError(f"Corrupt or unreadable image file: {exc}") from exc

    transform = build_inference_transform(img_size)
    tensor = transform(image)  # (3, H, W)
    tensor = tensor.unsqueeze(0)  # (1, 3, H, W) — batch dimension
    logger.debug("Image preprocessed to tensor shape %s", tensor.shape)
    return tensor
