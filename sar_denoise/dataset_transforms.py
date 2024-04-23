import numpy as np
from PIL import Image
from torchvision import transforms


def normalize_u8(img):
    img = np.array(img)
    img = (img - img.min()) / np.maximum(1.0, img.max() - img.min())
    return (255 * img).astype(np.uint8)


def to_u8(img):
    return Image.fromarray(normalize_u8(np.array(img)))


def to_log_u8(img):
    arr = np.array(img)
    arr -= np.min(arr - 1)
    return Image.fromarray(normalize_u8(np.log10(arr)))


def make_transform_images(
    resolution: int = 64, center_crop: bool = False, random_flip: bool = False
):
    """
    Make function that can be used to transform a dataset of images.

    Usage:
    ```python
    transform_images = make_transform_images(resolution=64, center_crop=False, random_flip=False)
    dataset.set_transform(transform_images)
    ```
    """
    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            # NOTE EDF all images will already be desired size, so make custom transform
            # that just ensures that they are in fact that size
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            (
                transforms.CenterCrop(resolution)
                if center_crop
                else transforms.RandomCrop(resolution)
            ),
            (
                transforms.RandomHorizontalFlip()
                if random_flip
                else transforms.Lambda(lambda x: x)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("L")) for image in examples["image"]]
        return {"input": images}

    return transform_images