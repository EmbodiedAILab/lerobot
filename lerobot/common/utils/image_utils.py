import numpy as np
import torch
import torch.nn.functional as F

def resize_with_pad_train(images: torch.Tensor, target_height: int, target_width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., channel, height, width] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    height, width = images.shape[-2], images.shape[-1]
    if height == target_height and width == target_width:
        return images    
    
    # scale either height or width to the given value
    scale = min(target_height / height, target_width / width)
    new_height = int(scale * height)
    new_width = int(scale * width)
    images = F.interpolate(images, size=(new_height, new_width), mode='bilinear', align_corners=False)
    
    # pad the other
    pad_width = target_width - new_width
    pad_height = target_height - new_height
    padding = [
        pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2
    ]    
    return F.pad(images, padding, value=0)