# Shreyas Prasad
# 10/18/23
# CS 7180: Advanced Perception

import cv2
import numpy as np
from PIL import Image

class CLAHETransform:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Args:
        clip_limit (float): Threshold for contrast limiting. Higher values result in more contrast.
        tile_grid_size (tuple): Size of the grid for histogram equalization. Higher values result in more fine-grained equalization.

    Returns:
        PIL.Image: The transformed image.

    Example:
        transform = CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))
        transformed_img = transform(img)
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        # Convert PIL image to numpy array
        np_img = np.array(img)

        # Check if the image is grayscale or RGB
        if len(np_img.shape) == 2 or np_img.shape[2] == 1:
            # Apply CLAHE to grayscale image
            np_img = self.clahe.apply(np_img)
        else:
            # Convert RGB to LAB
            lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L-channel
            cl = self.clahe.apply(l)

            # Merge the CLAHE enhanced L-channel back with A and B channels
            clab = cv2.merge((cl, a, b))

            # Convert LAB back to RGB
            np_img = cv2.cvtColor(clab, cv2.COLOR_LAB2RGB)

        # Convert numpy array back to PIL image
        img = Image.fromarray(np_img)

        return img
