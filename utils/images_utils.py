"""
Utility functions for image processing
"""
import numpy as np
import cv2
from omegaconf import DictConfig
import matplotlib.pyplot as plt


def read_image(img_path, color_mode):
    """
    Read and return image as np array from given path.
    In case of color image, it returns image in BGR mode.
    """
    return cv2.imread(img_path, color_mode)


def resize_image(img, height, width, resize_method=cv2.INTER_CUBIC):
    """
    Resize image
    """
    return cv2.resize(img, dsize=(width, height), interpolation=resize_method)


def prepare_image(path: str, resize: DictConfig, normalize_type: str):
    """
    Prepare image for model.
    read image --> resize --> normalize --> return as float32
    """
    image = read_image(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if resize.VALUE:
        # TODO verify image resizing method
        image = resize_image(image, resize.HEIGHT, resize.WIDTH, cv2.INTER_AREA)

    if normalize_type == "normalize":
        image = image / 255.0

    image = image.astype(np.float32)

    return image


def prepare_mask(path: str, resize: dict, normalize_mask: dict):
    """
        Prepare mask for model.
        read mask --> resize --> normalize --> return as int32
        """
    mask = read_image(path, cv2.IMREAD_COLOR)

    if resize.VALUE:
        mask = resize_image(mask, resize.HEIGHT, resize.WIDTH, cv2.INTER_NEAREST)


    mask = mask.astype(np.int32)

    return mask


def image_to_mask_name(image_name: str):
    """
    Convert image file name to it's corresponding mask file name e.g.
    image name     -->     mask name
    image_28_0.png         mask_28_0.png
    replace image with mask
    """

    return image_name.replace('image', 'mask')


def postprocess_mask(mask):
    """
    Post process model output.
    Covert probabilities into indexes based on maximum value.
    """
    mask = np.argmax(mask, axis=-1)
    return mask.astype(np.int32)


def denormalize_mask(mask, classes):
    """
    Denormalize mask by multiplying each class with higher
    integer (255 / classes) for better visualization.
    You can modify it to assign specific RGB colors for each class.
    """
    # Define a color palette for the classes
    color_map = {
        0: [0, 255, 0],       # Class 0: Green
        1: [124, 10, 4],      # Class 1: Red
        # Add more classes if needed
    }

    # Initialize the denormalized mask with the shape (height, width, 3) for RGB
    denormalized_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Vectorized approach to map classes to colors
    for class_id, color in color_map.items():
        denormalized_mask[mask == class_id] = color

    # If there are any undefined classes, they will be black by default (already initialized as [0, 0, 0])

    return denormalized_mask


def display(display_list, show_true_mask=False):
    """
    Show list of images. it could be
    either [image, true_mask, predicted_mask] or [image, predicted_mask].
    Set show_true_mask to True if true mask is available or vice versa
    """
    if show_true_mask:
        title_list = ('Input Image', 'True Mask', 'Predicted Mask')
        plt.figure(figsize=(12, 4))
    else:
        title_list = ('Input Image', 'Predicted Mask')
        plt.figure(figsize=(8, 4))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        if title_list is not None:
            plt.title(title_list[i])
        if len(np.squeeze(display_list[i]).shape) == 2:
            plt.imshow(np.squeeze(display_list[i]), cmap='gray')
            plt.axis('on')
        else:
            plt.imshow(np.squeeze(display_list[i]))
            plt.axis('on')
    plt.show()
