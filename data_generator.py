"""
Data generator class.
"""
import tensorflow as tf
import numpy as np
import os
from omegaconf import DictConfig

from utils.general_utils import join_paths, get_data_paths
from utils.images_utils import prepare_image, prepare_mask, image_to_mask_name


def assign_class_to_mask(mask, cfg):
    """
    주어진 마스크에서 각 색상값을 클래스 번호로 할당합니다.
    """
    class_map = {
        tuple(v): k for k, v in cfg.CLASS_COLOR_MAP.items()
    }

    # 마스크 이미지에서 각 색상값을 찾아 해당 클래스 번호를 할당합니다.
    class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)

    for color, class_id in class_map.items():
        # 특정 색상과 일치하는 픽셀을 찾아서 해당 클래스 ID로 설정
        color_mask = np.all(mask == color, axis=-1)
        class_mask[color_mask] = class_id

    # print("class_mask = ", class_mask)
    return class_mask


class DataGenerator(tf.keras.utils.Sequence):
    """
    Generate batches of data for model by reading images and their
    corresponding masks.
    """

    def __init__(self, cfg: DictConfig, mode: str):
        """
        Initialization
        """
        self.cfg = cfg
        self.mode = mode
        self.batch_size = self.cfg.HYPER_PARAMETERS.BATCH_SIZE
        np.random.seed(cfg.SEED)

        # Check if mask is available or not
        self.mask_available = False if cfg.DATASET[mode].MASK_PATH is None or str(
            cfg.DATASET[mode].MASK_PATH).lower() == "none" else True

        data_paths = get_data_paths(cfg, mode, self.mask_available)

        self.images_paths = data_paths[0]
        if self.mask_available:
            self.mask_paths = data_paths[1]

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        self.on_epoch_end()
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.images_paths))
        if self.cfg.PREPROCESS_DATA.SHUFFLE[self.mode].VALUE:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        indexes = self.indexes[
                  index * self.batch_size:(index + 1) * self.batch_size
                  ]
        return self.__data_generation(indexes)

    def __data_generation(self, indexes):
        """
        Generates batch data
        """
        batch_images = np.zeros((
            self.cfg.HYPER_PARAMETERS.BATCH_SIZE,
            self.cfg.INPUT.HEIGHT,
            self.cfg.INPUT.WIDTH,
            self.cfg.INPUT.CHANNELS
        )).astype(np.float32)

        if self.mask_available:
            batch_masks = np.zeros((
                self.cfg.HYPER_PARAMETERS.BATCH_SIZE,
                self.cfg.INPUT.HEIGHT,
                self.cfg.INPUT.WIDTH,
                self.cfg.OUTPUT.CLASSES
            )).astype(np.float32)

        for i, index in enumerate(indexes):
            img_path = self.images_paths[int(index)]
            if self.mask_available:
                mask_path = self.mask_paths[int(index)]

            image = prepare_image(
                img_path,
                self.cfg.PREPROCESS_DATA.RESIZE,
                self.cfg.PREPROCESS_DATA.IMAGE_PREPROCESSING_TYPE,
            )

            if self.mask_available:
                mask = prepare_mask(
                    mask_path,
                    self.cfg.PREPROCESS_DATA.RESIZE,
                    self.cfg.PREPROCESS_DATA.NORMALIZE_MASK,
                )

                # 색상-클래스 매핑을 적용하여 마스크 변환
                mask = assign_class_to_mask(mask, self.cfg)

            # numpy to tensorflow conversion
            if self.mask_available:
                image, mask = tf.numpy_function(
                    self.tf_func,
                    [image, mask],
                    [tf.float32, tf.int32]
                )
            else:
                image = tf.numpy_function(
                    self.tf_func,
                    [image, ],
                    [tf.float32, ]
                )

            # set shape attributes which were lost during tf conversion
            image.set_shape([
                self.cfg.INPUT.HEIGHT,
                self.cfg.INPUT.WIDTH,
                self.cfg.INPUT.CHANNELS
            ])
            batch_images[i] = image

            if self.mask_available:
                mask = tf.one_hot(
                    mask,
                    self.cfg.OUTPUT.CLASSES,
                    dtype=tf.int32
                )
                mask.set_shape([
                    self.cfg.INPUT.HEIGHT,
                    self.cfg.INPUT.WIDTH,
                    self.cfg.OUTPUT.CLASSES
                ])
                batch_masks[i] = mask

        if self.mask_available:
            return batch_images, batch_masks
        else:
            return batch_images,

    @staticmethod
    def tf_func(*args):
        return args
