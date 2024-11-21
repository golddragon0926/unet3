"""
Returns Unet3+ model
"""
from omegaconf import DictConfig

from .unet3plus import unet3plus, tiny_unet3plus
from .unet3plus_deep_supervision import unet3plus_deepsup
from .unet3plus_deep_supervision_cgm import unet3plus_deepsup_cgm


def prepare_model(cfg: DictConfig, training=False):
    """
    Creates and return model object based on given model type.
    """

    input_shape = [cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH, cfg.INPUT.CHANNELS]

    if cfg.MODEL.TYPE == "tiny_unet3plus":
        return tiny_unet3plus(
            input_shape,
            cfg.OUTPUT.CLASSES,
            training
        )
    elif cfg.MODEL.TYPE == "unet3plus":
        #  training parameter does not matter in this case
        return unet3plus(
            input_shape,
            cfg.OUTPUT.CLASSES
        )
    elif cfg.MODEL.TYPE == "unet3plus_deepsup":
        return unet3plus_deepsup(
            input_shape,
            cfg.OUTPUT.CLASSES,
            training
        )
    elif cfg.MODEL.TYPE == "unet3plus_deepsup_cgm":
        if cfg.OUTPUT.CLASSES != 1:
            raise ValueError(
                "UNet3+ with Deep Supervision and Classification Guided Module"
                "\nOnly works when model output classes are equal to 1"
            )
        return unet3plus_deepsup_cgm(
            input_shape,
            cfg.OUTPUT.CLASSES,
            training
        )
    else:
        raise ValueError(
            "Wrong model type passed."
            "\nPlease check config file for possible options."
        )
