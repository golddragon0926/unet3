"""
Prediction script used to visualize model output
"""
import os
import hydra
import cv2
from omegaconf import DictConfig

import data_generator
from utils.general_utils import join_paths
from utils.images_utils import display
from utils.images_utils import postprocess_mask, denormalize_mask
from models.model import prepare_model


def predict(cfg: DictConfig):
    """
    Predict and visualize given data
    """

    # set batch size to one
    cfg.HYPER_PARAMETERS.BATCH_SIZE = 1

    # data generator
    val_generator = data_generator.DataGenerator(cfg, mode="VAL")

    # create model
    model = prepare_model(cfg)

    # weights model path
    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.PATH,
        f"{cfg.MODEL.WEIGHTS_FILE_NAME}.hdf5"
    )

    assert os.path.exists(checkpoint_path), \
        f"Model weight's file does not exist at \n{checkpoint_path}"

    # load model weights
    model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
    # model.summary()

    # check mask are available or not
    mask_available = True
    if cfg.DATASET.VAL.MASK_PATH is None or \
            str(cfg.DATASET.VAL.MASK_PATH).lower() == "none":
        mask_available = False

    showed_images = 0
    save_dir = cfg.SAVE_DIR  # 예측 결과를 저장할 디렉토리 경로
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    for batch_data in val_generator:  # for each batch
        batch_images = batch_data[0]
        if mask_available:
            batch_mask = batch_data[1]

        # make prediction on batch
        batch_predictions = model.predict_on_batch(batch_images)
        if len(model.outputs) > 1:
            batch_predictions = batch_predictions[0]

        for index in range(len(batch_images)):

            image = batch_images[index]  # for each image
            if cfg.SHOW_CENTER_CHANNEL_IMAGE:
                # for UNet3+ show only center channel as image
                image = image[:, :, 1]

            # do postprocessing on predicted mask
            prediction = batch_predictions[index]
            prediction = postprocess_mask(prediction)

            # denormalize mask for better visualization
            prediction = denormalize_mask(prediction, cfg)

            if mask_available:
                mask = batch_mask[index]
                mask = postprocess_mask(mask)
                mask = denormalize_mask(mask, cfg)

            cv2.imwrite(os.path.join(save_dir, f"prediction_{showed_images}.png"), prediction)

            showed_images += 1
        # stop after displaying below number of images
        # if showed_images >= 10: break


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Read config file and pass to prediction method
    """
    predict(cfg)


if __name__ == "__main__":
    main()
