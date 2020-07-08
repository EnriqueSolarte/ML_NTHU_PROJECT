import src.setup as st
from src.reading_data import Data
from src.classifier import Classifier
from tensorflow.keras.callbacks import TensorBoard
import os
from glob import glob
import tensorflow as tf


def train(cfg):
    cnn = Classifier(input_shape=cfg["shape"], batch_size=cfg["batch"])
    if cfg["model"] == "AlexNet":
        cnn.set_default_AlexNet_Model()
    else:
        arch = cfg["arch"]
        cnn.set_custom_model(conv_layers=arch["conv"],
                             dense_layers=arch["dense"])
    if cfg["random"]:
        cnn.random_boost = True

    if cfg["pre_trained"] is not None:
        dir_model = glob(os.path.join(st.DIR_LOG, "RUNNING", "*" + cfg["pre_trained"] + "*"))
        assert len(dir_model) == 1
        # model_file = os.path.join(dir_model[0], "'model-086-1.000000.ckpt")
        # cnn.model.load_weights(model_file)
        cnn.model.load_weights(os.path.join(dir_model[0], "cp-model.ckpt"))

    val_data = cnn.get_data_generator(st.DATA_VALIDATION_DIR, dip_filter=cfg["dip"])
    train_data = cnn.get_data_generator(st.DATA_TRAIN_DIR, dip_filter=cfg["dip"])
    cnn.set_log_name(cfg)
    cnn.model_select = cfg["msk"]
    cnn.lr["lr"] = cfg["lr"]
    cnn.lr["decay_steps"] = cfg["dc_st"]
    cnn.lr["decay_rate"] = cfg["dc"]
    cnn.lr["momentum"] = cfg["mt"]

    cnn.callbacks.append(TensorBoard(log_dir=os.path.join(st.DIR_LOG, "RUNNING", cnn.log_name)))
    cnn.train(gen_train=train_data, gen_val=val_data,
              epochs=cfg["epochs"])


if __name__ == '__main__':
    # Data(validation_ratio=0.25)  # * Load the dataset in our expected format from the provided by AIDEA
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = dict(shape=(224, 224),
                  batch=200,
                  dip=True,
                  random=False,
                  epochs=100,
                  lr=1e-3,
                  dc_st=10,
                  dc=0.9,
                  mt=0.9,
                  #   model="AlexNet",
                  model=None,
                  arch=dict(conv=[[96, 11, 4],
                                  [128, 3, 2]],
                            dense=[1024]),
                  msk="model0",
                  pre_trained=None,
                  extra="Model_DIP")

    train(cfg=config)
