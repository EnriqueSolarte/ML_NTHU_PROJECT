import src.setup as st
from src.reading_data import Data
from src.classifier import Classifier
from tensorflow.keras.callbacks import TensorBoard
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(cfg):
    # dt = Data()  # * Load the dataset in our expected format from the provided by AIDEA

    cnn = Classifier(input_shape=cfg["shape"], batch_size=cfg["batch"])
    cnn.set_default_AlexNet_Model()

    val_data = cnn.get_data_generator(st.DATA_VALIDATION_DIR, dip_filter=cfg["dip"])
    train_data = cnn.get_data_generator(st.DATA_TRAIN_DIR, dip_filter=cfg["dip"])
    cnn.set_log_name(cfg)

    cnn.callbacks.append(TensorBoard(log_dir=os.path.join(st.DIR_LOG, "RUNNING", cnn.log_name)))
    cnn.train(gen_train=train_data, gen_val=val_data,
              epochs=cfg["epochs"])


if __name__ == '__main__':
    config = dict(shape=(224, 224),
                  batch=1,
                  dip=True,
                  epochs=1,
                  lr=1e-3,
                  model="AlexNet")

    train(cfg=config)
