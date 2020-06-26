import src
import src.setup as config
from src.reading_data import Data
from src.classifier import Classifier
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# dt = Data()  # * Load the dataset in our expected format from the provided by AIDEA
cnn = Classifier(input_shape=(224, 224), batch_size=10)
conv_layers = [(96, 5, 1), (256, 5, 1), (386, 3, 1), (256, 3, 1)]
dense_layers = [1024, 1024, 100]

cnn.set_custom_model(conv_layers=conv_layers, dense_layers=dense_layers)
NAME = cnn.get_name()
tensorboard = TensorBoard(log_dir=os.path.join(config.DIR_LOG, "RUNNING", NAME))

val_data = cnn.get_data_generator(config.DATA_VALIDATION_DIR, dip_filter=True)
train_data = cnn.get_data_generator(config.DATA_TRAIN_DIR, dip_filter=True)

cnn.train(gen_train=train_data, gen_val=val_data,
          epochs=50, callbacks=tensorboard)
print("done")
