import src
import src.setup as config
from src.reading_data import Data
from src.classifier import Classifier
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# dt = Data()  # * Load the dataset in our expected format from the provided by AIDEA
cnn = Classifier(input_shape=(224, 224), batch_size=10)
conv_layers = [(96, 11, 4), (256, 5, 1), (256, 3, 1)]
dense_layers = [1000]

cnn.set_custom_model(conv_layers=conv_layers, dense_layers=dense_layers)
NAME = cnn.get_name()
tensorboard = TensorBoard(log_dir=os.path.join(config.DATA_LOG, "RUNNING", NAME))

val_data = cnn.get_data_generator(config.DATA_VALIDATION_DIR, enhancement=True)
train_data = cnn.get_data_generator(config.DATA_TRAIN_DIR, enhancement=True)

cnn.train(gen_train=train_data, gen_val=val_data,
          epochs=50, callbacks=tensorboard)
print("done")
