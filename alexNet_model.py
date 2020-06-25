import src.setup as config
from src.reading_data import Data
from src.classifier import Classifier
from tensorflow.keras.callbacks import TensorBoard
import os

# dt = Data()  # * Load the dataset in our expected format from the provided by AIDEA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cnn = Classifier(input_shape=(224, 224), batch_size=10)
cnn.set_default_AlexNet_Model()

val_data = cnn.get_data_generator(config.DATA_VALIDATION_DIR, enhancement=False)
train_data = cnn.get_data_generator(config.DATA_TRAIN_DIR, enhancement=False)
NAME = cnn.get_name() + "reorder_cnvs"

tensorboard = TensorBoard(log_dir=os.path.join(config.DATA_LOG, "RUNNING", NAME))
cnn.train(gen_train=train_data, gen_val=val_data,
          epochs=500, callbacks=tensorboard)
print("done")
