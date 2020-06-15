import os
import classifier_cnn
from setup import *
from reading_data import Data

# * This line sort the dataset into train_images, validation_images, and testing_images
dt = Data()
# Create Alex net model
model = classifier_cnn.AlexNetModel()

# Visualize model
model.summary()

# General settings
# Train_dir = os.path.join(os.path.abspath('..'),'data/train_images')
# Test_dir = os.path.join(os.path.abspath('..'),'data/test_images')
# Val_dir = os.path.join(os.path.abspath('..'),'data/val_images')
train_dir = DATA_TRAIN_DIR
test_dir = DATA_TEST_DIR
val_dir = DATA_VALIDATION_DIR
batch_size = 5
epochs = 2

# Convert dataset into batches for training
Train_batch, Test_batch, Val_batch = classifier_cnn.Images_into_batches(train_dir, test_dir, val_dir, batch_size)

# Train the model
model = classifier_cnn.model_train(model, Train_batch, Val_batch, epochs, batch_size)

# Predict
# ! predition doesn't work
predicted = classifier_cnn.model_predict(model, Test_batch)

# Calculate accuracy
test_csv_path = os.path.join(os.path.abspath('..'), 'data/train.csv')  # Path of testing csv
accuracy = classifier_cnn.calculate_accuracy(predicted, test_csv_path)  # check predicted and ground truth
print(accuracy)
