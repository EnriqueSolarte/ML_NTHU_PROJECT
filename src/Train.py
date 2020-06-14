import os
import classifier_cnn


# Create Alex net model
model = classifier_cnn.AlexNetModel()

# Visualize model
model.summary()

# General settings
Train_dir = os.path.join(os.path.abspath('..'),'data/train_images')
Test_dir = os.path.join(os.path.abspath('..'),'data/test_images')
Val_dir = os.path.join(os.path.abspath('..'),'data/val_images')
batch_size = 5
epochs = 2

# Convert dataset into batches for training
Train_batch,Test_batch,Val_batch = classifier_cnn.Images_into_batches(Train_dir, Test_dir, Val_dir, batch_size)

# Train the model
model = classifier_cnn.model_train(model, Train_batch, Val_batch, epochs, batch_size)

# Predict  
predicted = classifier_cnn.model_predict(model, Test_batch)

# Calculate accuracy
test_csv_path = os.path.join(os.path.abspath('..'),'data/train.csv')# Path of testing csv
accuracy = classifier_cnn.calculate_accuracy(predicted, test_csv_path)# check predicted and ground truth
print(accuracy)
    