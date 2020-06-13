import os
import CNN_models


# Create Alex net model
model = CNN_models.AlexNetModel()

# Visualize model
model.summary()

# General settings
Train_dir = os.path.join(os.path.abspath('..'),'data/train_images')
Test_dir = os.path.join(os.path.abspath('..'),'data/test_images')
Val_dir = os.path.join(os.path.abspath('..'),'data/val_images')
batch_size = 5
epochs = 2

# Convert dataset into batches for training
Train_batch,Test_batch,Val_batch = CNN_models.Images_into_batches(Train_dir, Test_dir, Val_dir, batch_size)

# Train the model
model = CNN_models.model_train(model, Train_batch, Val_batch, epochs, batch_size)

# Predict  
predicted = CNN_models.model_predict(model, Test_batch)

# Calculate accuracy
test_csv_path = os.path.join(os.path.abspath('..'),'data/train.csv')# Path of testing csv
accuracy = CNN_models.calculate_accuracy(predicted, test_csv_path)# check predicted and ground truth
print(accuracy)
    