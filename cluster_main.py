#Imports
import keras #code only works with this import!?
import os
import glob
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
#import timm
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, VGG19, ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet101
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
from segmentation_models_pytorch import Unet
from tqdm import tqdm
from PIL import Image
import copy
import random  # For random sampling
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score
import copy
from cluster_helper import load_and_recreate_masked_images, preprocess_masked_images, segment_fish, apply_segmentation_model, fill_holes, grow_mask, balance_dataframe, create_data_and_labels

seed = 42
random.seed(seed)
np.random.seed(seed)

User="Mark" #Set to Mark if you are Mark XD

if User=="Mark":
    
    load_with_split = True #Load data with split or without split use the exel file
    excel_path = "Df_Zebrafish_with_splits.xlsx" #change if Load with split == True

    #Df_Zebrafish_with_splits.xlsx
    
    images_folder = "C:/Users/ma405l/Documents/Heidelberg_Schweregrad/Full_data/Raw_data_full_train"
    masks_folder = "C:/Users/ma405l/Documents/Heidelberg_Schweregrad/Full_data/Masked_images"
    #2. Set parameters for data processing
    target_size= (256,256)	#Size of the images for training
    num_classes = 4

    #3. Train, Val, Test Split
    skip_train = False
    retrain = True
    train_size = 0.6
    val_size = 0.2
    test_size = 0.2
    label_name = "Curved"

    #4. Balance datasets
    balance_train = True
    balance_val = False
    balance_test = False

    Model_type = "CNN" #Set to "Transformer" or "CNN"

    # Segmentation
    Model_seg = False
    num_epochs_seg = 1
    criterion_seg = torch.nn.BCEWithLogitsLoss()
    seg_directory = "Models/Segmentation"
    seg_train = False

    # HP Tuning
    hp_tuning = True
    hp_dense_layer = [512]
    hp_lr = [2e-5] #Learning rate for Adam optimizer

    #Use trained model for mask segmentation of images
    use_seg_model = True
    trained_seg_model = f"{seg_directory}/Segmentation/seg_model.pth"

    #5 Transformer. Augmentation parameters
    trans_rotation_range = 45
    trans_width_shift_range = 0.2
    trans_height_shift_range = 0.2
    trans_zoom_range = 0.1
    trans_horizontal_flip  = True
    trans_fill_mode="nearest"

    #6 Transformer. Training parameters
    trans_learning_rate = 0.00008 # Define learing rate
    trans_num_epochs = 5 # Define the number of epochs

    #7 Transformer. Save parameters
    trans_log_directory = f"Models/Transformer"
    trans_model_name = "trans_model.pth"
    trans_metrics_name = "trans_metrics.txt"

    #5 CNN. Augmentation parameters
    cnn_rotation_range = 45
    cnn_width_shift_range = 0.1
    cnn_height_shift_range = 0.1
    cnn_zoom_range = 0.1
    cnn_horizontal_flip  = True
    cnn_fill_mode="nearest"

    #6 CNN. Training parameters
    
    train_from_scratch = False #Train from scratch (vgg_16) or use pre-trained model (already trained)
    
    cnn_learning_rate = 0.001
    cnn_loss = 'crossentropy'
    cnn_num_epochs = 50
    cnn_num_epochs_pre = 5
    dense_layer = 512
    dropout = 0.3

    #7 CNN. Save parameters
    cnn_log_directory = "Models/CNN"
    cnn_model_name = "best_model.keras"
    cnn_metrics_name = "vgg_16_metrics.json"


    

# Load the data and recreate masked images
df_result = load_and_recreate_masked_images(excel_path)

#Preprocess masked images
df_result = preprocess_masked_images(df_result, target_size)

# Apply the segmentation model to the images and create Segmented Masks and Confidence Maps columns

df_result = apply_segmentation_model(df_result, seg_directory, target_size)

df_result["Filled Masks"] = df_result["Segmented Masks"].apply(fill_holes)

#Apply the function to the DataFrame (1 or 3 interations)
df_result["Grown Masks"] = df_result["Filled Masks"].apply(grow_mask, iterations=3, kernel_size=3)

#Apply segmented masks to the images and change columen name of masked images
#Preprocess masked images and create the column "processed_image"
masked_images = []

for idx, row in df_result.iterrows():
    img_path = row["Images"]
    #!!!currently the origninal masks are used!!!
    mask = row["Grown Masks"]

    # Load the original image
    original_image = cv2.imread(img_path)  # Load raw image
    
    # Resize image to match mask size
    original_image = cv2.resize(original_image, target_size, interpolation=cv2.INTER_LINEAR)

    # Ensure mask is in correct format
    if isinstance(mask, str):  # If it's a file path
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    elif isinstance(mask, Image.Image):  # If it's a PIL image
        mask = np.array(mask)

    elif isinstance(mask, np.ndarray):
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Ensure mask is binary (0 or 255)
    mask = (mask > 0).astype(np.uint8)

    # Convert mask to 3 channels (so it can be applied to RGB image)
    mask_3ch = np.stack([mask] * 3, axis=-1)

    # Apply the mask to the image
    masked_image = original_image * mask_3ch

    # Append result
    masked_images.append(masked_image)

# Overwrite the "Masked Images" column
df_result["Masked Images"] = masked_images

df_result = preprocess_masked_images(df_result, target_size)

#3) Train, Val, Test Split saved in 3 new DataFrames
if load_with_split and label_name == "Curved":
    
    # Create separate DataFrames based on "split_by_curve"
    df_train = df_result[(df_result["split_by_curve"] == 0) | (df_result["split_by_curve"] == 1)].reset_index(drop=True)
    df_val = df_result[df_result["split_by_curve"] == 2].reset_index(drop=True)
    df_test = df_result[df_result["split_by_curve"] == 2].reset_index(drop=True)

elif load_with_split and label_name != "Curved":
    
    # Create separate DataFrames based on "split_by_edema"
    df_train = df_result[df_result["split_by_edema"] == 0].reset_index(drop=True)
    df_val = df_result[df_result["split_by_edema"] == 1].reset_index(drop=True)
    df_test = df_result[df_result["split_by_edema"] == 2].reset_index(drop=True)


#4) Balance the dataframes



#Toggle to balance train, val and test dataframes
if balance_train==True:
    df_train = balance_dataframe(df_train, label_name)
if balance_val==True:
    df_val = balance_dataframe(df_val, label_name)
if balance_test==True:
    df_test = balance_dataframe(df_test, label_name)    


# Configure the ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=trans_rotation_range,
    width_shift_range=trans_width_shift_range,
    height_shift_range=trans_height_shift_range,
    zoom_range=trans_zoom_range,
    horizontal_flip=trans_horizontal_flip,
    fill_mode=trans_fill_mode,
)

# Function to apply data augmentation to a batch
def apply_data_augmentation(batch_data):
    batch_data_array = np.array(batch_data)
    augmented_batch_data = []
    for image in batch_data_array:
        augmented_image = train_datagen.random_transform(image)
        augmented_batch_data.append(augmented_image)
    return augmented_batch_data



train_data, train_labels = create_data_and_labels(df_train, label_name)
val_data, val_labels = create_data_and_labels(df_val, label_name)
test_data, test_labels = create_data_and_labels(df_test, label_name)


# Create DataLoaders for training, validation, and testing
train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=25, shuffle=True)
val_loader = DataLoader(list(zip(val_data, val_labels)), batch_size=25, shuffle=True)
test_loader = DataLoader(list(zip(test_data, test_labels)), batch_size=25, shuffle=False)




# Data augmentation function
def preprocess_function(image):
    # Adjust contrast of the image
    image = tf.image.adjust_contrast(image, 1.2)
    return image

train_datagen = ImageDataGenerator(
    rotation_range=cnn_rotation_range,
    width_shift_range=cnn_width_shift_range,
    height_shift_range=cnn_height_shift_range,
    zoom_range=cnn_zoom_range,
    horizontal_flip=cnn_horizontal_flip,
    fill_mode=cnn_fill_mode,
    preprocessing_function=preprocess_function  # Custom preprocessing function
)

# Normalize the data and convert to NumPy arrays
def normalize_images(data):
    # Check if data contains np.arrays, if yes, directly normalize them
    if isinstance(data[0], np.ndarray):
        return np.array(data, dtype=np.float32)
    else:
        return np.array([np.array(image) for image in data], dtype=np.float32) 

X_train = normalize_images(train_data)
X_val = normalize_images(val_data)
X_test = normalize_images(test_data)

# Preprocess input data for VGG16 (standardize based on ImageNet)
X_train = preprocess_input_resnet101(X_train)
X_val = preprocess_input_resnet101(X_val)
X_test_copy = copy.deepcopy(X_test)
X_test = preprocess_input_resnet101(X_test)

# Convert labels to NumPy arrays
y_train = np.array(train_labels) - 1
y_val = np.array(val_labels) - 1
y_test = np.array(test_labels) - 1

y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

@tf.keras.utils.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Instantiate the custom F1 metric
f1 = F1Score()

if Model_type == "CNN" and hp_tuning == True and skip_train == False:
    

    # Load the pre-trained VGG16 model
    # Include only convolutional base (no fully connected layers) and use the specified input shape
    input_shape = (target_size[0], target_size[1], 3)  # Specify the input shape


    # Define the hyperparameter grid
    param_grid = {
        'dense_layer': hp_dense_layer,
        'learning_rate': hp_lr, 
    }

    best_val_accuracy = 0
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        print(f"Training with params: {params}")

        
        vgg16 = ResNet101(weights='imagenet', include_top=False, input_shape=input_shape)

        # Freeze all layers in the pre-trained model
        for layer in vgg16.layers:
            layer.trainable = False

        # Add custom classification head
        x = Flatten()(vgg16.output)  # Flatten the feature map into a 1D vector
        x = Dense(params['dense_layer'], activation='relu', kernel_regularizer=l1(0.001))(x)  # Fully connected layer with L1 regularization
        x = Dropout(dropout)(x)  # Dropout to prevent overfitting
        predictions = Dense(num_classes, activation='softmax')(x) # Output layer for binary classification

        # Define the complete model
        model = Model(inputs=vgg16.input, outputs=predictions)

        # Create a training data generator with data augmentation
        train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss=cnn_loss,
            metrics=['accuracy', f1]  # Include F1 score as a metric
        )

        # Train the model
        history1 = model.fit(
            train_generator,  # Use the augmented data generator for training
            epochs=cnn_num_epochs_pre,  # Train for specified epochs
            validation_data=(X_val, y_val),  # Use the validation set for evaluation
        )


        for layer in model.layers:
            layer.trainable = True

        # Define a callback to save the best model based on validation accuracy
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cnn_log_directory, 'best_model.keras'),
            monitor='val_f1_score',
            save_best_only=True,
            mode='max',
            verbose=1
        )


        # Train the model
        history2 = model.fit(
            train_generator,  # Use the augmented data generator for training
            epochs=cnn_num_epochs,  # Train for specified epochs
            validation_data=(X_val, y_val),  # Use the validation set for evaluation
            callbacks=[checkpoint_callback]  # Include the checkpoint callback
        )

        metrics = {
            "train_loss1": history1.history['loss'],  # Training loss for each epoch
            "train_accuracy1": history1.history['accuracy'],  # Training accuracy for each epoch
            "val_accuracy1": history1.history['val_accuracy'],  # Validation accuracy for each epoch
            "train_loss2": history2.history['loss'],  # Training loss for each epoch
            "train_accuracy2": history2.history['accuracy'],  # Training accuracy for each epoch
            "val_accuracy2": history2.history['val_accuracy'],  # Validation accuracy for each epoch
        }

        metrics_path = os.path.join(cnn_log_directory, cnn_metrics_name)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")

        # Load the best model after training
        best_model_path = os.path.join(cnn_log_directory, 'best_model.keras')
        model = tf.keras.models.load_model(best_model_path)
        print(f"Best model loaded from {best_model_path}")

        # Evaluate the model on the validation set
        val_predictions = model.predict(X_val)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        val_true_classes = np.argmax(y_val, axis=1)
        val_accuracy = accuracy_score(val_true_classes, val_pred_classes)

        print(f"Validation accuracy: {val_accuracy:.4f}")

        # Update the best model if the current one is better
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = params
            best_model = model

    # Save the best model to a file
    best_model_save_path = os.path.join(cnn_log_directory, 'final_best_model.keras')
    best_model.save(best_model_save_path)
    print(f"Best model saved to {best_model_save_path}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best parameters: {best_params}")

    # Evaluate the best model on the test set
    test_predictions = best_model.predict(X_test)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    test_true_classes = np.argmax(y_test, axis=1)
    test_accuracy = accuracy_score(test_true_classes, test_pred_classes)


    print(f"Test accuracy: {test_accuracy:.4f}")

# if retrain:

#     best_model_save_path = os.path.join(cnn_log_directory, 'final_best_model.keras')
#     best_model = tf.keras.models.load_model(best_model_save_path)

#     train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# # Define a callback to save the best model based on validation accuracy
#     checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         filepath=os.path.join(cnn_log_directory, 'best_model.keras'),
#         monitor='val_accuracy',
#         save_best_only=True,
#         mode='max',
#         verbose=1
#     )

#     # Train the model
#     history2 = best_model.fit(
#         train_generator,  # Use the augmented data generator for training
#         epochs=5,  # Train for specified epochs
#         validation_data=(X_val, y_val),  # Use the validation set for evaluation
#         callbacks=[checkpoint_callback]  # Include the checkpoint callback
#     )

#     # Load the best model after training
#     best_model_path = os.path.join(cnn_log_directory, 'best_model.keras')
#     best_model = tf.keras.models.load_model(best_model_path)
#     print(f"Best model loaded from {best_model_path}")

#     # Evaluate the best model on the test set
#     test_predictions = best_model.predict(X_test)
#     test_pred_classes = np.argmax(test_predictions, axis=1)
#     test_true_classes = np.argmax(y_test, axis=1)
#     test_accuracy = accuracy_score(test_true_classes, test_pred_classes)


#     print(f"Test accuracy: {test_accuracy:.4f}")


# Plot the first 5 images of X_test
plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow((X_test_copy[i] * 255).astype(np.uint8))  # Convert back to uint8 for display
    plt.title(f"Label: {np.argmax(y_test[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()
# Save the best model to a file
best_model_save_path = os.path.join(cnn_log_directory, 'final_best_model.keras')
best_model = tf.keras.models.load_model(best_model_save_path)


# Evaluate the best model on the test set
test_predictions = best_model.predict(X_test)
test_pred_classes = np.argmax(test_predictions, axis=1)
test_true_classes = np.argmax(y_test, axis=1)
test_accuracy = accuracy_score(test_true_classes, test_pred_classes)


print(f"Test accuracy: {test_accuracy:.4f}")