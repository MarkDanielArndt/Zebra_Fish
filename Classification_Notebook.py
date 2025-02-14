#Imports

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
import timm
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
seed = 111


#Set Parameters
#1. Set parameters for data loading
excel_path = "Annotation_Zebrafish_full.xlsx"
images_folder = "C:/Users/ma405l/Documents/Heidelberg_Schweregrad/Full_data/Raw_data_full_train"
masks_folder = "C:/Users/ma405l/Documents/Heidelberg_Schweregrad/Full_data/Masked_images"

#2. Set parameters for data processing
target_size=(256,256)	#Size of the images for training
num_classes = 4

#3. Train, Val, Test Split
train_size = 0.6
val_size = 0.2
test_size = 0.2
label_name = "Curved"

#4. Balance datasets
balance_train = True
balance_val = False
balance_test = False

Model_type = "CNN"

# HP Tuning
hp_tuning = True
hp_dense_layer = [512, 1024]
hp_lr = [2e-4]

#5 Transformer. Augmentation parameters
trans_rotation_range = 45
trans_width_shift_range = 0.2
trans_height_shift_range = 0.2
trans_zoom_range = 0.1
trans_horizontal_flip  = False
trans_fill_mode="nearest"

#6 Transformer. Training parameters
trans_learning_rate = 0.00008 # Define learing rate
trans_num_epochs = 1 # Define the number of epochs

#7 Transformer. Save parameters
trans_log_directory = f"Models/Transformer"
trans_model_name = "trans_model.pth"
trans_metrics_name = "trans_metrics.txt"

#5 CNN. Augmentation parameters
cnn_rotation_range = 45
cnn_width_shift_range = 0.2
cnn_height_shift_range = 0.2
cnn_zoom_range = 0.1
cnn_horizontal_flip  = False
cnn_fill_mode="nearest"

#6 CNN. Training parameters
cnn_learning_rate = 0.001
cnn_loss = 'crossentropy'
cnn_num_epochs = 10
cnn_num_epochs_pre = 10
dense_layer = 512
dropout = 0.3

#7 CNN. Save parameters
cnn_log_directory = "Models/CNN"
cnn_model_name = "vgg_16_model.keras"
cnn_metrics_name = "vgg_16_metrics.json"

#1) Match the images and masks to the exel data. 
# Create Df with the following columns: Image, Mask Path, Sample, Fish_Num, Edema, Curved, Masked Image

def process_fish_data(excel_path, images_folder, masks_folder):
    # Read the Excel file
    df = pd.read_excel(excel_path, dtype={'Sample': str, 'Fish_Num': int, 'Edema': str, 'Curved': str})

    # Convert Fish_Num to two-digit format (01, 02, ...)
    df['Fish_Num'] = df['Fish_Num'].apply(lambda x: f"{x:02d}")

    # Store results
    results = []

    for _, row in df.iterrows():
        sample = row['Sample']
        fish_num = row['Fish_Num']
        edema = row['Edema']
        curved = row['Curved']

        # Find the image
        image_pattern = os.path.join(images_folder, f"*pr_{sample}-{fish_num}*.jpg")
        image_files = glob.glob(image_pattern)

        # Find the mask
        mask_pattern = os.path.join(masks_folder, f"*pr_{sample}-{fish_num}*_mask.jpg")
        mask_files = glob.glob(mask_pattern)

        # Ensure exactly one match
        if len(image_files) != 1 or len(mask_files) != 1:
            print(f"Skipping Sample {sample}, Fish {fish_num}: Image or mask missing/multiple found.")
            continue

        image_path = image_files[0]
        mask_path = mask_files[0]

        # Load image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Skipping {image_path} or {mask_path}: Unable to read file.")
            continue

        # Apply the mask: Everything outside the mask becomes black
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Store in results list
        results.append([image_path, mask_path, sample, fish_num, edema, curved, masked_image])

    # Convert to DataFrame
    columns = ['Images', 'Masks', 'Sample', 'Fish_Num', 'Edema', 'Curved', 'Masked Images']
    result_df = pd.DataFrame(results, columns=columns)

    return result_df

df_result = process_fish_data(excel_path, images_folder, masks_folder)


#2) Preprocess masked images.
def preprocess_masked_images(df, target_size):
    processed_images = []

    for i, row in df.iterrows():
        masked_image = row['Masked Images']

        # Step 1: Remove black-only rows and columns
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        coords = cv2.findNonZero(gray)  # Get non-black pixel coordinates

        if coords is None:  # In case the image is fully black (shouldn't happen)
            print(f"Skipping image {i}, it's fully black!")
            #TODO: funktioniert nicht, wenn es schwarze Bilder gibt, da es dann einen Fehler gibt, da die erwartete Größe nicht mit der echten Größe übereinstimmt
            continue
        
        x, y, w, h = cv2.boundingRect(coords)  # Get bounding box
        cropped_image = masked_image[y:y+h, x:x+w]  # Crop to bounding box

        # Step 2: Pad to square size (symmetrically)
        height, width = cropped_image.shape[:2]
        max_dim = max(height, width)

        # Calculate symmetric padding
        pad_top = (max_dim - height) // 2
        pad_bottom = max_dim - height - pad_top
        pad_left = (max_dim - width) // 2
        pad_right = max_dim - width - pad_left

        padded_image = cv2.copyMakeBorder(
            cropped_image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]  # Black padding
        )

        # Step 3: Resize to target size
        resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)

        # Save processed image
        processed_images.append(resized_image)

    # Update the DataFrame
    df['Processed Masked Images'] = processed_images
    return df

df_result = preprocess_masked_images(df_result, target_size)

#Free up memory 
#Delete unnecessary columns of images and masks
df_result = df_result.drop(columns=["Images", "Masks", "Masked Images"])
#Delete Rows with NAW in the label column
df_result = df_result[df_result[label_name] != "NAW"]
# Convert label to integers
df_result["Curved"] = df_result["Curved"].astype(int) 


#3) Train, Val, Test Split saved in 3 new DataFrames

def split_data(df, train_size=70, val_size=20, test_size=10, label_name="Curved"):
    # Normalize percentages if they don’t sum to 100%
    total = train_size + val_size + test_size
    train_size, val_size, test_size = train_size / total, val_size / total, test_size / total

    # Shuffle data before splitting
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train and temp (val + test), stratified by label_name
    df_train, df_temp = train_test_split(df, test_size=(1 - train_size), random_state=42, stratify=df[label_name])

    # Split temp into val and test, also stratified
    df_val, df_test = train_test_split(df_temp, test_size=(test_size / (test_size + val_size)), 
                                       random_state=42, stratify=df_temp[label_name])

    return df_train, df_val, df_test

# Example usage
df_train, df_val, df_test = split_data(df_result, train_size, val_size, test_size, label_name)


#4) Balance the dataframes

#Augmentation functions
def apply_augmentation(image, augmentation_type):
    if augmentation_type == 'horizontal_flip':
        return cv2.flip(image, 1)
    elif augmentation_type == 'vertical_flip':
        return cv2.flip(image, 0)
    elif augmentation_type == 'both_flip':
        return cv2.flip(image, -1)
    elif augmentation_type == 'zoom_out':
        height, width = image.shape[:2]
        zoom_factor = 0.8  # 20% zoom out
        
        # Resize the image to 80% of its original size (zoom-out)
        resized_image = cv2.resize(image, (int(width * zoom_factor), int(height * zoom_factor)))
        
        # Create a black image of the original size
        padded_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate the padding size (difference between original and resized dimensions)
        top = (height - resized_image.shape[0]) // 2
        bottom = height - resized_image.shape[0] - top
        left = (width - resized_image.shape[1]) // 2
        right = width - resized_image.shape[1] - left
        
        # Place the resized image into the black canvas with padding
        padded_image[top:top+resized_image.shape[0], left:left+resized_image.shape[1]] = resized_image
        
        return padded_image
    return image

#Balance the dataframes by counting the most frequent class and augmenting the other classes
def balance_dataframe(df_frame, label_name, augmentations=['horizontal_flip', 'vertical_flip', 'both_flip', 'zoom_out']):
    # 1. Find the most represented class (most frequent class)
    most_frequent_class = df_frame[label_name].mode()[0]
    largest_class_size = (df_frame[label_name] == most_frequent_class).sum()

    # 2. Create a list to store augmented data
    augmented_rows = []

    # 3. Iterate over all the other classes
    for current_class in df_frame[label_name].unique():
        if current_class == most_frequent_class:
            continue  # Skip the most frequent class

        # Calculate the imbalance factor
        current_class_size = (df_frame[label_name] == current_class).sum()
        imbalance_factor = largest_class_size / current_class_size

        # Find the closest augmentation set to apply
        num_augmentations = min(5, round(imbalance_factor)) - 1  # Max 4 augmentations per image

        # Get the rows for the current class
        class_df = df_frame[df_frame[label_name] == current_class]

        # 4. Apply augmentations to each image in the current class
        for _, row in class_df.iterrows():
            original_image = row['Processed Masked Images']  # Image path or image itself

            for i in range(num_augmentations):
                augmented_image = apply_augmentation(original_image, augmentations[i])

                # Copy the row and replace the augmented image
                augmented_row = row.copy()
                augmented_row['Processed Masked Images'] = augmented_image
                augmented_rows.append(augmented_row)

    # 5. Combine the original and augmented data
    augmented_df = pd.DataFrame(augmented_rows)
    balanced_df = pd.concat([df_frame, augmented_df], ignore_index=True)

    return balanced_df

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

# Feature extractor function (you might want to adjust this based on the specific model you are using)
from transformers import AutoFeatureExtractor
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/deit-small-distilled-patch16-224")

#Create data and Labels from the dataframes
def create_data_and_labels(df_frame, label_name):
    data = []
    labels = []

    for _, row in df_frame.iterrows():
        image = row['Processed Masked Images']
        label = row[label_name]

        data.append(image)
        labels.append(int(label))  # Convert label to integer

    return data, labels

train_data, train_labels = create_data_and_labels(df_train, label_name)
val_data, val_labels = create_data_and_labels(df_val, label_name)
test_data, test_labels = create_data_and_labels(df_test, label_name)


# Create DataLoaders for training, validation, and testing
train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=25, shuffle=True)
val_loader = DataLoader(list(zip(val_data, val_labels)), batch_size=25, shuffle=True)
test_loader = DataLoader(list(zip(test_data, test_labels)), batch_size=25, shuffle=False)

# Load the pre-trained DeiT model with 224x224 resolution
trans_model = timm.create_model('deit_small_distilled_patch16_224', pretrained=True)

# Define the optimizer and loss function
#optimizer = Adam(trans_model.parameters(), lr=trans_learning_rate)
criterion = CrossEntropyLoss()

# Lists to track metrics
training_loss_list = []
training_acc_list = []
val_acc_list = []

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

if Model_type == "CNN" and hp_tuning == True:
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
    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)
    X_test = preprocess_input(X_test)

    # Convert labels to NumPy arrays
    y_train = np.array(train_labels) - 1
    y_val = np.array(val_labels) - 1
    y_test = np.array(test_labels) - 1

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # Load the pre-trained VGG16 model
    # Include only convolutional base (no fully connected layers) and use the specified input shape
    input_shape = (target_size[0], target_size[1], 3)  # Specify the input shape


    # Define the hyperparameter grid
    param_grid = {
        'dense_layer': hp_dense_layer,
        'learning_rate': hp_lr
    }

    best_val_accuracy = 0
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        print(f"Training with params: {params}")

        
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

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

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),  # Optimizer with specified learning rate
            loss=cnn_loss,  # Binary cross-entropy loss for binary classification
            metrics=['accuracy']  # Track accuracy during training
        )

        # Train the model
        history = model.fit(
            train_generator,  # Use the augmented data generator for training
            epochs=cnn_num_epochs_pre,  # Train for specified epochs
            validation_data=(X_val, y_val),  # Use the validation set for evaluation
        )


        for layer in model.layers:
            layer.trainable = True

        # Define a callback to save the best model based on validation accuracy
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cnn_log_directory, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        # Train the model
        history = model.fit(
            train_generator,  # Use the augmented data generator for training
            epochs=cnn_num_epochs,  # Train for specified epochs
            validation_data=(X_val, y_val),  # Use the validation set for evaluation
            callbacks=[checkpoint_callback]  # Include the checkpoint callback
        )

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

    # Write the best validation accuracy and parameters to a file
    with open(os.path.join(cnn_log_directory, 'best_params.txt'), 'w') as f:
        f.write(f"Best validation accuracy: {best_val_accuracy:.4f}\n")
        f.write(f"Best parameters: {best_params}\n")

    # Evaluate the best model on the test set
    test_predictions = best_model.predict(X_test)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    test_true_classes = np.argmax(y_test, axis=1)
    test_accuracy = accuracy_score(test_true_classes, test_pred_classes)

    # Write the test accuracy to the file
    with open(os.path.join(cnn_log_directory, 'best_params.txt'), 'a') as f:
        f.write(f"Test accuracy: {test_accuracy:.4f}\n")


    

