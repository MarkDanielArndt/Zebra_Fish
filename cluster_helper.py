import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from segmentation_models_pytorch import Unet


def load_and_recreate_masked_images(excel_path):
    # Load the Excel file
    df = pd.read_excel(excel_path, dtype={'Sample': str, 'Fish_Num': str, 'Edema': str, 'Curved': str})
    
    # Ensure Fish_Num is two-digit format
    df['Fish_Num'] = df['Fish_Num'].apply(lambda x: f"{int(x):02d}")

    # Placeholder for the masked images
    masked_images = []

    for _, row in df.iterrows():
        image_path = row['Images']
        mask_path = row['Masks']

        # Load image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Skipping {image_path} or {mask_path}: Unable to read file.")
            masked_images.append(None)
            continue

        # Apply the mask: Everything outside the mask becomes black
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        masked_images.append(masked_image)

    # Add the "Masked Images" column back to the DataFrame
    df['Masked Images'] = masked_images

    return df

#2) Preprocess masked images.
def preprocess_masked_images(df, target_size):
    processed_images = []
    rows_to_delete = []  # Store indices of black images

    for i, row in df.copy().iterrows():  # Iterate over a copy to prevent index shifting
        masked_image = row['Masked Images']

        # Step 1: Convert to grayscale and find non-black pixels
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)

        if coords is None:  # Image is fully black
            print(f"Deleting row {i} from DataFrame because image was fully black.")
            rows_to_delete.append(row.name)  # Use row.name (original index) instead of i
            continue

        # Step 2: Crop to bounding box
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = masked_image[y:y+h, x:x+w]

        # Step 3: Pad to square size
        height, width = cropped_image.shape[:2]
        max_dim = max(height, width)
        pad_top = (max_dim - height) // 2
        pad_bottom = max_dim - height - pad_top
        pad_left = (max_dim - width) // 2
        pad_right = max_dim - width - pad_left

        padded_image = cv2.copyMakeBorder(
            cropped_image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # Step 4: Resize
        resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)
        processed_images.append(resized_image)

    # **Drop rows after looping to avoid index shifting**
    df = df.drop(index=rows_to_delete).reset_index(drop=True)
    
    # Update DataFrame
    df['Processed Masked Images'] = processed_images
    return df

#define segment_fish function
def segment_fish(image, model):
    """
    Segment fish from the image using a pre-trained Unet model.
    
    Parameters:
        image (PIL.Image): The input image.
        
    Returns:
        PIL.Image: The segmented image with fish highlighted.
    """
    # Transform the image
    #image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image
    
    with torch.no_grad():
        # Get predictions from the model
        prediction = model(image_tensor)
    
    # Get the mask
    mask = prediction.squeeze().cpu().numpy()
    # Convert the mask to a confidence map
    confidence_map = (mask - mask.min()) / (mask.max() - mask.min()) * 255
    confidence_map = confidence_map.astype(np.uint8)
    
    # Convert the confidence map to a binary mask
    binary_mask = (confidence_map > 81).astype(np.uint8) * 255
    
    # Convert the binary mask to a PIL image
    segmented_image = Image.fromarray(binary_mask)
    # Find the largest connected component in the binary mask
    num_labels, labels_im = cv2.connectedComponents(binary_mask)

    # Find the largest component
    largest_component = 1 + np.argmax(np.bincount(labels_im.flat)[1:])

    # Create a mask for the largest component
    largest_component_mask = (labels_im == largest_component).astype(np.uint8) * 255

    # Convert the largest component mask to a PIL image
    segmented_image = Image.fromarray(largest_component_mask)
    
    return segmented_image, confidence_map

def apply_segmentation_model(df_result, seg_directory, target_size):
    """
    Apply a pre-trained segmentation model to the images in the DataFrame.

    Args:
    - df_result: DataFrame containing image paths and other metadata.
    - seg_directory: Directory where the segmentation model is stored.
    - target_size: Tuple specifying the target size for resizing images.

    Returns:
    - Updated DataFrame with "Confidence Maps" and "Segmented Masks" columns.
    """
    # Load model state
    loaded_model = Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    model_path = f"{seg_directory}/segmentation_model.pth"
    if os.path.exists(model_path):
        loaded_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        loaded_model.eval()  # Set model to evaluation mode
        print(f"Model loaded from {model_path}")
        
    # Define preprocessing parameters
    mean = np.array([0.485, 0.456, 0.406])  # Normalization mean
    std = np.array([0.229, 0.224, 0.225])  # Normalization std

    # Initialize lists to store the confidence maps and segmented masks
    confidence_maps = []
    segmented_masks = []

    for idx, row in df_result.iterrows():
        img_path = row["Images"]

        # Load image
        original_image = cv2.imread(img_path)  # Load raw image

        # Resize image
        original_image = cv2.resize(original_image, target_size, interpolation=cv2.INTER_LINEAR)

        # Normalize image
        processed_image = (original_image / 255.0 - mean) / std

        # Convert to PyTorch tensor (C, H, W) format
        input_image = torch.tensor(processed_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        # Get segmentation result from segment_fish function
        segmented_mask, confidence_map = segment_fish(input_image, loaded_model)

        # Convert the PIL image to a NumPy array
        segmented_mask_array = np.array(segmented_mask)
        
        # Append the confidence map and segmented mask to the lists
        confidence_maps.append(confidence_map)
        segmented_masks.append(segmented_mask_array)

    # Add the confidence maps and masks as new columns in the DataFrame
    df_result["Confidence Maps"] = confidence_maps    
    df_result["Segmented Masks"] = segmented_masks

    return df_result

#Floodfill the holes in segmented masks
def fill_holes(mask):
    """
    Fill all holes in a binary mask (0-1) by flood filling from the background.
    Tries top-left corner first, falls back to bottom-right if needed.
    """
    mask = (mask > 0).astype(np.uint8)  # Ensure the mask is uint8 (0-1 range)
    h, w = mask.shape

    flood_filled = mask.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)

    # Determine a background seed point (top-left or bottom-right)
    if mask[0, 0] == 0:
        seed = (0, 0)
    elif mask[h - 1, w - 1] == 0:
        seed = (w - 1, h - 1)
    else:
        return mask  # Return the original mask if no safe corner found

    # Perform flood fill
    cv2.floodFill(flood_filled, mask_ff, seedPoint=seed, newVal=1)

    # Invert the flood fill result (this now represents the "holes")
    flood_filled_inv = 1-flood_filled
        
    # Combine the original mask with the holes to a new mask: result (0-1 range)
    filled_mask = mask | flood_filled_inv

    # Scale to 0-255 before returning
    return (filled_mask * 255).astype(np.uint8)

#Grow mask with morphology filter
def grow_mask(mask, iterations=1, kernel_size=3):
    """
    Dilate the mask to grow the region by a number of iterations.

    Args:
    - mask: binary mask (numpy array with 0 and 1 or 0 and 255)
    - iterations: how many times to apply dilation
    - kernel_size: size of the structuring element

    Returns:
    - grown mask (same shape)
    """
    # Ensure binary with values 0 and 1
    mask = (mask > 0).astype(np.uint8)

    # Create a structuring element (you can also try cv2.MORPH_RECT or MORPH_CROSS)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply dilation
    dilated = cv2.dilate(mask, kernel, iterations=iterations)

    return dilated*255

def apply_masks_and_preprocess(df_result, target_size):
    masked_images = []

    for idx, row in df_result.iterrows():
        img_path = row["Images"]
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

        # Convert mask to
    mask_3ch = np.stack([mask] * 3, axis=-1)

    # Apply the mask to the image
    masked_image = original_image * mask_3ch

    # Append result
    masked_images.append(masked_image)

    # Overwrite the "Masked Images" column
    df_result["Masked Images"] = masked_images

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