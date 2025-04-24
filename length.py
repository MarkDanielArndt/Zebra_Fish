import numpy as np
import cv2
import os
import tensorflow as tf
import torch
from tensorflow.keras.applications.vgg16 import preprocess_input

def normalize_images(data):
        # Check if data contains np.arrays, if yes, directly normalize them
        if isinstance(data[0], np.ndarray):
            return np.array(data, dtype=np.float32)
        else:
            return np.array([np.array(image) for image in data], dtype=np.float32) 
        
def get_fish_length(img, original_size = [256,256]):
    image_array = np.array(img)

    # Calculate the maximum and minimum values of the image array
    max_value = np.max(image_array)
    min_value = np.min(image_array)
    
    # Calculate the image width and height
    image_height, image_width = image_array.shape
    # Resize the image to 256x256
    image_array = cv2.resize(image_array, (256, 256), interpolation=cv2.INTER_LINEAR)
    # Find the leftmost and rightmost white parts of the image
    white_rows = np.where(np.any(image_array == max_value, axis=1))[0]
    topmost_white = white_rows[0]
    bottommost_white = white_rows[-1]
    white_columns = np.where(np.any(image_array == max_value, axis=0))[0]
    leftmost_white = white_columns[0]
    rightmost_white = white_columns[-1]

    max_distance = 0
    for random_y1 in white_rows[::2]:
        for random_x1 in white_columns[::2]:
            if image_array[random_y1, random_x1] > 0:
                for random_y2 in white_rows[::2]:
                    for random_x2 in white_columns[::2]:
                        if image_array[random_y2, random_x2] > 0:
                            distance = np.hypot(
                                (random_x2 - random_x1) * (original_size[0] / 256),
                                (random_y2 - random_y1) * (original_size[1] / 256)
                            )
                            if distance > max_distance:
                                max_distance = distance
                                x1, y1 = random_x1, random_y1
                                x2, y2 = random_x2, random_y2

    result_image = cv2.cvtColor(image_array.copy(), cv2.COLOR_GRAY2BGR)

    max_curve_length = np.inf
    found_valid_curve = False
    curve_type = "none"

    temp_curve_x = np.linspace(x1, x2, 100, dtype=int)
    temp_curve_y = np.linspace(y1, y2, 100, dtype=int)

    if image_array[temp_curve_y, temp_curve_x].all() > 0:
        curve_length = np.hypot(
            np.diff(temp_curve_x) * (original_size[0] / 256),
            np.diff(temp_curve_y) * (original_size[1] / 256)
        ).sum()

        if curve_length < max_curve_length:
            max_curve_length = curve_length
            curve_x = temp_curve_x
            curve_y = temp_curve_y
            mid_point1 = None
            mid_point2 = None
            found_valid_curve = True
            curve_type = "no_midpoint"

    if not found_valid_curve:
        mid_y1, mid_x1 = np.meshgrid(
            np.arange(topmost_white, bottommost_white, 4),
            np.arange(leftmost_white, rightmost_white, 4),
            indexing='ij'
        )
        mid_y2, mid_x2 = np.meshgrid(
            np.arange(topmost_white, bottommost_white, 4),
            np.arange(leftmost_white, rightmost_white, 4),
            indexing='ij'
        )

        mid_y1 = mid_y1.ravel()
        mid_x1 = mid_x1.ravel()
        mid_y2 = mid_y2.ravel()
        mid_x2 = mid_x2.ravel()


        for i in range(len(mid_y1)):
            if image_array[mid_y1[i], mid_x1[i]] > 0:
                for j in range(len(mid_y2)):
                    if image_array[mid_y2[j], mid_x2[j]] > 0:
                        temp_curve_x1 = np.linspace(x1, mid_x1[i], 50, dtype=int)
                        temp_curve_y1 = np.linspace(y1, mid_y1[i], 50, dtype=int)
                        temp_curve_x2 = np.linspace(mid_x1[i], mid_x2[j], 50, dtype=int)
                        temp_curve_y2 = np.linspace(mid_y1[i], mid_y2[j], 50, dtype=int)
                        temp_curve_x3 = np.linspace(mid_x2[j], x2, 50, dtype=int)
                        temp_curve_y3 = np.linspace(mid_y2[j], y2, 50, dtype=int)

                        if (
                            image_array[temp_curve_y1, temp_curve_x1].all() > 0 and
                            image_array[temp_curve_y2, temp_curve_x2].all() > 0 and
                            image_array[temp_curve_y3, temp_curve_x3].all() > 0
                        ):
                            curve1_distance = np.hypot(
                                np.diff(temp_curve_x1) * (original_size[0] / 256),
                                np.diff(temp_curve_y1) * (original_size[1] / 256)
                            ).sum()
                            curve2_distance = np.hypot(
                                np.diff(temp_curve_x2) * (original_size[0] / 256),
                                np.diff(temp_curve_y2) * (original_size[1] / 256)
                            ).sum()
                            curve3_distance = np.hypot(
                                np.diff(temp_curve_x3) * (original_size[0] / 256),
                                np.diff(temp_curve_y3) * (original_size[1] / 256)
                            ).sum()
                            curve_length = curve1_distance + curve2_distance + curve3_distance

                            if curve_length < max_curve_length:
                                max_curve_length = curve_length
                                curve_x1 = temp_curve_x1
                                curve_y1 = temp_curve_y1
                                curve_x2 = temp_curve_x2
                                curve_y2 = temp_curve_y2
                                curve_x3 = temp_curve_x3
                                curve_y3 = temp_curve_y3
                                mid_point1 = (mid_x1[i], mid_y1[i])
                                mid_point2 = (mid_x2[j], mid_y2[j])
                                found_valid_curve = True
                                curve_type = "two_midpoints"

    if curve_type == "no_midpoint":
        for i in range(len(curve_x) - 1):
            cv2.line(result_image, (curve_x[i], curve_y[i]), (curve_x[i + 1], curve_y[i + 1]), (255, 0, 0), thickness=1)

    elif curve_type == "two_midpoints":
        for i in range(len(curve_x1) - 1):
            cv2.line(result_image, (curve_x1[i], curve_y1[i]), (curve_x1[i + 1], curve_y1[i + 1]), (255, 0, 0), thickness=1)
        for i in range(len(curve_x2) - 1):
            cv2.line(result_image, (curve_x2[i], curve_y2[i]), (curve_x2[i + 1], curve_y2[i + 1]), (0, 255, 0), thickness=1)
        for i in range(len(curve_x3) - 1):
            cv2.line(result_image, (curve_x3[i], curve_y3[i]), (curve_x3[i + 1], curve_y3[i + 1]), (0, 0, 255), thickness=1)

    return result_image, max_curve_length

def apply_mask(original_image, mask):
    """
    Apply the mask to the original image.
    """
    # Convert the mask to a 3-channel image
    original_image = cv2.resize(original_image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(original_image, mask_3ch)

    return masked_image

def classification_curvature(image, mask, cnn_log_directory="Models/CNN", cnn_model_name="grownmask_1404.keras"):
    
    masked_image = apply_mask(image, mask)

    # Ensure the masked image is in RGB format
    masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    
    # Ensure the image is scaled to [0, 255] before preprocessing
    masked_image_rgb = np.clip(masked_image_rgb, 0, 255).astype(np.uint8)
    
    # Preprocess the image
    normalize_image = normalize_images([masked_image_rgb])

    processed_image = preprocess_input(normalize_image)

    best_model_path = os.path.join(cnn_log_directory, cnn_model_name)
    #best_model_path= f"{load_directory}/{load_model_name}"
    model = tf.keras.models.load_model(best_model_path)
    
    curvature = np.argmax(model.predict(processed_image))
    
    return masked_image, curvature