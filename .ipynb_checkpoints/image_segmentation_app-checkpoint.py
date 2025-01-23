# Image Segmentation Pipeline using DeepLabV3

## Import Required Libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
from matplotlib import pyplot as plt  # For plotting and visualization
from PIL import Image  # For image processing
import torch  # For PyTorch framework
from torchvision import models, transforms  # For pre-trained models and image transformations
import cv2  # For image processing tasks
import os  # For directory and file handling
from glob import glob  # For file path handling
from random import sample  # For random selection of images
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import streamlit as st  # For building the web app

## Define Functions

@st.cache_resource
def load_model():
    """
    Load the pre-trained DeepLabV3 model and cache it.

    Returns:
        model: Loaded DeepLabV3 model.
    """
    model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).eval()
    return model

@st.cache_data
def load_dataset(dataset_dir):
    """
    Load the dataset paths and cache the results.

    Args:
        dataset_dir (str): Root directory of the dataset.

    Returns:
        tuple: Lists of original image paths and no-background image paths.
    """
    original_folders = glob(os.path.join(dataset_dir, "*", "original"))
    no_bg_folders = glob(os.path.join(dataset_dir, "*", "no_bg"))

    original_images = [glob(os.path.join(original_folder, "*.*")) for original_folder in original_folders]
    no_bg_images = [glob(os.path.join(no_bg_folder, "*.*")) for no_bg_folder in no_bg_folders]

    return original_images, no_bg_images

@st.cache_data
def process_images(file_paths, kernel_size, iterations, _model):
    """
    Process all selected images and cache the results.

    Args:
        file_paths (list): List of image file paths (original and no-background).
        kernel_size (tuple): Size of the kernel for morphological operations.
        iterations (int): Number of iterations for morphological operations.
        model: Pre-trained DeepLabV3 model.

    Returns:
        list: List of processed image dictionaries.
    """
    all_images = [fn_apply_segmentation(files, kernel_size, iterations) for files in file_paths]
    return all_images

### Convert Input Image into a Tensor
def fn_convert_input_tensor(image):
    """
    Converts an input PIL image to a PyTorch tensor with necessary preprocessing.

    Args:
        image (PIL.Image): Input image.

    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension.
    """
    transform = transforms.Compose([
        transforms.Resize((1024), interpolation=Image.BICUBIC),  # Resize image
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Convert the image into a tensor and add a batch dimension
    input_tensor = transform(image).unsqueeze(0)

    return input_tensor

### Fine-Tune the Segmentation Mask
def fn_tune_morphological_metrics(image, segmentation_mask_resized, kernel_size, iterations):
    """
    Refine the segmentation mask using morphological operations and connected component analysis.

    Args:
        image (PIL.Image): The input image.
        segmentation_mask_resized (np.ndarray): The resized binary segmentation mask.
        kernel_size (tuple): Size of the kernel for morphological operations.
        iterations (int): Number of iterations for morphological operations.

    Returns:
        np.ndarray: Refined segmentation mask.
    """
    # Ensure the mask is binary
    refined_mask = segmentation_mask_resized

    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # Find contours
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if contours:
        # Get the largest contour
        hand_contour = max(contours, key=cv2.contourArea)

        # Create a blank mask and draw the largest contour
        refined_mask = np.zeros_like(refined_mask)
        cv2.drawContours(refined_mask, [hand_contour], -1, 255, thickness=cv2.FILLED)
    else:
        # Return an empty mask if no contours are found
        return np.zeros_like(refined_mask)

    # Extract the largest connected component
    num_labels, labels_im = cv2.connectedComponents(refined_mask, connectivity=4)
    if num_labels > 1:
        label_sizes = [(label, np.sum(labels_im == label)) for label in range(1, num_labels)]
        largest_label = max(label_sizes, key=lambda x: x[1])[0]
        refined_mask = (labels_im == largest_label).astype(np.uint8) * 255

    return refined_mask

### Compute Intersection over Union (IoU) Score
def fn_compute_iou(predicted, ground_truth):
    """
    Compute the Intersection over Union (IoU) score between the predicted and ground truth masks.

    Args:
        predicted (np.ndarray): Predicted binary mask.
        ground_truth (np.ndarray): Ground truth binary mask.

    Returns:
        float: IoU score.
    """
    intersection = np.logical_and(predicted, ground_truth).sum()
    union = np.logical_or(predicted, ground_truth).sum()
    iou_score = intersection / union if union != 0 else 0.0

    return iou_score

### Process Each Image and Apply Segmentation
def fn_apply_segmentation(files, kernel_size, iterations):
    """
    Process a pair of images (original and no-background) and apply segmentation.

    Args:
        files (tuple): A tuple containing paths to the original and no-background images.
        kernel_size (tuple): Size of the kernel for morphological operations.
        iterations (int): Number of iterations for morphological operations.

    Returns:
        dict: Dictionary containing the original image, segmented images, and IoU scores.
    """
    orig_image_path = files[0][0]  # Original image path
    no_bg_image_path = files[1][0]  # No-background image path

    # Load the original image
    orig_image = Image.open(orig_image_path).convert("RGB")

    # Convert image to tensor
    input_tensor = fn_convert_input_tensor(orig_image)

    # Perform segmentation and generate the initial segmentation mask
    with torch.no_grad():
        output = model(input_tensor)["out"][0]

    # Initial segmentation mask
    init_segmentation_mask = output.argmax(0).byte().cpu().numpy()

    # Resize the mask to match the original image size
    init_segmentation_mask_resized = cv2.resize(
        init_segmentation_mask.astype(np.uint8),
        (orig_image.width, orig_image.height),
        interpolation=cv2.INTER_NEAREST
    )

    # Fine-tune segmentation mask
    refined_mask = fn_tune_morphological_metrics(orig_image, init_segmentation_mask_resized, kernel_size, iterations)

    # Apply the refined mask to the original image
    init_segmented_image = cv2.bitwise_and(np.array(orig_image), np.array(orig_image), mask=init_segmentation_mask_resized)
    fine_tune_segmented_image = cv2.bitwise_and(np.array(orig_image), np.array(orig_image), mask=refined_mask)

    # Load the no-background image
    no_bg_image = np.array(Image.open(no_bg_image_path).convert("RGB"))

    # Compute IoU scores
    init_segmented_score = fn_compute_iou(init_segmented_image, no_bg_image)
    refined_segmented_score = fn_compute_iou(fine_tune_segmented_image, no_bg_image)

    # Output dictionary
    output = {
        'Original': np.array(orig_image),
        'Initial Segmentation': init_segmented_image,
        'Fine-Tuned Segmentation': fine_tune_segmented_image,
        'Expected Segmentation': no_bg_image,
        'Initial_Segmented_Image_IoU_Score': init_segmented_score,
        'Fine_Tuned_Segmented_Image_IoU_Score': refined_segmented_score
    }

    return output

### Streamlit App
st.title("Image Segmentation App")

# User inputs
num_images = st.slider("Select the number of images to display:", min_value=1, max_value=20, value=5)
kernel_size = st.slider("Kernel size for morphological operations:", min_value=1, max_value=20, value=5, step=2)
iterations = st.slider("Number of iterations for morphological operations:", min_value=1, max_value=20, value=12)

# Add submit and reset buttons
col1, col2 = st.columns(2)
with col1:
    submit = st.button("Run Segmentation")
with col2:
    reset = st.button("Reset")

# Define the root directory of the dataset
dataset_dir = "./public_hand_dataset/"

if submit:
    st.write("Loading dataset and initializing model...")

    # Load dataset and model
    original_images, no_bg_images = load_dataset(dataset_dir)
    model = load_model()

    # Zip the information into one iterator
    file_paths = list(zip(original_images, no_bg_images))

    # Randomly select the number of images specified by the user
    selected_file_paths = sample(file_paths, min(num_images, len(file_paths)))

    st.write("Processing images...")

    # Process all selected images
    all_images = process_images(selected_file_paths, (kernel_size, kernel_size), iterations, model)

    st.write("Displaying results...")

    # Display results
    for image_index, image_dict in enumerate(all_images):
        st.subheader(f"Image Results (Image {image_index + 1})")
        cols = st.columns(4)
        keys = ['Original', 'Initial Segmentation', 'Fine-Tuned Segmentation', 'Expected Segmentation']

        for col, key in zip(cols, keys):
            if key in image_dict:
                caption = f"{key}"
                if key == 'Initial Segmentation':
                    caption += f" (IoU: {image_dict['Initial_Segmented_Image_IoU_Score']:.4f})"
                elif key == 'Fine-Tuned Segmentation':
                    caption += f" (IoU: {image_dict['Fine_Tuned_Segmented_Image_IoU_Score']:.4f})"
                col.image(image_dict[key], caption=caption, use_column_width=True)

if reset:
    st.write("Resetting parameters...")
    st.rerun()
