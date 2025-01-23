# Image Segmentation Pipeline Using DeepLabV3

This repository contains an **Image Segmentation Application** built with **DeepLabV3**, leveraging PyTorch and Streamlit for a seamless and interactive experience. The application processes images to perform segmentation tasks, providing fine-tuned results with performance metrics such as Intersection over Union (IoU).

## Features

- **DeepLabV3 Model**: Utilizes a pre-trained `DeepLabV3-ResNet101` model for semantic segmentation.
- **Interactive Interface**: Built with Streamlit, enabling parameter tuning and visualization.
- **Customizable Segmentation**: Adjust kernel size and iterations for morphological operations.
- **IoU Score Calculation**: Evaluate segmentation accuracy against ground truth masks.
- **Dataset Support**: Supports datasets with original images and corresponding no-background masks.

## Requirements

Ensure you have Python 3.8 or higher installed. The required Python libraries can be installed using the provided `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-segmentation-pipeline.git
   cd image-segmentation-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run image_segmentation_app.py
   ```

4. Access the application in your browser at `http://localhost:8501`.

## Input Parameters

- **Number of Images**: Select the number of images to process and display.
- **Kernel Size**: Define the size of the kernel for morphological operations.
- **Iterations**: Specify the number of iterations for refining segmentation masks.

## Directory Structure

```
image-segmentation-pipeline/
├── public_hand_dataset/           # Dataset directory (default)
│   ├── [image_id]/
│       ├── original/              # Original images
│       └── no_bg/                 # Corresponding no-background masks
├── image_segmentation_app.py      # Main application file
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

## Dataset

The app expects the dataset to be structured as follows:

```
public_hand_dataset/
├── [image_id]/
│   ├── original/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   └── no_bg/
│       ├── image1.jpg
│       ├── image2.jpg
```

Place your dataset in the `public_hand_dataset/` directory or specify a custom path in the code.

## Key Functions

- `load_model()`: Loads the pre-trained DeepLabV3 model.
- `load_dataset(dataset_dir)`: Loads image paths from the dataset.
- `process_images(file_paths, kernel_size, iterations, model)`: Processes images and applies segmentation.
- `fn_compute_iou(predicted, ground_truth)`: Computes the IoU score for evaluation.

## Results Visualization

The app displays the following:

1. **Original Image**: Input image for segmentation.
2. **Initial Segmentation**: Initial segmentation mask with IoU score.
3. **Fine-Tuned Segmentation**: Refined segmentation mask with IoU score.
4. **Expected Segmentation**: Ground truth segmentation mask.

## Technologies Used

- **Python**: Primary language for implementation.
- **PyTorch**: For deep learning model loading and inference.
- **Streamlit**: For creating an interactive web app.
- **OpenCV**: For morphological operations and image processing.
- **Pillow**: For image manipulation and preprocessing.

## Contributions

Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request.


### Acknowledgments

- **DeepLabV3**: Pre-trained model from PyTorch's TorchVision library.
- **Streamlit**: Simplifies building data-driven apps.

