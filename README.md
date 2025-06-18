# Alligator and Reticulated Python Classifier

An image classification project based on neural networks using TensorFlow and MobileNetV2 to distinguish between alligators and reticulated pythons. Includes image preprocessing, data augmentation, and training with validation.

## Description

This project uses transfer learning with a pretrained MobileNetV2 to extract features from images resized to 224x224 pixels. Data augmentation techniques such as rotation, shifting, zooming, and shearing are applied to improve model generalization. The model classifies images into two classes: alligators and reticulated pythons.

## Structure

- `dataset/alligators` - Images of alligators  
- `dataset/reticulated_python` - Images of reticulated pythons  
- Python code for preprocessing, training, and prediction.

## Usage

1. Prepare the datasets with images organized into folders.  
2. Run the script to train the model.  
3. Use the `classify_image(path)` function to predict the class of an image.

## Requirements

- Python 3.x  
- TensorFlow  
- TensorFlow Hub  
- PIL (Pillow)  
- Matplotlib  
- Numpy  




