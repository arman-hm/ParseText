"""
Transform Module

Author: Arman Hajimirza
Date: January 24, 2025

This module contains the "Training_transforms" "Testing_transforms" and  functions, which is used
to to transform imput image for train and inference data.

Functions:
    The `Training_transforms` include a series of augmentations to 
    increase the variability of the training data:
        - `InvertImg`: Randomly inverts the colors of the image.
        - `Perspective`: Applies a random perspective transformation.
        - `ColorJitter`: Randomly changes the brightness, contrast, saturation, and hue.
        - `Rotate`: Randomly rotates the image within a given range.
        - `Sharpen`: Randomly sharpens the image.
        - `Resize`: Resizes the image to a fixed size.
        - `ToTensorV2`: Converts the image to a PyTorch tensor.
    The `Testing_transforms` include transformations for preprocessing
    the testing data:
        - `Resize`: Resizes the image to a fixed size.
        - `ToTensorV2`: Converts the image to a PyTorch tensor.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Augmentation pipeline
Training_transforms = A.Compose([

    A.InvertImg(p=0.5),
    A.Perspective(scale=0.1, keep_size=True, p=0.5 , fit_output=True),
    A.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.5, p=0.5),
    A.Rotate(limit=15, border_mode=0, p=0.5),
    A.Sharpen(alpha=(0.5, 1.0), lightness=(0.9, 1.1), p=0.5),
    A.Resize(height=32, width=100),
    ToTensorV2(),  # Convert to PyTorch tensor
])

Testing_transforms = A.Compose([
    A.Resize(height=32, width=100),
    ToTensorV2(),  # Convert to PyTorch tensor
])
