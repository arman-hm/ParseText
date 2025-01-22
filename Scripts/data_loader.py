"""
hjhfjkhvj
"""
import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset


class OCRDataset(Dataset):
    """
    class for load image from dict and store them to Memory to speedup loading process.
    
    Args:
        image_dir (str): Directory containing the images.
        labels (dict): A dictionary mapping image filenames to their labels.
        char_to_index (dict): A mapping from characters to indices.
        input_length (int): The maximum input length for sequences.
        transform (callable, optional): Transformations to apply to the images. Defaults to None.
        preload (bool, optional): Whether to preload images into memory. Defaults to True.

    """

    def __init__(self, image_dir, labels, char_to_index, input_length, transform=None
                 , preload=True):

        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.image_files = list(self.labels.keys())
        self.char_to_index = char_to_index
        self.input_length = input_length
        self.preload = preload

        # Precompute label targets
        try:
            self.precomputed_targets = {
                img: torch.tensor(
                    [self.char_to_index[char] for char in label], dtype=torch.long
                )
                for img, label in self.labels.items()
            }
        except KeyError as e:
            raise ValueError(f"Character {e} in labels is not in char_to_index mapping.")
        # Preload images if requested
        self.images = {}
        if self.preload:
            for img_name in tqdm(self.image_files, desc="Preloading Images"):
                self.images[img_name] = self.load_image(img_name)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        # Load preloaded image or read it
        if self.preload:
            image = self.images[img_name]
        else:
            image = self.load_image(img_name)

        if self.transform:
            image = self.transform(image=image)["image"]

        target_indices = self.precomputed_targets[img_name]
        target_length = len(target_indices)

        return image, target_indices, self.input_length, target_length

    def load_image(self, img_name):
        """
        load single PIL image and convert it to grayscale and numpy array. 
        Args:
            img_name (str): path of image.
        return:
            Numpy array: image 
        """
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert('L')
        img = np.array(img)
        img = (img/255).astype('float32')
        return np.expand_dims(img, axis=2)
