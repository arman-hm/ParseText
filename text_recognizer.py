"""
Text Recognizer Module

Author: Arman Hajimirza
Date: January 21, 2025

This module contains the `TextRecognizer` class, which is used to recognize words in an image.
This is a word-level OCR designed for recognizing a single word in each image, not for line-level or
sentence-level recognition.

Classes:
    TextRecognizer: A class to check and handle device availability for image and model loading.
"""
import logging
import torch
import numpy as np
from Scripts import Transforms
from Models import CRNN
# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class TextRecognizer():
    """
    A class used to recognize text from images using
    a Convolutional Recurrent Neural Network (CRNN).
    Attributes:
        transform (callable): A function/transform to apply to the input images.
        char_to_index (dict): A dictionary mapping characters to their respective indices. 
        device (str): The device to run the model on ('cpu' or 'cuda').
        model (torch.nn.Module): The CRNN model for text recognition.

    Methods:
        load_model(path, num_classes):
            Loads the pre-trained model from the specified path.
        check_device(device):
            Checks if CUDA is available and sets the device accordingly.
        preprocess(images):
            Preprocesses a list of images for model input.
        convert_image(image):
            Converts an image to grayscale and normalizes it.
        inference(images):
            Performs inference on a list of images and returns recognized text.
        ctc_decode(predictions, blank=0):
            Decodes model predictions using CTC decoding.
        index_to_string(decoded):
            Converts decoded indices to corresponding strings using the character
            index.
    """

    def __init__(self, model_path: str, char_index=dict, transform=Transforms.Testing_transforms
                 , device="cpu"):
        self.transform = transform
        self.char_to_index = char_index
        self.device = self.check_device(device=device)
        self.load_model(path=model_path, num_classes=len(char_index)+1)

    def load_model(self, path, num_classes):
        """
        Loads the pre-trained model from the specified path.
        Args:
            path (str): path of model.
            num_classes: number of classes(characters) that model can predict.
        """
        try:
            checkpoint = torch.load(path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found at path: {path}") from e
        self.model = CRNN(num_classes=num_classes)
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(e)
            raise RuntimeError("Error loading the model state dictionary") from e
        self.model.to(self.device)

    def check_device(self, device):
        """
        Check if Cuda device is available to load images and model on gpu.
        Args:
            device (str): Device to check ('cpu' or 'cuda').
        return:
            str: 'cuda' if CUDA is available, else 'cpu'.
        Raises:
            ValueError: If an invalid device string is provided.
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError("Invalid device specified. Choose either 'cpu' or 'cuda'.")
        if device == "cuda" and torch.cuda.is_available():
            logging.info("CUDA is available")
            logging.info("Device count: %s",{torch.cuda.device_count()})
            logging.info("Current device: %s", {torch.cuda.current_device()})
            logging.info("Device name: %s"
                         ,{torch.cuda.get_device_name(torch.cuda.current_device())})
            return "cuda"
        logging.info("CUDA is not available. Using CPU instead.")
        return "cpu"

    def preprocess(self, images: list):
        """
        Preprocess list of PIl images, convert them to Tensor and returns batched Tensor
        for further processing. 
        Args:
            images (list): list of input PIL images.
        return:
            Tensor
        """
        tensors_list = []
        for image in images:
            image = self.convert_image(image)
            image = self.transform(image=image)["image"]
            tensors_list.append(image)
        return torch.stack(tensors_list)

    def convert_image(self, image):
        """
        converts single PIL image to grayscale and numpy array. 
        Args:
            image (PIL image): a single image.
        return:
            Numpy array: image 
        """
        image = np.array(image.convert('L'))
        image = (image/255).astype('float32')
        return np.expand_dims(image, axis=2)

    def inference(self, images):
        """ Perform inference on a list of images to recognize and return text.
            Args:
                images (list): A list of images to be processed and recognized.
            Returns:
                list: A list of recognized text strings for each image.
            This method:
                1. Preprocesses the input images.
                2. Transfers the batched images to the specified device ('cpu' or 'cuda').
                3. Passes the batched images through the CRNN model to get model output.
                4. Permutes the dimensions of the model output.
                5. Extracts the predicted character indices.
                6. Decodes the predictions using CTC decoding.
                7. Converts the decoded indices to strings using the character index dictionary.
        """
        batched = self.preprocess(images=images)
        batched = batched.to(self.device)
        model_output = self.model(batched)
        model_output = model_output.permute(1, 0, 2)
        predictions = model_output.argmax(2).transpose(0, 1).cpu().numpy()
        decoded = self.ctc_decode(predictions, blank=0)
        return self.index_to_string(decoded=decoded)

    def ctc_decode(self, predictions, blank=0):
        """
        Decode predictions using CTC decoding.

        Args:
            predictions (Tensor): Model predictions.
            blank (int): Blank token index.

        Returns:
            list: Decoded text predictions.
        """
        pred_texts = []
        for pred in predictions:
            pred_chars = []
            previous_char = None
            for char in pred:
                if char != previous_char and char != blank:
                    pred_chars.append(char)
                previous_char = char
            pred_texts.append(pred_chars)
        return pred_texts

    def index_to_string(self, decoded):
        """
        Convert decoded character indices into strings using the character index dictionary.
        
        Args:
            decoded (list of lists): A list of lists where each sub-list contains
            character indices.
        
        Returns: 
            list: A list of strings representing the decoded text for each sequence of
            character indices.
        """
        word_list = []
        for word in decoded:
            word_list.append("".join([list(self.char_to_index.keys())[list(
                self.char_to_index.values()).index(index)] for index in word]))
        return word_list
