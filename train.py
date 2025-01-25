"""
Train Module

Author: Arman Hajimirza
Date: January 25, 2025

This module handles the data loading, transformation, and training process for 
an OCR (Optical Character Recognition) model using CRAFT Text Detection. 

It provides the following functionalities:
- Argument Parsing: Parses command-line arguments for various configurations like paths, 
  batch size, learning rate, etc.
- Data Loading: Loads training and testing data from CSV files.
- Data Transformation: Applies data augmentation and transformation techniques using Albumentations.
- Model Definition: Defines the CRNN (Convolutional Recurrent Neural Network) model.
- Training: Trains the CRNN model using a custom training loop.
- Error Handling: Implements error handling for loading data and JSON files.

Modules and Libraries:
- argparse: For parsing command-line arguments.
- json: For handling JSON file operations.
- pandas: For data manipulation and analysis.
- torch: For PyTorch functionalities including neural networks and optimizers.
- torch.nn.utils.rnn: For handling variable-length sequences.
- torch.utils.data: For data loading utilities.
- Scripts.data_loader: Custom module for loading OCR dataset.
- Scripts.transforms: Custom module for data transformations.
- Models.CRNN: Custom CRNN model definition.
- Scripts.training_loop: Custom training loop for model training.

Arguments:
- --traindf: Path to the training dataset CSV file.
- --testdf: Path to the testing dataset CSV file.
- --label2index: Path to the labels index JSON file.
- --main_path: Path to the image folder.
- --batchsize: Batch size for training the model.
- --shuffle: Whether to shuffle the data.
- --input_length: Maximum size of input length of text.
- --num_workers: Number of workers for loading data.
- --lr: Learning rate for training.
- --device: Device to load the model on (e.g., 'cuda' or 'cpu').
- --checkpointpath: Path for saving model checkpoints.
- --logpath: Path for saving logs.
- --plotpath: Path for saving plots.
- --metrics: List of metrics for checkpoint.
- --num_epochs: Number of epochs for training.

Example Usage:
    python script_name.py --traindf Data/train_data.csv --testdf Data/test_data.csv
    --label2index char_to_index.json
"""
import argparse
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from Scripts import data_loader, transforms
from Models import CRNN
import Scripts.training_loop as training_loop


parser = argparse.ArgumentParser(description='Train CRNN')
parser.add_argument('--traindf', default='Data/train_data.csv', type=str,
                     help='Path to train dataset')
parser.add_argument('--testdf', default='Data/test_data.csv', type=str,
                     help='Path to test dataset')
parser.add_argument('--label2index', default='char_to_index.json', type=str,
                     help='Path to labels index json file')
parser.add_argument('--main_path', default="Data/persian_unique_ocr_dataset", type=str,
                     help='Path to image folder')
parser.add_argument('--batchsize', default=1000, type=int,
                     help='Batchsize for training model')
parser.add_argument('--shuffle', default=True, type=bool,
                     help='Shuffle data default is True')
parser.add_argument('--input_length', default=24, type=int,
                     help='max size of input length of text')
parser.add_argument('--num_workers', default=4, type=int,
                     help='Number of workers for loading data')
parser.add_argument('--lr', default=0.001, type=float,
                     help='learning rate for training')
parser.add_argument('--device', default="cuda", type=str,
                     help='type of device for load model on that device')
parser.add_argument('--checkpointpath', default="Models/checkpoints/v1.00", type=str,
                     help='path for saving checkpoints')
parser.add_argument('--logpath', default="Logs", type=str,
                     help='path for saving logs of model')
parser.add_argument('--plotpath', default="Plots", type=str,
                     help='path for saving plots of model')
parser.add_argument('--metrics', default=["Train_Loss", "Val_Loss", "Train_CER","Val_CER"
                               , "Train_character_accuracy", "Train_word_accuracy"
                               , "Val_character_accuracy", "Val_word_accuracy"], type=list,
                     help='list of metrics for checkpoint')
parser.add_argument('--num_epochs', default=150, type=int,
                     help='Number of epochs')

args = parser.parse_args()

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences.

    Args:
        batch: A list of data samples, where each sample is a tuple
                of (image, label, length).

    Returns:
        A tuple containing:
            images: A batched tensor of images.
            labels: A padded tensor of labels.
            target_lengths: A tensor of label lengths.
    """
    images, labels, input_lengths, tagret_lengths = zip(*batch)

    # Pad the labels to the maximum length in the batch
    labels = pad_sequence([label.clone().detach() for label in labels],
                    batch_first=True, padding_value=0)
    # Stack the images and lengths
    images = torch.stack(images, 0)
    input_lengths = torch.tensor(input_lengths)
    tagret_lengths = torch.tensor(tagret_lengths)

    return images, labels, input_lengths, tagret_lengths

if __name__ == '__main__':

    #load train dataframe
    try:
        df = pd.read_csv(args.traindf)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        print("Training dataframe loaded successfully.")
    except FileNotFoundError as e:
        print("Error loading training dataframe: %s", e)

    #load test dataframe
    try:
        tdf = pd.read_csv(args.testdf)
        tdf = tdf.loc[:, ~tdf.columns.str.contains('^Unnamed')]
        print("Testing dataframe loaded successfully.")
    except FileNotFoundError as e:
        print("Error loading testing dataframe: %s", e)
    # load json file
    try:
        with open(args.label2index, 'r',  encoding="utf-8") as file:
            data = json.load(file)
            char_list = data["char_to_index_list"]
            char2index = data["char_to_index"]
        print("label2index json file loaded successfully.")
    except FileNotFoundError:
        print("Error: The file 'char_to_index.json' was not found.")
    except json.JSONDecodeError:
        print("Error: The file 'char_to_index.json' contains invalid JSON.")

    # convert dataframe to dict
    dataset = df.set_index('path')['label'].to_dict()
    tdataset = tdf.set_index('path')['label'].to_dict()

    # load data
    try:
        train_dataset = data_loader.OCRDataset(args.main_path, dataset, char2index
                                            , input_length=args.input_length
                                            , transform=transforms.Training_transforms)
        train_data = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=args.shuffle
                                , collate_fn=collate_fn, num_workers= 4)
    except RuntimeError as e:
        print("Error: something were wrong while loading train_dataset: %S", e)
    try:
        val_dataset = data_loader.OCRDataset(args.main_path, tdataset, char2index
                                            , input_length=args.input_length
                                            , transform=transforms.Testing_transforms)
        val_data = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=args.shuffle
                              , collate_fn=collate_fn, num_workers= args.num_workers)
    except RuntimeError as e:
        print("Error: something were wrong while loading test_dataset: %S", e)

    print("loading model")
    model = CRNN(num_classes=len(char2index)+1)
    # Define CTC Loss and Optimizer
    criterion = nn.CTCLoss(blank=0)  # 0 as the blank token index
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.device == "cuda" and torch.cuda.is_available():
        print("CUDA is available")
        print("Device count: %s",{torch.cuda.device_count()})
        print("Current device: %s", {torch.cuda.current_device()})
        print("Device name: %s"
                        ,{torch.cuda.get_device_name(torch.cuda.current_device())})
        device = "cuda"
    else:
        print("CUDA is not available. Using CPU instead.")
        device = "cpu"

    checkpoint = {"path": args.checkpointpath, "log_path":args.logpath
                  , "metrics":args.metrics}
    plots = {"path": args.plotpath}

    train_CRNN = training_loop.Train(train_loader = train_data, val_loader = val_data
                             , criterion=criterion, optimizer=optimizer, device = device
                             , char_to_index=char2index
                             , metrics=["character_accuracy", "word_accuracy", "CER"]
                             , num_epochs=args.num_epochs)

    train_CRNN.fit(model=model,plots = plots, checkpoint= checkpoint)
