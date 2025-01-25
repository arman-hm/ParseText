"""
test Module

Author: Arman Hajimirza
Date: January 25, 2025

This module handles the data loading and OCR (Optical Character Recognition) process 
using a pre-trained CRNN model. It reads images from a specified directory, performs 
text recognition, and saves the results to a CSV file.

It provides the following functionalities:
- Argument Parsing: Parses command-line arguments for various configurations like paths,
  model weights, device type, etc.
- Data Loading: Loads images from a specified directory.
- Error Handling: Implements error handling for loading images and JSON files.
- Text Recognition: Utilizes a pre-trained CRNN model for recognizing text in images.
- Result Saving: Saves the recognition results to a CSV file.

Modules and Libraries:
- argparse: For parsing command-line arguments.
- json: For handling JSON file operations.
- os: For interacting with the operating system.
- time: For measuring the duration of the text recognition process.
- PIL (Pillow): For image processing.
- pandas: For data manipulation and analysis.
- text_recognizer: Custom module for text recognition.

Arguments:
- --imgdir: Path to the directory containing image files for OCR.
- --CRNN: Path to the pre-trained CRNN model weights.
- --outputdir: Path to the directory where the results will be saved.
- --label2index: Path to the JSON file containing character-to-index mapping.
- --device: Device to load the model on (e.g., 'cuda' or 'cpu').

Example Usage:
    python script_name.py --imgdir test/page_0007
                          --CRNN model_weights/best_model_Val_word_accuracy.pth
                          --outputdir result --label2index char_to_index.json --device cuda
"""
import argparse
import json
import os
import time
from PIL import Image, UnidentifiedImageError
import pandas as pd
import text_recognizer



parser = argparse.ArgumentParser(description='Test Parsetext OCR')
parser.add_argument('--imgdir', default='test/page_0007', type=str,
                     help='image dir for test module')
parser.add_argument('--CRNN', default='model_weights/best_model_Val_word_accuracy.pth', type=str,
                     help='path of model weight')
parser.add_argument('--outputdir', default='result', type=str,
                     help='outputdir for saving results')
parser.add_argument('--label2index', default='char_to_index.json', type=str,
                     help='Path to labels index json file')
parser.add_argument('--device', default="cuda", type=str,
                     help='type of device for load model on that device')
args = parser.parse_args()

def load_images_from_dir(dir_path):
    """
    Load all image files from a specified directory and return two lists:
    one with the PIL Image objects and the other with the corresponding file names.

    Args:
        dir_path (str): The path to the directory containing image files.

    Returns:
        tuple: A tuple containing two lists:
            - images (list): List of PIL Image objects.
            - image_files (list): List of file names corresponding to the images.

    Note:
        This function processes files with the following extensions:
        .jpg, .jpeg, .png, .gif, .bmp, .tiff

    Example:
        images, image_files = load_images_from_dir('path_to_your_directory')
    """
    allowed_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_files = []
    images = []
    for file_name in os.listdir(dir_path):
        if file_name.lower().endswith(allowed_extensions):
            file_path = os.path.join(dir_path, file_name)
            try:
                with Image.open(file_path) as img:
                    images.append(img.copy())
                    image_files.append(file_name)
            except UnidentifiedImageError:
                print(f"Failed to load image {file_name}: Unidentified image format.")
            except IOError as e:
                print(f"Failed to load image {file_name}: I/O error - {e}")
            except Exception as e:
                print(f"Failed to load image {file_name}: {e}")

    return images, image_files

if __name__ == '__main__':
    # load images into list of PIL images
    images, image_files = load_images_from_dir(args.imgdir)
    # load label_to_index
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

    tr = text_recognizer.TextRecognizer(model_path= args.CRNN, char_index=char2index
                                        , device=args.device)
    # inference images and get time taken for process.
    start_time = time.time()
    result = tr.inference(images=images)
    end_time = time.time()
    duration = end_time - start_time

    # save result in csv beside their files names to clarify
    data = pd.DataFrame({"path":image_files , "label": result})
    df_path = os.path.join(args.outputdir,"data.csv")
    data.to_csv(df_path)
    print("\n----------------------------------------------------------")
    print(f"Result saved as csv in 'label' column in: {df_path}")
    print("#####",f"Time taken for inference {len(images)} images: {duration:.4f}S","#####")
    print("----------------------------------------------------------\n")
