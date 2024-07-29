import numpy as np 
import cv2
import os
from typing import Iterable 

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_data(dataset_path:str) -> np.ndarray: 
    """Loads and returns the dataset from the given path. Expects a dataset in .npy format."""
    
    dataset = np.load(dataset_path)
    print(f'\033[92mLoaded data from "{dataset_path}"\033[0m')
    print(f'\033[90mDataset shape: {dataset.shape}\033[0m')
    
    return dataset


def create_shifted_frames(data:Iterable | np.ndarray) -> tuple[list, list]:
    """Takes in a dataset and returns a time-series-like couple of arrays, where every value in 
    y represents the "next value" in the corresponding value in X. In other words, for every X[i],
    the corresponding y value is at index i + 1 in the original dataset."""
    
    X = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return X, y


def load_images_from_folder(folder:str, img_size:tuple[int, int]) -> list:
    """Takes in the path to a folder and an image size, reads all the images in the folder,
    normalizes them, and returns a list of the images. NOTE: assumes greyscale images."""
    # Init list of images to return
    images:list = []
    
    # Iterate over all the files in the given folder
    for filename in sorted(os.listdir(folder)):
        
        # Construct the path to the image and read it
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        
        # Check that the image was read successfully
        if img is not None:
            img = cv2.resize(img, img_size)  # Resize
            img = img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img)               # Add to the list of images
        else:
            print(f"Warning: Could not load image {filename}")
    
    # Return the populated list of images
    return images


def create_dataset_from_folders(parent_folder:str, img_size:tuple[int, int]) -> list[list]:
    """Takes in the path to a folder and an image size, and reads all the images from the given folder; calls
    load_images_from_folder to read, resize, and normalize the images in each sequence, and returns a 2D list
    of sequences, where each sequence is a series of images in a subdir of the given parent_folder."""
    # Init list of sequences to return 
    sequences:list = []
    
    # Iterate over all the folders in the given directory 
    for folder in sorted(os.listdir(parent_folder)):
        
        # Construct the folder path and confirm that it is a folder 
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            
            # Load the sequence of images
            images = load_images_from_folder(folder_path, img_size)
            
            # Ensure there are at least two images in the sequence and add the sequence to the list
            if len(images) > 30:  sequences.append(images)
            else:
                # Print warning to the user if the folder does not have at least 30 images
                print(f"\033[93mWARNING: \033[90mFolder {folder} does not contain enough images.\033[0m")
                
    # Return the 2D list of sequences
    return sequences


def pad_sequences_to_same_length(sequences:Iterable):
    """Takes in a 2D list of sequences and adds padding to make them all the same length."""
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', dtype='float64')
    return padded_sequences