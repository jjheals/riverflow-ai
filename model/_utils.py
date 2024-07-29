import numpy as np 
import cv2
import os

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_data(dataset_path:str) -> np.ndarray: 
    dataset = np.load(dataset_path)
    print(f'\033[92mLoaded data from "{dataset_path}"\033[0m')
    print(f'\033[90mDataset shape: {dataset.shape}\033[0m')
    
    return dataset


def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y


def load_images_from_folder(folder, img_size=(256, 256)):
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img)
        else:
            print(f"Warning: Could not load image {filename}")
    return images


def create_dataset_from_folders(parent_folder, img_size=(256, 256)):
    sequences = []
    for folder in sorted(os.listdir(parent_folder)):
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            images = load_images_from_folder(folder_path, img_size)
            if len(images) > 30:  # Ensure there are at least two images in the sequence
                sequences.append(images)
            else:
                print(f"Warning: Folder {folder} does not contain enough images.")
    return sequences


def pad_sequences_to_same_length(sequences):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', dtype='float64')
    return padded_sequences