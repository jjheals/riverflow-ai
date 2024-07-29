import json 
import numpy as np 
from _utils import create_dataset_from_folders, pad_sequences_to_same_length


# Load config
with open('config/config.json', 'r') as file: 
    config:dict = json.load(file)
    
# Recreate the dataset using the PNGs
print(f'\033[92mLoading PNGs from "{config["seg-pngs-dir"]}"...\033[0m')

img_size:tuple[int, int] = (config['image-width'], config['image-height'])  
sequences:list = create_dataset_from_folders(config["seg-pngs-dir"], img_size)
padded_sequences = pad_sequences_to_same_length(sequences)

dataset = np.array(padded_sequences)
np.save(config['dataset-path'], dataset)
print(f'\033[93mNOTICE: \033[90mSaved dataset to "{config["dataset-path"]}"\033[0m')

# Split into train and validation sets using indexing to optimize memory
indexes = np.arange(dataset.shape[0])
np.random.shuffle(indexes)

train_index = indexes[: int(config['train-split'] * dataset.shape[0])]
val_index = indexes[int(config['train-split'] * dataset.shape[0]) :]

train_dataset:np.ndarray = dataset[train_index]
val_dataset:np.ndarray = dataset[val_index]

# Save the train and val datasets
np.save(config['train-dataset-path'], train_dataset)
print(f'\033[93mNOTICE: \033[90msaved train dataset to {config["train-dataset-path"]}\033[0m')

np.save(config['val-dataset-path'], val_dataset)
print(f'\033[93mNOTICE: \033[90msaved train dataset to {config["val-dataset-path"]}\033[0m')
