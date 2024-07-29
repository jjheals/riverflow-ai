import numpy as np
import tensorflow as tf 
import json 
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow.keras.optimizers import Adam

from _utils import load_data, create_shifted_frames


# ----------------------------------------------------------------------- #
# Load config
with open('config/config.json', 'r') as file: 
    config:dict = json.load(file)

input_dir:str = config['seg-pngs-dir']
train_split:float = config['train-split']

# Setting hyperparams from config
epochs:int = config['hyperparams']['epochs']
batch_size:int = config['hyperparams']['batch-size']

# Configure to use GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\033[93mNOTICE: \033[90musing GPU: {gpus}\033[0m")
    except RuntimeError as e:
        print(e)
else:
    print("\033[93mNOTICE: \033[90mNo GPUs found.\033[0m")

# Load the dataset from disk
train_dataset:np.ndarray = load_data(config['train-dataset-path'])
val_dataset:np.ndarray = load_data(config['val-dataset-path'])

# Create the X and y sets by shifting the frames 
X_train, y_train = create_shifted_frames(train_dataset)
X_val, y_val = create_shifted_frames(val_dataset)

# Inspect the datasets
print("\033[92mTraining Dataset Shapes: \033[90m" + str(X_train.shape) + ", " + str(y_train.shape) + '\033[0m')
print("\033[92mValidation Dataset Shapes: \033[90m" + str(X_val.shape) + ", " + str(y_val.shape) + '\033[0m')

# Construct the input layer with no definite frame size
inp = layers.Input(shape=(None, *X_train.shape[2:]))

# Define the model
model = Sequential([
    ConvLSTM2D(filters=64, kernel_size=(1, 1), padding='same', return_sequences=True, activation='relu'),
    BatchNormalization(),
    ConvLSTM2D(filters=64, kernel_size=(1, 1), padding='same', return_sequences=True, activation='relu'),
    BatchNormalization(),
    ConvLSTM2D(filters=64, kernel_size=(1, 1), padding='same', return_sequences=True, activation='relu'),
    Conv3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid', padding='same')
])
model.compile(optimizer=Adam(), loss='mean_squared_error')

print('\033[93mNOTICE: \033[90mcompiled the model.\033[0m')

# Fit the model to the training data.
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
)

# Save the entire model
model.save(config['output-model-path'])
print(f"\033[92mModel saved to {config['output-model-path']}\033[0m")