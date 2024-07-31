import numpy as np 
import pandas as pd
from _utils import load_data, z_score_scale_frames
import json 
import os 
from PIL import Image
import imageio

from tensorflow.keras.models import load_model


# ----- Config ----- #
with open('config/config.json', 'r') as file: 
    config:dict = json.load(file)
    
output_dir:str = config['output-dir']
num_frames_predict:int = config['num-frames-predict']
num_predictions:int = config['num-predictions'] 

# Create the output dirs if they does not exist 
os.makedirs(output_dir, exist_ok=True)


# ----- Load the model & val data ----- #
model = load_model(config['output-model-path'])
print(f"\033[92mModel loaded from {config['output-model-path']}\033[0m")

# Load the val dataset
val_dataset:np.ndarray = load_data(config['val-dataset-path'])


# ----- PREDICTIONS ----- #
print("\033[93mNOTICE: \033[90mPredictions Start\033[0m")

for n in range(0, num_predictions): 
    print(f'\n\033[92mMaking prediction {n}/{num_predictions}')
    
    # Construct an output dir for this prediction/truth pair
    n_dir:str = os.path.join(output_dir, f'run_{n}')
    os.makedirs(n_dir, exist_ok=True)
    
    # Construct the subdirs for the gifs and csvs
    gif_output_dir:str = os.path.join(n_dir, 'gifs')
    csv_output_dir:str = os.path.join(n_dir, 'csvs')
    
    os.makedirs(gif_output_dir, exist_ok=True)
    os.makedirs(csv_output_dir, exist_ok=True) 
    
    # Select a random example from the validation dataset.
    example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]

    # Extract frames for prediction
    frames = example[:num_frames_predict]

    # Write the truth frames for comparison
    truth_frames_filename:str = f'example_TRUTH_{n}.gif'
    truth_frames = np.squeeze(frames)
    truth_frames = (truth_frames * 255).astype(np.uint8)
    truth_gif_path = os.path.join(gif_output_dir, truth_frames_filename)

    with open(truth_gif_path, 'wb') as truth_gif_file:
        imageio.mimsave(truth_gif_file, truth_frames, 'GIF', duration=1000)

    print(f'\033[93mNOTICE: \033[90mSaved the truth frames ({len(truth_frames)}) to "{truth_gif_path}')

    # Add batch dimension
    frames_input = np.expand_dims(frames, axis=0)  # Shape will be (1, num_frames_predict, height, width, channels)
    print(f'frames_input.shape (after expanding dims) = {frames_input.shape}')

    # Predict the next frames
    new_predictions = model.predict(frames_input)
    print(f'new_predictions.shape (before squeezing) = {new_predictions.shape}')

    # Remove batch dimension
    predicted_frames = np.squeeze(new_predictions, axis=0)  # Shape will be (num_frames_predict, height, width, channels)
    print(f'predicted_Frames.shape (after squeezing new_predictions) = {predicted_frames.shape}')

    # Convert predicted frames to uint8 and ensure correct format
    predicted_frames = (predicted_frames * 255).astype(np.uint8)

    # Scale predicted frames using Z-scores to get an image that is closer to black and white instead of a 
    # shade of grey
    predicted_frames = z_score_scale_frames(predicted_frames)

    # Save predicted frames as a GIF using PIL
    predicted_gif_path = os.path.join(gif_output_dir, f'predicted_frames_{n}.gif')
    images = [Image.fromarray(frame.squeeze().squeeze(), 'L') for frame in predicted_frames]  # Use 'L' mode for grayscale
    images[0].save(predicted_gif_path, save_all=True, append_images=images[1:], duration=1000, loop=0)
    print(f'\033[93mNOTICE: \033[90mSaved output gifs to {gif_output_dir}\033[0m\n')


    # ----- Save Frames as CSV ----- #
    for i, frame in enumerate(predicted_frames):
        # Convert frame to DataFrame
        df = pd.DataFrame(np.squeeze(frame))
        
        # Define the CSV filename
        csv_filename = f'predicted_frame_{i}.csv'
        csv_path = os.path.join(csv_output_dir, csv_filename)
        
        # Save DataFrame to CSV
        df.to_csv(csv_path, index=False, header=False)
        
        print(f'\033[93mNOTICE: \033[90mSaved frame {i} to "{csv_path}"\033[0m')

    # Save the truth frames to CSV
    for i, frame in enumerate(truth_frames):
        # Convert frame to DataFrame
        df = pd.DataFrame(np.squeeze(frame))
        
        # Define the CSV filename
        csv_filename = f'truth_frame_{i}.csv'
        csv_path = os.path.join(csv_output_dir, csv_filename)
        
        # Save DataFrame to CSV
        df.to_csv(csv_path, index=False, header=False)
        
        print(f'\033[93mNOTICE: \033[90mSaved frame {i} to "{csv_path}"\033[0m')