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

# NOTE: to be able to accurately score the model, we have to cut the first 5 frames from the prediction because all predictions have 
# the first five frames as white - this is an issue that comes from the padding of sequence lengths, but is consistent with all 
# predictions. This also means we will have to cut the first 5 frames off the truth frames as well. To compensate, we increase the
# num_frames_predict by this value as well; this way, we cut IDX_TRIM frames off the beginning of the truth and predicted sequences
# but we still have num_frames_predict frames.
IDX_TRIM:int = 5
num_frames_predict += IDX_TRIM

# Create the output dirs if they does not exist 
os.makedirs(output_dir, exist_ok=True)


# ----- Load the model & val data ----- #
model = load_model(config['output-model-path'])
print(f"\033[92mModel loaded from {config['output-model-path']}\033[0m")

# Load the val dataset
val_dataset:np.ndarray = load_data(config['val-dataset-path'])


# ----- PREDICTIONS ----- #
print("\033[93mNOTICE: \033[90mPredictions Start\033[0m")

n:int = 0
while n < num_predictions: 
    
    # Select a random example from the validation dataset
    example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]
    
    # Extract frames for prediction and add batch dim
    frames = example[:num_frames_predict]
    frames_input = np.expand_dims(frames, axis=0)  # Shape will be (1, num_frames_predict, height, width, channels)
    
    # Predict the next frames
    new_predictions = model.predict(frames_input)
    
    # Remove batch dimension
    predicted_frames = np.squeeze(new_predictions, axis=0)  # Shape will be (num_frames_predict, height, width, channels)

    # Convert predicted frames to uint8 and ensure correct format
    predicted_frames = (predicted_frames * 255).astype(np.uint8)
    predicted_frames = predicted_frames[IDX_TRIM:]
    
    # Scale predicted frames using Z-scores to get an image that is closer to black and white instead of a shade of grey
    predicted_frames = z_score_scale_frames(predicted_frames)
    
    # Check if all black prediction
    if not predicted_frames.any():
        #print(f'\033[91mERROR: \033[90mall black prediction for run {n}\033[0m')
        continue 
    
    print(f'\n\033[92mMaking prediction {n}/{num_predictions}')
    
    # Construct an output dir for this prediction/truth pair
    n_dir:str = os.path.join(output_dir, f'run_{n}')
    os.makedirs(n_dir, exist_ok=True)
    
    # Construct the subdirs for the gifs and csvs
    gif_output_dir:str = os.path.join(n_dir, 'gifs')
    csv_output_dir:str = os.path.join(n_dir, 'csvs')
    
    os.makedirs(gif_output_dir, exist_ok=True)
    os.makedirs(csv_output_dir, exist_ok=True) 
    
    # Write the truth frames for comparison
    truth_frames_filename:str = f'example_TRUTH_{n}.gif'
    truth_frames = np.squeeze(frames)
    truth_frames = (truth_frames * 255).astype(np.uint8)
    truth_frames = truth_frames[IDX_TRIM:]
    
    # Z-score the truth frames if configured 
    if config['z-score-truth-frames']:
        truth_frames = z_score_scale_frames(truth_frames)
    
    # Write truth gif
    truth_gif_path = os.path.join(gif_output_dir, truth_frames_filename)
    with open(truth_gif_path, 'wb') as truth_gif_file:
        imageio.mimsave(truth_gif_file, truth_frames, 'GIF', duration=1000)

    print(f'\033[93mNOTICE: \033[90mSaved the truth frames ({len(truth_frames)}) to "{truth_gif_path}')

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
        csv_filename = f'run_{n}-predicted_frame_{i + IDX_TRIM}.csv'
        csv_path = os.path.join(csv_output_dir, csv_filename)
        
        # Save DataFrame to CSV
        df.to_csv(csv_path, index=False, header=False)
        
        
    # Save the truth frames to CSV
    for i, frame in enumerate(truth_frames):        
        # Convert frame to DataFrame
        df = pd.DataFrame(np.squeeze(frame))
        
        # Define the CSV filename
        csv_filename = f'run_{n}-truth_frame_{i + IDX_TRIM}.csv'
        csv_path = os.path.join(csv_output_dir, csv_filename)
        
        # Save DataFrame to CSV
        df.to_csv(csv_path, index=False, header=False)
        
                
    # Increment n
    n+=1