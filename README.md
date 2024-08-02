# riverflow-ai

Predicting river erosion and path changes is crucial for managing ecological balance, agriculture, and human settlements. This report leverages satellite imagery and state-of-the-art AI methodologies to forecast these changes, aiming to mitigate the devastating effects of riverbank erosion, such as agricultural land loss, infrastructure destruction, water pollution, and community displacement. Current methods, including Convolutional Neural Networks (CNNs), Long Short-Term Memory networks (LSTMs), Vision Transformers (VITs), and Superpixel Transformers, each offer unique advantages and challenges. This project combines Superpixel segmentation and LSTMs to predict river erosion using Google Earth Timelapse Engine videos from 1984 to 2022. Preprocessing involves extracting regions of interest and chronological sorting of images. Superpixel segmentation simplifies the classification problem, while LSTMs handle the temporal aspects of river morphology. The proposed model's performance will be evaluated using accuracy, precision, recall, F1 score, and Matthews correlation coefficient (MCC), with a comparative analysis against traditional CNN models. This approach aims to provide valuable insights for environmental management and disaster prevention, highlighting the potential of AI in predicting and mitigating the impacts of natural disasters.


# Usage


## Configuration Settings 

The config JSON file is located at [config/config.json](config/config.json). 

Since the workflow is split into multiple files to help mitigate limitations due to hardware and to make the process easier to follow, there is a [config JSON file](config/config.json) that helps standardize the inputs and outputs across files. This is important because, for example, the "train-dataset-path" is used to output the train dataset .npy file in the construct dataset step, but it is also used in the train model step to load the train dataset. Using the config file makes it easier to keep track of these cross-referenced paths and other variables. 

The following table describes each variable in the config file. Note that *all of the variables are required, and the scripts will throw errors if any are missing or not properly defined.*

| Variable name | Data type | Description | 
| ------------- | --------- | ----------- | 
| seg-pngs-dir | str | Path to the folder containing the segmented PNGs, with subdirectories for each river, each containing ordered PNG files (e.g. 0.png, 1.png, 2.png, ...) | 
| dataset-path | str | Path to output the dataset.npy file in the [construct dataset step](#construct-the-dataset) | 
| train-dataset-path | str | Path to output the train_dataset.npy file in the [construct dataset step](#construct-the-dataset), which is also read in the [train the model step](#train-the-model) |
| val-dataset-path | str | Path to output the val-dataset.npy file in the [construct dataset step](#construct-the-dataset), which is also read in the [train the model step](#train-the-model) |
| output-model-path | str | Path to output the trained model in the [train the model step](#train-the-model), and to read the model from when [testing the model](#test-the-model) | 
| output-dir | str | Path to output the results from the [test the model step](#test-the-model); will be created if it does not exist, and a subdir for each prediction will be created, and two more subdirs will be created in each of the prediction folders for "gifs" and "csvs" | 
| image-width | int | Define a standard width for the images; this variable is used in each of the scripts. Note that this value can range depending on your available hardware and computational power. Setting a lower value (e.g. 64 or 128) may impact the accuracy of results, but higher values may require significantly more resources (see [train the model step](#train-the-model) for explanation) | 
| image-height | int | Define a standard height for the images; see image-width above. | 
| train-split | float | Define what percentage of the dataset should be used for training; the complement will be used for the validation set | 
| hyperparams["epochs"] | int | Number of epochs when training | 
| hyperparams["batch-size"] | int | Batch size for training | 
| num-frames-predict | int | Number of frames to predict when generating predictions | 
| num-predictions | int | Number of predictions to make and output to the "output-dir" | 
| z-score-truth-frames | bool | Specify whether to z-score scale the truth frames to pixel values of binary 0 \| 255; see [test the model](#notes-about-model-testing) section for more discussion about this setting and why it is included/the output effects | 

The default config file contains the following settings: 
```json 
{
    "seg-pngs-dir": "data/segmented_pngs",
    "dataset-path": "data/dataset/dataset.npy",
    "train-dataset-path": "data/dataset/train_dataset.npy",
    "val-dataset-path": "data/dataset/val_dataset.npy",
    "output-model-path": "model/trained_models/trained_model-no-callbacks.keras",
    "output-dir": "data/outputs",
    "image-width": 128,
    "image-height": 128,
    "train-split": 0.8,
    "hyperparams": {
        "epochs": 10,
        "batch-size": 3
    },
    "num-frames-predict": 15,
    "num-predictions": 10
}
```

## Workflow 

To use the model, the workflow is as follows: 

0. [Clean the data](#cleaning)
1. [Segmentation](#segmentation)
2. [Construct the dataset](#construct-the-dataset)
3. [Train the model](#train-the-model)
4. [Test the model](#test-the-model) 
5. [Evaluate performance](#evaluate-performance)


## Cleaning


### DataCleaner 

The [DataCleaner](./cleaning/DataCleaner.py) class is used to extract the timeframe from an image and to save the image with an appropriate filename to the specified output directory. The [Cleaner.py](./cleaning/Cleaner.py) file implements the DataCleaner. The DataCleaner is not meant to be a fully-functional, universal tool - it was built for a very specific purpose and for this very specific implementation, not a wider application. 


### NameCleaner 

The [NameCleaner](./cleaning/NameCleaner.py) script is a standalone script that, like DataCleaner, is not meant to be a universal tool and was built for this very specific purpose. It simply standardizes and cleans up the filenames of the files in the given input folder. 


## Segmentation 

The [1-segmentation.ipynb](model/1-segmentation.ipynb) notebook walks through the segmentation process. 

**Note:** this notebook was developed in Google Colab and does not directly integrate with the rest of this project structure. It is meant to provide a snapshot of the process, not to be universally implemented.

To avoid having to implement the segmentation process yourself, you can download the [segmented_pngs dataset](https://drive.google.com/file/d/1cDMDohsJj10xV_y_AbzXsv0sun4TxXh5/view?usp=sharing) (hosted on Google Drive) that contains all of the segmented PNGs that can be used to train the model. Simply extract the ZIP into the [data/](data/) folder and make sure the [config](#configuration-settings) is set with the correct "seg-pngs-path" variable. 


## Construct the Dataset

After cleaning the data, the first step is to construct the dataset from the images and conduct the necessary preprocessing. This is done by the [1-construct_dataset.py](model/2-construct_dataset.py) script in the [model](model/) folder. 

*Note: This process was split from the training phase because, depending on the size of the dataset and computational power available, it can take a very long time.*

This script reads in the segmented PNGs from the path specified in the [config JSON](#configuration-settings), normalizes the pixel values, pads the sequences to the same length, resizes the images to the same size (specified in the config file), and constructs three files: 

| Filename | Path Variable in Config JSON | Description | 
| -------- | ---------------------------- | ----------- | 
| dataset.npy | dataset-path | The full dataset after it's been normalized. Contains arrays for each of the image sequences. | 
| train_dataset.npy | train-dataset-path | Subset of dataset.npy containing the train images that were selected. | 
| val_dataset.npy | val-dataset-path | Subset of dataset.npy containing the val/test images that were selected. |


## Train the Model

The [training step](model/3-train_model.py) reads in the train and val datasets created by the [previous step](#construct-the-dataset) and trains the Convolutional LSTM model. The model has six layers: 
* 2D ConvLSTM 
* Batch Normalization
* 2D ConvLSTM 
* Batch Normalization
* 2D ConvLSTM 
* 3D Convolutional Neural Network

Note that, due to hardware limitations, the kernel size in all of the Conv layers (layers 1, 3, 5, 6) is set to 1x1. This is because compressing the images to a size of 128x128 pixels causes the features to be much less obvious, which means the model must have a more granular kernel to extract them.

For example, the original images (before constructing the dataset and resizing to a standard size) may be near 1000x1000 pixels. Reducing an image of 1000x1000 pixels down to 128x128 pixels condenses the features by nearly ten-fold, meaning each feature is defined by roughly a factor of 1/10th of the original number of pixels. Too large of a kernel will miss these abnormally small features. 

This could be remedied by increasing the standard image size, which unfortunately was not possible in our experiments due to hardware limitations. 

The trained model will be written to a .keras folder at the path defined in the config JSON by the "output-model-path" variable.

## Test the Model

[Testing the model](model/4-test_model.py) reads in the validation set and selects [num-predictions] random rivers, and for each river predicts the next [num-frames-predict] frames. 

The outputs are written to the [output-dir] in the following format, where N is in the range [0, num-predictions) and i is in the range [0, num-frames-predict): 

```
output-dir/
├── run_N/
│   ├── cvs/
│   │   ├── predicted_frame_i.csv
│   │   ├── ...
│   │   ├── truth_frame_i.csv
│   │   └── ...
│   └── gifs/
│       ├── example_TRUTH_N.gif
│       └── predicted_frames_N.gif
├── ...
```
### Notes about model testing 

There are three primary issues with the model that came up in testing; the [4-test_model.py](model/4-test_model.py) has been altered to account for these issues, but they are still important to discuss. We believe all these issues stem from the fact that all the sequences of river images are padded to be of a consistent length, and as a result, when the model predicts a sequence, it thinks the previous sequence ends with all white frames and that there are more "white" values than there actually should be.

The first issue that arises because of this padding is that the model consistently predicts all white frames for the first 4-5 frames of a predicted sequence. We accounted for this by dropping the first IDX_TRIM frames (set to 5 in the code, but can be altered) from the predicted frames AND truth frames, and increasing the number of frames to predict by IDX_TRIM frames as well. This way, the model drops the first IDX_TRIM frames from the beginning but predicts IDX_TRIM extra frames at the end, effectively ending with the same amount of frames as if this trimming was not applied.

The second issue stems from the same fundamental issue with padded sequences, and is that the model was predicting frames that were a shade of grey, rather than values close to 0 or 255 for each pixel. We accounted for this by using z-score scaling to scale the pixel values in each frame to 0 or 255 depending on their distance from the mean (e.g. values < mean go to 0, values > mean go to 255). This creates a predicted frame that consists of pixels with a binary value of 0 or 255; however, to accurately test the model's performance with this alteration, we had to also scale the truth frames using the same method, which caused some ambiguity in the results, since the evaluation measures, such as precision, are now slightly skewed (likely in favor of the model).

The third issue with the model's performance is that it occasionally would predict a sequence of frames that were all black. We believe this stems from the same issue regarding padding sequences, where when the model predicted a sequence, it would get "stuck" at a local minima and an all black prediction would be the "statistically most accurate" sequence. To counter this, we added a conditional in the test model script that checks if all the predicted frames in a sequence are black, and if they are, it throws out that sample and picks a new sample of num_frames_predict frames to test. 

## Evaluate Performance

The [5-evaluate-performance.py](model/5-evaluate-performance.py) script evaluates the performance of the model using four measures: 

* Precision
* Recall
* F1 Score
* Matthew's Correlation Coefficient (MCC) 

The eval performance script prints the score for each individual frame, then prints the average across all frames for each of the four measures. 
