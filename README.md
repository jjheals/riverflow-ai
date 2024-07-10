# riverflow-ai

# Usage

## Cleaning

### DataCleaner 

The [DataCleaner](./cleaning/DataCleaner.py) class is used to extract the timeframe from an image and to save the image with an appropriate filename to the specified output directory. The [Cleaner.py](./cleaning/Cleaner.py) file implements the DataCleaner. The DataCleaner is not meant to be a fully-functional, universal tool - it was built for a very specific purpose and for this very specific implementation, not a wider application. 

### NameCleaner 

The [NameCleaner](./cleaning/NameCleaner.py) script is a standalone script that, like DataCleaner, is not meant to be a universal tool and was built for this very specific purpose. It simply standardizes and cleans up the filenames of the files in the given input folder. 
