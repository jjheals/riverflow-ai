import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import pytesseract


class DataCleaner: 
    
    
    def __init__(self, input_dir:str, output_dir:str): 
        self.input_dir = input_dir
        self.output_dir = output_dir
        

    def run(self) -> None: 
        """ 
        Main function for the DataCleaner to clean all the files in self.input_dir and
        output the results to self.output_dir as .avi files.
        """
        # List all files in the folder
        all_files = [f for f in os.listdir(self.input_dir) if '.' in f]
        
        # Filter video files
        is_video = lambda f: f.endswith('mp4') or f.endswith('webm') or f.endswith('mkv')
        video_files = [f for f in all_files if is_video(f)]
        
        # Iterate over the video files and extract the time series for each 
        for f in video_files:
            video_path = os.path.join(self.input_dir, f)
            output_path = os.path.join(self.output_dir, os.path.splitext(f)[0] + '.avi')
            
            # Run until user quits this file
            res = self.extract_time_series(video_path, output_path)
            while not res:
                res = self.extract_time_series(video_path, output_path)
        
        
    def get_ROIs(self, image) -> tuple[np.s_, np.s_]:
        """
        Allows the user to select two ROIs (Regions of Interest) from the image
        using a rectangle selector.

        Args:
            image: The input image.

        Returns:
            List of two slice objects corresponding to the selected ROIs.
        """
        def onselect(eclick, erelease):
            """
            Callback function for rectangle selection.

            Args:
                eclick: MouseEvent when the mouse is clicked.
                erelease: MouseEvent when the mouse is released.
            """
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            rois.append(np.s_[y1:y2, x1:x2])
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none'))
            plt.draw()
        
        # Init an empty array for the ROIs
        rois:list = []
        
        # Init plots for selecting ROIs
        fig, ax = plt.subplots()
        ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        
        # Create a rectangle on mouse events using onselect function
        rs = RectangleSelector(ax, onselect, interactive=True)
        print("Select two ROIs and then press any key to continue...")
        
        plt.show(block=True)
        plt.close(fig)
        
        # Return the selected ROIs
        return rois
    
    
    def extract_year_from_image(self, time_image) -> int:
        """
        Extracts the year from the provided image using OCR.

        Args:
            time_image: Image containing the year.

        Returns:
            Extracted year as an integer. Returns 0 if no valid year is found.
        """
        # Resize image for better OCR accuracy
        width = int(time_image.shape[1] * 4)
        height = int(time_image.shape[0] * 4)
        dim = (width, height)
        resized_image = cv.resize(time_image, dim, interpolation=cv.INTER_LANCZOS4)
        
        # Convert image to grayscale and apply thresholding
        gray_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
        _, threshold = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        blurred_image = cv.GaussianBlur(threshold, (5, 5), 0)
        _, threshold = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        # Invert image if necessary
        top_right_pixel_value = threshold[0, -1]
        if top_right_pixel_value < 127:
            threshold = cv.bitwise_not(threshold)
        
        # OCR to extract digits
        config = '--oem 3 --psm 6 outputbase digits'
        text = pytesseract.image_to_string(threshold, config=config)
        text = "".join([i for i in text if i in '0123456789'])
        
        # Return 0 if no text is found
        if not text: return 0
        
        # Convert the text to an int & check if it's within a reasonable range 
        text = int(text)
        if text < 1970 or text > 2024: return 0
        
        # Return the year
        return text
    
    
    def extract_time_series(self, video_path, output_path) -> bool:
        """
        Extracts a time series from the video and saves it along with corresponding frames.

        Args:
            video_path: Path to the input video file.
            output_path: Path to the output video file.

        Returns:
            True if extraction and saving are successful, False otherwise.
        """
        
        # Init capture obj
        cap:cv.VideoCapture = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'Error: Could not open video at {video_path}')
            return False

        ret, frame = cap.read()
        if not ret:
            print(f'Error: Failed to read capture')
            return False
        
        # Select ROIs
        rois = self.get_ROIs(frame)
        if len(rois) != 2:
            print("Error: Please select at least two ROIs.")
            return False
            
        time_roi, river_roi = rois

        last_year:int = 0
        time_series_data:list = []
        output_images:list = []
        
        # Run while video capture is open
        while cap.isOpened():
            
            # Read the frame
            ret, frame = cap.read()
            if not ret: break
            
            # Get the new year and make sure the year is next in the time series 
            # (i.e. last_year != new_year or the video didn't reset back to a prior year)
            new_year:int = self.extract_year_from_image(frame[time_roi])
            if new_year <= last_year: continue

            print(f'Read frame from {new_year}')
            
            
            time_series_data.append(new_year)       # Append the new year to the time series data for tracking 
            output_images.append(frame[river_roi])  # Append the frame to the output images
            last_year = new_year                    # Move to the next year
        
        # Release the capture 
        cap.release()
        
        # Check that the time series data is spaced by one year 
        if DataCleaner.spaced_by_one(time_series_data):
            
            # Write time series data to file
            with open(output_path.split('.')[0] + '_timeseries.txt', 'w') as f:
                for year in time_series_data:
                    f.write(f"{year}\n")
            
            # Write output video
            out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'XVID'), 2.0, (output_images[0].shape[1], output_images[0].shape[0]))
            for image in output_images:
                out.write(image)

            out.release()
            
            # Print debug info
            observed:int = len(time_series_data)
            expected:int = time_series_data[-1] - time_series_data[0] + 1        
            print(f'Approximate years dropped: {expected} - {observed} = {expected - observed}')
            
            # Return true for success
            return True
        else:
            print('FAIL')
            # Return false for failed 
            return False


    @staticmethod
    def spaced_by_one(x) -> bool:
        """
        Checks if elements in the list are spaced by one.

        Args:
            x: List of integers.

        Returns:
            True if elements are spaced by one, False otherwise.
        """
        current = x[0]
        for item in x[1:]:
            current += 1
            if current != item:
                return False
        return True