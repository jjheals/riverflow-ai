import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import pytesseract

folder = "Data"
target_folder = "CleanData"
all_files = [f for f in os.listdir(folder) if '.' in f]
is_video = lambda f: f.endswith('mp4') or f.endswith('webm') or f.endswith('mkv')
video_files = [f for f in all_files if is_video(f)]

def get_ROIs(image) -> [np.s_, np.s_]:
    rois = []
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        rois.append(np.s_[y1:y2, x1:x2])
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none'))
        plt.draw()
    fig, ax = plt.subplots()
    ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    rs = RectangleSelector(ax, onselect, interactive=True)
    print("Select two ROIs and then press any key to continue...")
    plt.show(block=True)
    plt.close(fig)
    return rois

def extract_year_from_image(time_image) -> int:
    width = int(time_image.shape[1] * 4)
    height = int(time_image.shape[0] * 4)
    dim = (width, height)
    resized_image = cv.resize(time_image, dim, interpolation=cv.INTER_LANCZOS4)
    gray_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    blurred_image = cv.GaussianBlur(threshold, (5, 5), 0)
    _, threshold = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    top_right_pixel_value = threshold[0, -1]
    if top_right_pixel_value < 127:
        threshold = cv.bitwise_not(threshold)
    config = '--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(threshold, config=config)
    text = "".join([i for i in text if i in '0123456789'])
    if text == "" or text == None:
        return 0
    text = int(text)
    if text < 1970 or text > 2024:
        return 0
    return text
    
def spaced_by_one(x) -> bool:
    current = x[0]
    for item in x[1:]:
        current += 1
        if current != item:
            return False
    return True

def extract_time_series(video_path, output_path) -> bool:
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Error: Could not open video at {video_path}')
        return False

    ret, frame = cap.read()
    if not ret:
        print(f'Error: Failed to read capture')
        return False
    
    rois = get_ROIs(frame)
    if len(rois) != 2:
        print("Error: Please select at least two ROIs.")
        return False
        
    time_roi, river_roi = rois

    last_year = 0
    time_series_data = []
    output_images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        new_year = extract_year_from_image(frame[time_roi])
        if new_year <= last_year:
            continue

        print(f'Read frame from {new_year}')
        time_series_data.append(new_year)
        last_year = new_year
        output_images.append(frame[river_roi])
    cap.release()
    
    if time_series_data is not []:
    #if spaced_by_one(time_series_data):
        with open(output_path.split('.')[0] + '_timeseries.txt', 'w') as f:
            for year in time_series_data:
                f.write(f"{year}\n")
        
        out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'XVID'), 2.0, (output_images[0].shape[1], output_images[0].shape[0]))
        for image in output_images:
            out.write(image)
                    
        out.release()
        observed = len(time_series_data)
        expected = time_series_data[-1]-time_series_data[0]+1        
        print(f'Approximate frames dropped: {expected}-{observed}={expected-observed}')
        return True
    else:
        return False

for f in video_files:
    video_path = os.path.join(folder, f)
    output_path = os.path.join(target_folder, os.path.splitext(f)[0] + '.avi')
    res = extract_time_series(video_path, output_path)
    while not res:
        res = extract_time_series(video_path, output_path)