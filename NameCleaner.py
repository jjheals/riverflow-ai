import os

folder = "Data"

if not os.path.exists(folder):
    print(f"Error: Folder '{folder}' does not exist.")
else:
    for filename in os.listdir(folder):
        if ' ' in filename:
            new_filename = filename.replace(' ', '_')
            try:
                os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))
                print(f"Renamed '{filename}' to '{new_filename}'.")
            except Exception as e:
                print(f"Error renaming '{filename}': {e}")
        else:
            print(f"No spaces found in '{filename}', skipping.")
    
    print("File renaming completed.")
