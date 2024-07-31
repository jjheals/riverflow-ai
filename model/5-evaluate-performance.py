import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

# Define the paths to the folders
ground_truth_folder = 'C:\\Users\\Adam\\Documents\\SCHOOL\\WPI\\CS534\\pythonProject\\Results\\Truth'
predicted_folder = 'C:\\Users\\Adam\\Documents\\SCHOOL\\WPI\\CS534\\pythonProject\\Results\\Predictions'

# Get the list of files in each folder
ground_truth_files = sorted(os.listdir(ground_truth_folder))
predicted_files = sorted(os.listdir(predicted_folder))

# Initialize lists to store results
precisions = []
recalls = []
f1_scores = []
mcc_scores = []

# Loop through each pair of files
for gt_file, pred_file in zip(ground_truth_files, predicted_files):
    gt_path = os.path.join(ground_truth_folder, gt_file)
    pred_path = os.path.join(predicted_folder, pred_file)

    # Load the CSV files
    ground_truth = pd.read_csv(gt_path)
    predicted = pd.read_csv(pred_path)

    # Flatten the labels
    y_true = ground_truth.values.flatten()
    y_pred = predicted.values.flatten()

    # Calculate Precision and Recall
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    # Calculate F1 Score
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Calculate MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    # Append the results to the lists
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    mcc_scores.append(mcc)

    # Print results for each pair (optional)
    print(f'Results for {gt_file} and {pred_file}:')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'MCC: {mcc}')
    print()

# Calculate the average of each metric
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)
avg_f1 = sum(f1_scores) / len(f1_scores)
avg_mcc = sum(mcc_scores) / len(mcc_scores)


# Print the average results
print('Average Precision:', avg_precision)
print('Average Recall:', avg_recall)
print('Average F1 Score:', avg_f1)
print('Average MCC:', avg_mcc)