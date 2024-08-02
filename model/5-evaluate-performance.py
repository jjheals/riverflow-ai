import os
import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef


# ---- Config ---- # 
with open('config/config.json', 'r') as file:
    config:dict = json.load(file)

output_dir:str = config['output-dir']


# ---- Loading data ----- # 
# Get all of the truth and prediction files 
ground_truth_files:list[str] = []
predicted_files:list[str] = []

for run_dir in os.listdir(output_dir): 
    # We only care about the 'csvs' subdir 
    this_csv_dir_path:str = os.path.join(output_dir, run_dir, 'csvs')
    
    for csv_file in os.listdir(this_csv_dir_path):
        if 'truth' in csv_file: ground_truth_files.append(os.path.join(this_csv_dir_path, csv_file))
        elif 'predicted' in csv_file: predicted_files.append(os.path.join(this_csv_dir_path, csv_file))

ground_truth_files = sorted(ground_truth_files)
predicted_files = sorted(predicted_files)

# Initialize lists to store results
precisions = []
recalls = []
f1_scores = []
mcc_scores = []

# Loop through each pair of files
for gt_file_path, pred_file_path in zip(ground_truth_files, predicted_files):
    
    ground_truth = pd.read_csv(gt_file_path)
    predicted = pd.read_csv(pred_file_path)

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
    print(f'\033[90mResults for \033[92m{os.path.basename(gt_file_path)} \033[90mand \033[92m{os.path.basename(pred_file_path)}\033[90m:')
    print(f'\033[92mPrecision: \033[90m{precision}')
    print(f'\033[92mRecall: \033[90m{recall}')
    print(f'\033[92mF1 Score: \033[90m{f1}')
    print(f'\033[92mMCC: \033[90m{mcc}')
    print()

# Calculate the average of each metric
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)
avg_f1 = sum(f1_scores) / len(f1_scores)
avg_mcc = sum(mcc_scores) / len(mcc_scores)


# Print the average results
print('\033[93m--- Average Results ---\033[90m')
print('\033[92mAverage Precision:\033[90m', avg_precision)
print('\033[92mAverage Recall:\033[90m', avg_recall)
print('\033[92mAverage F1 Score:\033[90m', avg_f1)
print('\033[92mAverage MCC:\033[90m', avg_mcc)
print('\033[0m')