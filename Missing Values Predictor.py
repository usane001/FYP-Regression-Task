import pandas as pd

eval_predictions = '.csv'  
test_df = pd.read_csv(eval_predictions)


nan_predictions = test_df['predicted_label'].isnull().sum()
total_predictions = len(test_df)

nan_percentage = nan_predictions / total_predictions * 100

print(f"Total Predictions: {total_predictions}")
print(f"NaN Predictions: {nan_predictions}")
print(f"Percentage of NaN Predictions: {nan_percentage:.2f}%")
