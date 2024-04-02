import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np

file_path = 'updated_with_rmse_columns'  
data = pd.read_csv(file_path)

data.fillna(data.median(), inplace=True)  

y_true = data['label']
y_pred_chat_gpt = data['intimacy Chat GPT 4']
y_pred_roberta = data['predicted_roberta']


mse_chat_gpt = mean_squared_error(y_true, y_pred_chat_gpt)
rmse_chat_gpt = np.sqrt(mse_chat_gpt)
mse_roberta = mean_squared_error(y_true, y_pred_roberta)
rmse_roberta = np.sqrt(mse_roberta)


corr_chat_gpt, _ = pearsonr(y_true, y_pred_chat_gpt)
corr_roberta, _ = pearsonr(y_true, y_pred_roberta)


data['RMSE for Chat GPT'] = rmse_chat_gpt
data['RMSE for RoBERTa'] = rmse_roberta
data['Pearson Correlation (Label vs. Chat GPT 4)'] = corr_chat_gpt
data['Pearson Correlation (Label vs. Predicted RoBERTa)'] = corr_roberta


updated_file_path = 'updated_with_rmse_and_correlations.csv' 
data.to_csv(updated_file_path, index=False)

print(f"Updated CSV with RMSE and Pearson's correlation columns saved to: {updated_file_path}")
