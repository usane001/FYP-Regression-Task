from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def main():

    file_path = 'updated_with_predictions.csv'



    df = pd.read_csv(file_path)


    df = df[df['language'] == 'English']

    df['original_index'] = range(len(df))

    df['label'] = df['label'].astype(float)  
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df = train_df[['text', 'label']]
    test_df = test_df[['text', 'label']]
    
    train_df.columns = ['text', 'labels']
    test_df.columns = ['text', 'labels']

    model_args = ClassificationArgs(
        num_train_epochs=3,
        regression=True,
        overwrite_output_dir=True,
        evaluate_during_training=True,
        evaluate_during_training_verbose=True
    )

    model = ClassificationModel(
        model_type='roberta',
        model_name='roberta-base',
        num_labels=1,
        args=model_args,
        use_cuda=torch.cuda.is_available() 
    )

 
    model.train_model(train_df, test_df=test_df)

 
    predictions, raw_outputs = model.predict(test_df['text'].tolist())
    test_df['predicted_label'] = predictions

 
    merged_df = df.merge(test_df[['original_index', 'predicted_label']], on='original_index', how='left')


    updated_file_path = 'updated_with_new_predictions.csv' 
    merged_df.drop(columns=['original_index']).to_csv(updated_file_path, index=False)

    print("Updated dataframe with predictions saved to:", updated_file_path)

if __name__ == '__main__':
    main()
