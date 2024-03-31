import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs

def main():

    file_path = 'english_data_only.csv'
    df = pd.read_csv(file_path)


    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    
    train_df = train_df.rename(columns={'label': 'labels'})


    model_args = ClassificationArgs(
        num_train_epochs=3,
        regression=True,
        overwrite_output_dir=True,
        output_dir='./model_output'  
    )


    model = ClassificationModel(
        "roberta",
        "roberta-base",
        num_labels=1,
        args=model_args,
        use_cuda=True,  
    )

    model.train_model(train_df[['text', 'labels']])
    

    test_predictions, _ = model.predict(test_df['text'].tolist())
    
    test_df['predicted_roberta'] = test_predictions


    test_df.to_csv('roberta_test_predictions.csv', index=False)
    print("Test set with predictions saved successfully.")


    all_predictions, _ = model.predict(df['text'].tolist())
    
    df['predicted_roberta'] = all_predictions


    df.to_csv('roberta_full_predictions.csv', index=False)
    
    print("Full dataset with predictions saved successfully.")
    print(f"Total rows in dataset: {len(df)}")
    print(f"Total predictions made: {len(all_predictions)}")

if __name__ == '__main__':
    main()
