from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
from sklearn.model_selection import train_test_split


def main():

    file_path = 'updated_with_predictions.csv'
    

    df = pd.read_csv(file_path)
 

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df = train_df.rename(columns={'label': 'labels'})

    model_args = ClassificationArgs(
        num_train_epochs=3,
        regression=True,
        overwrite_output_dir=True,
        output_dir='./model_output'
    )

    model = ClassificationModel(
        'roberta',
        'roberta-base',
        num_labels=1,
        args=model_args,
        use_cuda=True,
    )

 
    model.train_model(train_df[['text', 'labels']])

 
    predictions, raw_outputs = model.predict(test_df['text'].tolist())
    
    assert len(predictions) == len(test_df), "Mismatch between predictions and rows in test_df"
    
    test_df['predicted_roberta'] = predictions
    
    test_df.to_csv('roberta_predictions.csv', index=False)
    print("Test set with predictions saved successfully.")
    print(f"Total rows in test set: {len(test_df)}")
    print(f"Total predictions made: {len(predictions)}")

   

if __name__ == '__main__':
    main()
