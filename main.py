import logging
import os
from src.data_ingestion import ZipDataIngestor, CSVDataIngestor
from src.data_preprocessing import DataPreprocessing
from src.data_splitting import split_data
from src.models_training import get_model_trainer, save_model
from src.models_evaluation import evaluate_model

def main():
    # Ingest the data
    logging.info('[data ingestion step]')
    data_ingestor = ZipDataIngestor()
    csv_ingestor = CSVDataIngestor()
    youtube = data_ingestor.ingest(r'.\data\youtube.zip')
    reddit = csv_ingestor.ingest('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
    reddit.to_csv(r'.\extracted_data\reddit.csv', index=False)
    # Preprocess the data
    logging.info('[data cleaning step]')
    preprocessor = DataPreprocessing()
    df = preprocessor.clean_data(reddit, youtube)
    # Split the data
    logging.info('[data splitting step]')
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    # Vectorize the data
    logging.info('[data vectorizing step]')
    X_train_vectorized, X_test_vectorized = preprocessor.vectorize_data(X_train, X_test)
    # Train the model
    logging.info('[model training step]')
    model_name = 'lightgbm'  # Options: 'random_forest', 'lightgbm'
    trained_model = get_model_trainer(model_name, X_train_vectorized, y_train, X_test_vectorized, y_test)
    # Save the model
    logging.info('[model saving step]')
    file_path = os.path.join(r'.\artifacts', f'{model_name}_model.pkl')
    save_model(trained_model, file_path)
    # Evaluate the model
    logging.info('[model evaluation step]')
    metrics = evaluate_model(trained_model, X_test_vectorized, y_test)
    
    return trained_model

if __name__ == '__main__':
    main()