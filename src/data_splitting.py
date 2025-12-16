import logging
from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(
    data: pd.DataFrame,
    test_size: float = 0.2):
    try:
        X = data['comment'].values
        y = data['category'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logging.info("Data Split completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f'Error in Data Split: {e}')
        raise e