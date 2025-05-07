# train_pipeline.py
import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import train_model, evaluate_model

# Logging setup
logger = logging.getLogger('train_pipeline')
logger.setLevel(logging.DEBUG)

log_dir = '../logs'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'train_pipeline.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logger.debug("Data loaded from %s and %s", train_path, test_path)
    return train_df, test_df

def split_and_scale(train_df, test_df, target_column='loan_status'):
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.debug("Features scaled.")
    return X_train_scaled, X_test_scaled, y_train, y_test

def main():
    try:
        train_df, test_df = load_data('../data/raw/train.csv', '../data/raw/test.csv')
        X_train, X_test, y_train, y_test = split_and_scale(train_df, test_df)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
