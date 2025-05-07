

import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Ensure the "logs" directory exists
log_dir = '../logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def clean_data(st):
    """Clean the input string (e.g., strip whitespace)."""
    st = st.strip()
    return st

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data with machine learning encoding techniques."""
    try:
        logger.debug("Starting data preprocessing.")

        # Drop the loan_id column
        df.drop(columns=['loan_id'], inplace=True)
        logger.debug("'loan_id' column dropped.")

        # Clean column names
        df.columns = df.columns.str.strip()
        logger.debug("Column names stripped of whitespace.")

        # Create 'Assets' column
        df['Assets'] = df.residential_assets_value + df.commercial_assets_value + df.luxury_assets_value + df.bank_asset_value
        logger.debug("'Assets' column created.")

        # Drop original asset columns
        df.drop(columns=['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'], inplace=True)
        logger.debug("Original asset columns dropped.")

        # Check for missing values
        missing = df.isnull().sum().sum()
        logger.debug("Total missing values in dataset: %d", missing)

        # Initialize LabelEncoder
        le = LabelEncoder()

        # Encode 'education'
        df['education'] = df['education'].apply(clean_data)
        df['education'] = le.fit_transform(df['education'])
        logger.debug("'education' column encoded.")

        # Encode 'self_employed'
        df['self_employed'] = df['self_employed'].apply(clean_data)
        df['self_employed'] = le.fit_transform(df['self_employed'])
        logger.debug("'self_employed' column encoded.")

        # Encode 'loan_status'
        df['loan_status'] = df['loan_status'].apply(clean_data)
        df['loan_status'] = le.fit_transform(df['loan_status'])
        logger.debug("'loan_status' column encoded.")

        logger.debug("Data preprocessing completed successfully.")
        return df

    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise