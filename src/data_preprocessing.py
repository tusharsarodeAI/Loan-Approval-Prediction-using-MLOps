

import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import yaml
from sklearn.model_selection import train_test_split



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

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'processed')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise



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





def main():
    try:
        # Resolve absolute path to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Construct absolute paths from project root
        raw_data_path = os.path.join(project_root, 'data', 'raw', 'raw_data.csv')
        processed_data_path = os.path.join(project_root, 'data')

        # Load and preprocess
        df = pd.read_csv(raw_data_path)
        final_df = preprocess_data(df)

        # Train/test split
        train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=2)

        # Save
        save_data(train_data, test_data, data_path=processed_data_path)

    except Exception as e: 
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()