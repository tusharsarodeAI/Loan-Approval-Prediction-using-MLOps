# data_ingestion.py
import os
import pandas as pd
import logging
import yaml
from sklearn.model_selection import train_test_split

# Ensure the "logs" directory exists
log_dir = '../logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def save_data_raw(df: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        save_path = os.path.join(raw_data_path, 'raw_data.csv')
        df.to_csv(save_path, index=False)

        logger.debug('Data saved to %s', save_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise




def main():
    try:
        # Always resolve paths relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_path = os.path.join(project_root, 'data')

        data_url = 'https://raw.githubusercontent.com/tusharsarodeAI/Loan-Approval-Prediction-using-MLOps/refs/heads/master/loan_approval_dataset.csv'
        df = load_data(data_url=data_url)
        save_data_raw(df, data_path=data_path)

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
