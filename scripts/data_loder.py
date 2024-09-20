import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_data_path: str, test_data_path: str, store_data_path: str):
    logging.info("Loading data from file...")
    train_data = pd.read_csv(train_data_path)
    test_data  = pd.read_csv(test_data_path)
    store_data = pd.read_csv(store_data_path)
    logging.info(f"Train Data, Test Data and Store Data loaded with shape {train_data.shape}, {test_data.shape}, and {store_data.shape} respectively")
    return train_data, test_data, store_data

