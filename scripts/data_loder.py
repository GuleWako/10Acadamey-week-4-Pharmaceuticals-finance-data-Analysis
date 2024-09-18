import pandas as pd

def load_data(train_data_path: str, test_data_path: str, store_data_path: str):
    train_data = pd.read_csv(train_data_path)
    test_data  = pd.read_csv(test_data_path)
    store_data = pd.read_csv(store_data_path)
    return train_data, test_data, store_data
