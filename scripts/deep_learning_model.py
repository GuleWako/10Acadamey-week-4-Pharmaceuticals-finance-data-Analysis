import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Function to build the LSTM model
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)  # Predicting one value (sales)
    ])
    model.compile(optimizer='adam', loss='mse')  # Using Mean Squared Error loss
    return model

# Function to prepare data for LSTM
def prepare_lstm_data(data, target_col, window_size=60):
    data_lstm=data
    """
    Prepare the data for LSTM by creating sequences.
    :param data: DataFrame containing the features.
    :param target_col: Column name of the target variable ('Sales').
    :param window_size: Number of past days to use for predicting future sales.
    :return: Scaled X (features), y (target), and the scaler object for inverse transformation.
    """
    # Scale the target column using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data_lstm[target_col].values.reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])  # Previous 'window_size' days
        y.append(scaled_data[i, 0])  # Next day's sales

    # Convert to numpy arrays
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape to LSTM expected input

    return X, y, scaler


# Training the LSTM model
def train_lstm_model(X_train, y_train, input_shape):
    """
    Train the LSTM model using training data and validate on the validation set.
    :param X_train: Features for training.
    :param y_train: Target for training.
    :param X_val: Features for validation.
    :param y_val: Target for validation.
    :param input_shape: Shape of the input data for the LSTM model.
    :return: Trained LSTM model.
    """
    model = build_lstm_model(input_shape)
    # Train the model
    model.fit(X_train, y_train, epochs=2, batch_size=64)
    return model