
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from datetime import datetime

# Model Building
def build_ml_model():
    """
    Build a Random Forest Regressor with 100 estimators and random_state for reproducibility.
    """
    model = RandomForestRegressor(n_estimators=50, max_depth=2, n_jobs=-1)
    return model

# Model Training with Sklearn Pipeline
def train_ml_model(train_data):
    train_data_model = train_data
    train_data_no_sales = train_data_model.drop('Sales',axis=1)
    X_train = train_data_no_sales
    y_train = train_data.pop('Sales')

    
    # Defining the pipeline
    pipeline = Pipeline([
        ('model', build_ml_model())
    ])
    
    # Fitting the model
    pipeline.fit(X_train, y_train)
    
    # Making predictions on validation data
    predictions = pipeline.predict(X_train)
    
    # Calculating evaluation metrics
    rmse = mean_squared_error(y_train, predictions)
    mae = mean_absolute_error(y_train, predictions)
    
    print(f"Validation RMSE: {rmse}")
    print(f"Validation MAE: {mae}")
   
    
    return pipeline,X_train,y_train



def post_prediction_feature_importance(model, train_data):
    """
    Analyze feature importance and estimate confidence intervals for predictions.
    """
    # Extracting feature importance from the Random Forest model
    model_rf = model.named_steps['model']

    feature_importances = model_rf.feature_importances_

    # Sorting feature importances for visualization
    sorted_idx = np.argsort(feature_importances)
    plt.figure(figsize=(10, 8))
    plt.barh(np.array(train_data.columns)[sorted_idx], feature_importances[sorted_idx], color='g')
    plt.xlabel("Random Forest Feature Importance")
    plt.title("Feature Importance in Sales Prediction")
    plt.show()
    
def post_prediction_confidence_interval(model, train_data):
    # Confidence interval estimation
    # For simplicity, we can look at the variance across trees in the forest

    model_rf = model.named_steps['model']

    predictions_per_tree = np.array([tree.predict(train_data)
                                        for tree in model_rf.estimators_])

    # Variance of predictions across trees
    prediction_variance = np.var(predictions_per_tree, axis=0)

    # Estimate the 95% confidence interval using variance
    lower_bound = model.predict(train_data) - 1.9 * np.sqrt(prediction_variance)
    upper_bound = model.predict(train_data) + 1.9 * np.sqrt(prediction_variance)

    lower_upper_bound=pd.DataFrame({"Lower Bound":lower_bound, "Upper Bound":upper_bound})

    return lower_upper_bound



def save_model(model, path):
    """
    Save the trained model to a file with a timestamp.
    :param model: The model to save.
    :param path: Base path where the model will be saved.
    """
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_path = f"{path}/model-{timestamp}.pkl"
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")