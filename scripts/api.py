import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import uvicorn
from glob import glob
import numpy as np
# Initialize logging
logging.basicConfig(level=logging.INFO)

def get_latest_model(model_dir):
    """
    Get the path of the most recently saved model in the specified directory.
    :param model_dir: Directory where the models are stored.
    :return: Path to the most recent model file.
    """
    # Find all .pkl files in the directory
    model_files = glob(f"{model_dir}/model-*.pkl")
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in directory: {model_dir}")
    
    # Sort the files by modification time (newest first)
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

# Directory where models are saved
model_directory = '../models'

# Get the latest model file path
latest_model_path = get_latest_model(model_directory)

# Load the latest model using joblib
logging.info(f"Loading model from {latest_model_path}")
model = joblib.load(latest_model_path)

# Create a FastAPI instance
app = FastAPI()

# Define input data schema using Pydantic
class InputData(BaseModel):		
    Store: float
    DayOfWeek: float
    Open: float
    Promo: float
    StateHoliday_0: float
    StateHoliday_a: float
    StateHoliday_b: float
    StateHoliday_c: float
    SchoolHoliday: float
    StoreType_a: float
    StoreType_b: float
    StoreType_c: float
    StoreType_d: float
    Assortment_b: float
    Assortment_c: float
    Assortment_a: float
    CompetitionDistance: float
    CompetitionOpenSinceMonth: float
    CompetitionOpenSinceYear: float
    Promo2: float
    Promo2SinceWeek: float
    Promo2SinceYear: float
    PromoInterval_Feb_May_Aug_Nov: float
    PromoInterval_Jan_Apr_Jul_Oct: float
    PromoInterval_Mar_Jun_Sept_Dec: float
    Year: float
    Month: float
    Day: float
    WeekOfYear: float
    IsWeekend: float
    MonthPeriod_middle: float
    MonthPeriod_end: float
    MonthPeriod_beginning: float
    DaysTo_A_Holiday: float
    DaysTo_B_Holiday: float
    DaysTo_C_Holiday: float
    DaysAfter_A_Holiday: float
    DaysAfter_B_Holiday: float
    DaysAfter_C_Holiday: float

    

#  endpoint
@app.post('/predict')
async def predict(input_data: InputData):
    try:
        # Convert input data to DataFrame
        input_df = np.array([[input_data.Store, input_data.DayOfWeek,input_data.DaysTo_A_Holiday,input_data.SchoolHoliday,input_data.DaysAfter_A_Holiday,input_data.DaysTo_B_Holiday,input_data.DaysAfter_B_Holiday,input_data.DaysTo_C_Holiday,
                                  input_data.DaysAfter_C_Holiday,input_data.Open,input_data.Promo,input_data.Promo2,input_data.Promo2SinceWeek, input_data.Promo2SinceYear,input_data.Year, input_data.Month,input_data.Day,input_data.WeekOfYear,
                                  input_data.IsWeekend,input_data.CompetitionDistance,input_data.CompetitionOpenSinceMonth, input_data.CompetitionOpenSinceYear, input_data.StoreType_a,input_data.StoreType_b,input_data.StoreType_c,input_data.StoreType_d,
                                  input_data.Assortment_a,input_data.Assortment_b,input_data.Assortment_c,input_data.StateHoliday_0,input_data.StateHoliday_a,input_data.StateHoliday_b,input_data.StateHoliday_c,
                                  input_data.PromoInterval_Feb_May_Aug_Nov,input_data.PromoInterval_Jan_Apr_Jul_Oct,input_data.PromoInterval_Mar_Jun_Sept_Dec,
                                  input_data.MonthPeriod_beginning,input_data.MonthPeriod_end,input_data.MonthPeriod_middle]], 
                                 )
        
        # Make predictions
        predictions = model.predict(input_df)
        predicted_values = predictions.tolist() 

        # Return the predictions as a JSON response
        return {'predictions': predicted_values}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)





# columns=['Store', 'DayOfWeek', 'DaysTo_A_Holiday', 'SchoolHoliday',
#        'DaysAfter_A_Holiday', 'DaysTo_B_Holiday', 'DaysAfter_B_Holiday',
#        'DaysTo_C_Holiday', 'DaysAfter_C_Holiday', 'Open', 'Promo', 'Promo2',
#        'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 'Month', 'Day',
#        'WeekOfYear', 'IsWeekend', 'CompetitionDistance',
#        'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'StoreType_a',
#        'StoreType_b', 'StoreType_c', 'StoreType_d', 'Assortment_a',
#        'Assortment_b', 'Assortment_c', 'StateHoliday_0', 'StateHoliday_a',
#        'StateHoliday_b', 'StateHoliday_c', 'PromoInterval_Feb_May_Aug_Nov',
#        'PromoInterval_Jan_Apr_Jul_Oct', 'PromoInterval_Mar_Jun_Sept_Dec',
#        'MonthPeriod_beginning', 'MonthPeriod_end', 'MonthPeriod_middle']