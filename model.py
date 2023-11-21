"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

# Libraries for data preparation and model building
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

# Setting global constants to ensure notebook results are reproducible
PARAMETER_CONSTANT = 42

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    # predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    
    predict_vector = feature_vector_df.dropna(subset=['Valencia_pressure'])
    
    # Fill null values by the mode in df_Merge based on the length of df_test
    predict_mode = predict_vector['Valencia_pressure'].mode().values[0]
    predict_vector.iloc[len(predict_vector):, predict_vector.columns.get_loc('Valencia_pressure')].fillna(predict_mode, inplace=True)
    predict_vector['time'] = pd.to_datetime(predict_vector['time'])
    
    # Extracting date and time components from the 'time' column
    predict_vector['Day'] = predict_vector['time'].dt.day
    predict_vector['Month'] = predict_vector['time'].dt.month
    predict_vector['Year'] = predict_vector['time'].dt.year
    predict_vector['Hour'] = predict_vector['time'].dt.hour
    predict_vector['Minute'] = predict_vector['time'].dt.minute
    predict_vector['Seconds'] = predict_vector['time'].dt.second
    predict_vector['Weekend'] = predict_vector['time'].dt.weekday
    predict_vector['Week_of_year'] = predict_vector['time'].dt.isocalendar().week
    
    # Check if the 'Valencia_wind_deg' column is of string type
    if predict_vector['Valencia_wind_deg'].dtype == 'object':
    	# Remove "level_" prefix from the "Valencia_wind_deg" column and convert to int
    	predict_vector['Valencia_wind_deg'] = predict_vector['Valencia_wind_deg'].str.replace('level_', '').astype(int)
    
    # Check if the 'Seville_pressure' column is of string type
    if predict_vector['Seville_pressure'].dtype == 'object':
    	# Remove "sp" prefix from the "Seville_pressure" column and convert to float
    	predict_vector['Seville_pressure'] = predict_vector['Seville_pressure'].str.replace('sp', '').astype(int)
    
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
