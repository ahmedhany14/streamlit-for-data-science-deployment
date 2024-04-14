import pandas as pd
import streamlit as st
import numpy as np

import preprocessing as pp
import ML_model as ml

st.write(
    '''
        # simple Penguin Classification Web App  
    
        This app predicts the species type!
    '''   
)


peng_data = pd.read_csv(
    "/home/ahmed/Ai/streamlit-for-data-science-deployment/Data sets/penguins_cleaned.csv"
)
x_train = peng_data.drop("species", axis=1)
y_train = peng_data["species"]


x_processed = pp.transform_data(x_train)
y_processed = pp.transform_train(y_train)



st.sidebar.header("input the penguin's features")
def user_input():
    bill_length_mm = st.sidebar.slider('bill_length_mm', 32.1, 59.6, 43.9)
    bill_depth_mm = st.sidebar.slider('bill_depth_mm', 13.1, 21.5, 17.2)
    flipper_length_mm = st.sidebar.slider('flipper_length_mm', 172.0, 231.0, 201.0)
    body_mass_g = st.sidebar.slider('body_mass_g', 2700.0, 6300.0, 4207.0)
    island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    sex = st.sidebar.selectbox('Sex',('male', 'female'))
    data = {'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'island': island,
            'sex' : sex
    }
    features = pd.DataFrame(data, index=[0])
    return data, features

data_input, features = user_input()
X_test = pp.get_test_data(data_input, x_processed)

st.subheader('User Input parameters')
st.write(features)


st.subheader('Class labels and their corresponding index number')
st.write(peng_data['species'].unique())

model = ml.model(x_processed, y_processed)
predition, converted_predition = ml.predict_data(X_test, model)

st.write('Prediction')
st.write(predition)

st.write('converted predition')
st.write(converted_predition)
