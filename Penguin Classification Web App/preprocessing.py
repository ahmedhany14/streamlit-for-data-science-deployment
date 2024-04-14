import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.ensemble import RandomForestClassifier

peng_data = pd.read_csv(
    "/home/ahmed/Ai/streamlit-for-data-science-deployment/Data sets/penguins_cleaned.csv"
)


def transform_data(data):
    columns = data.columns

    dummy = []
    for col in columns:
        if data[col].dtype == "object":
            dummy.append(col)

    X = pd.get_dummies(data, columns=dummy, drop_first=True)


    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype("int32")
    return X


def get_test_data(data, x_train):
    train_columns = x_train.columns
    mp = {}
    for col in train_columns:
        val = 0

        for i,j in data.items():
            if i == col:
                val = j
                break
        mp[col] = val
    features = pd.DataFrame(mp, index=[0])
    return features
    
def transform_train(data):
    data = data.map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})
    return data

