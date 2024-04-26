import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
import pickle

URL = "https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv"
data_set = pd.read_csv(URL)
# print(data_set)

X = data_set.drop(columns=["logS"])
Y = data_set["logS"]


class Models:

    def linear_regression(self, X, Y):
        scaler = StandardScaler()
        regression = LinearRegression()
        model = Pipeline([("scaler", scaler), ("regression", regression)])
        model.fit(X, Y)
        return model

    def support_vector_regression(self, X, Y):
        scaler = StandardScaler()
        svr_model = SVR()
        model = Pipeline([("scaler", scaler), ("SVR", svr_model)])
        model.fit(X, Y)
        return model

    def full_model(self, reg, svr, X, Y):
        model = VotingRegressor(estimators=[("reg", reg), ("svr", svr)])
        model.fit(X, Y)
        return model

ML = Models()
linear_model = ML.linear_regression(X, Y)
svr_model = ML.support_vector_regression(X, Y)

final_model = ML.full_model(linear_model, svr_model, X, Y)


pickle.dump(final_model, open("solubility_model.pkl", "wb"))