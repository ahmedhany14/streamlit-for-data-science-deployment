import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.ensemble import RandomForestClassifier

import preprocessing as pp


def model(x_train, y_train):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model


def predict(model, data):
    return model.predict(data)


def convert_pred(pred):
    if pred == 0:
        return "Adelie"
    elif pred == 1:
        return "Gentoo"
    else:
        return "Chinstrap"


def predict_data(data, model):
    data = pp.transform_data(data)
    pred = predict(model, data)
    pred_proba = model.predict_proba(data)
    return pred, convert_pred(pred), pred_proba