import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
import plotly.graph_objects as go

import base64

# setting the page configuration
st.set_page_config(
    page_title="ML Hyperparameter Optimization",
    page_icon="🧊",
    layout="wide",
)

# setting the title of the app
st.title("ML Hyperparameter Optimization App")
st.write("##### Regression edition")
st.write("#### Optimizing the hyperparameters of a Random Forest Regressor")

# creatin a sidebar, to take the input from the user
#    input format:
#    1. csv file
#    2. hyperparameters
#        # data ratio
#        1) percetage of spliting data
#
#        # hyperparameters
#        2) n_estimators
#        3) max_features
#        4) min_samples_split
#
#        # general hyperparameters
#        5) random_state
#        6) n_jobs
#        7) bootstrap
#        8) oob_score
#        9) criterion

# creating a sidebar for the user input
st.sidebar.title("User Input")
# 1. csv file
with st.sidebar.header("1. Upload your CSV data"):
    uploaded_file = st.sidebar.file_uploader(
        label="Upload your CSV file",
        type=["csv"],
    )
    st.sidebar.markdown(
        """
            [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
        """
    )

with st.sidebar.header("2. Set Parameters"):
    # 2.1 data split ratio
    split_size = st.sidebar.slider(
        label="Data split ratio (% for Training Set)",
        min_value=10,
        max_value=90,
        value=20,
        step=1,
    )
    # 2.2 hyperparameters
    parameter_n_estimators = st.sidebar.slider(
        "Number of estimators (n_estimators)", 0, 500, (10, 50), 50
    )
    parameter_n_estimators_step = st.sidebar.number_input(
        "Step size for n_estimators", 10
    )
    st.sidebar.write("---")

    parameter_max_features = st.sidebar.slider(
        "Max features (max_features)", 1, 50, (1, 3), 1
    )
    st.sidebar.number_input("Step size for max_features", 1)
    st.sidebar.write("---")
    parameter_min_samples_split = st.sidebar.slider(
        "Minimum number of samples required to split an internal node (min_samples_split)",
        1,
        10,
        2,
        1,
    )
    parameter_min_samples_leaf = st.sidebar.slider(
        "Minimum number of samples required to be at a leaf node (min_samples_leaf)",
        1,
        10,
        2,
        1,
    )


with st.sidebar.header("3. Model Complexity Parameters"):
    parameter_random_state = st.sidebar.number_input(
        label="Random state (random_state)",
        min_value=0,
        max_value=100,
        value=42,
        step=1,
    )
    parameter_n_jobs = st.sidebar.radio(
        label="Number of jobs to run in parallel (n_jobs)",
        options=[1, -1],
    )
    parameter_bootstrap = st.sidebar.radio(
        label="Bootstrap samples when building trees (bootstrap)",
        options=[True, False],
    )
    parameter_oob_score = st.sidebar.radio(
        label="Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)",
        options=[True, False],
    )
    parameter_criterion = st.sidebar.select_slider(
        label="Criterion (criterion)",
        options=["squared_error", "absolute_error"],
    )

    n_estimators_range = np.arange(
        parameter_n_estimators[0],
        parameter_n_estimators[1] + parameter_n_estimators_step,
        parameter_n_estimators_step,
    )
    max_features_range = np.arange(
        parameter_max_features[0], parameter_max_features[1] + 1, 1
    )
    param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)


# download dataset
def download_data(data):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV File</a>'
    return href


def model(data):
    X = data.drop(columns=["target"])
    y = data["target"]

    rfc = RandomForestRegressor(
        n_estimators=parameter_n_estimators,
        max_features=parameter_max_features,
        min_samples_split=parameter_min_samples_split,
        random_state=parameter_random_state,
        n_jobs=parameter_n_jobs,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        criterion=parameter_criterion,
    )

    # model validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size / 100, random_state=parameter_random_state
    )

    rfc_grid = GridSearchCV(rfc, param_grid=param_grid, cv=5)
    rfc_grid.fit(X_train, y_train)
    y_pred = rfc_grid.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    train_score = rfc_grid.score(X_train, y_train)
    test_score = rfc_grid.score(X_test, y_test)
    st.write("target that is being predicted")
    st.info(y.name)

    st.write("features that will be used to predict the target")
    st.info(list(X.columns))

    st.markdown("Model is being trained...")

    st.markdown("Model Performance...")

    st.write(f"Mean Squared Error:")
    st.info(mse)
    st.write(f"R2 Score:")
    st.info(r2)
    st.write(f"Train score:")
    st.info(train_score)
    st.write(f"Test score:")
    st.info(test_score)
    st.markdown("Model Parameters")
    st.info(rfc_grid.best_params_)

    # plot 3d graph for the hyperparameters
    grid_results = pd.concat(
        [
            pd.DataFrame(rfc_grid.cv_results_["params"]),
            pd.DataFrame(rfc_grid.cv_results_["mean_test_score"], columns=["R2"]),
        ],
        axis=1,
    )
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(["max_features", "n_estimators"]).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ["max_features", "n_estimators", "R2"]

    grid_pivot = grid_reset.pivot(index="max_features", columns="n_estimators")
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    # -----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="n_estimators")),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="max_features")),
    )
    fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
    fig.update_layout(
        title="Hyperparameter tuning",
        scene=dict(
            xaxis_title="n_estimators", yaxis_title="max_features", zaxis_title="R2"
        ),
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90),
    )
    st.plotly_chart(fig)

    return


data_set = pd.DataFrame(load_diabetes().data, columns=load_diabetes().feature_names)
data_set["target"] = load_diabetes().target


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())
    model(data)
else:
    st.info("Waiting for the user to upload the data...")

    if st.button("Press to use the default dataset"):
        st.write("Using the default dataset...")
        data_set
        model(data_set)
