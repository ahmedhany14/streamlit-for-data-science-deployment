import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


column_names = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]

boston = pd.read_csv(
    "/home/ahmed/Ai/streamlit-for-data-science-deployment/Data sets/housing.csv",
    header=None,
    delimiter=r"\s+",
    names=column_names,
)
# setting the Page layout
st.set_page_config(page_title="The Machine Learning App", layout="wide", page_icon="🧊")


# building a model
def model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    y = df.iloc[:,-1] # Selecting the last column as Y

    st.markdown("**1.0 The Data Splitting**")
    st.info(f"X shape: {X.shape}")
    st.info(f"y shape: {y.shape}")

    st.markdown("**Variables Splitting**")
    st.info(f"X: { list(X.columns) }")
    st.info(f"y: MEDV")

    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split_size)

    rf = RandomForestRegressor(
        n_estimators=params_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs,
    )

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    st.markdown("Model Performance")
    st.info(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    st.info(f"R2 Score: {r2_score(y_test, y_pred)}")
    st.markdown('Model score: {}'.format(rf.score(x_test, y_test)))

    st.markdown("Model Parameters")
    st.info(rf.get_params())
    st.markdown("Feature Importance")
    st.write(
        pd.DataFrame(
            rf.feature_importances_, index=X.columns, columns=["Importance"]
        ).sort_values("Importance", ascending=False)
    )


st.write(
    """
    # Machine Learning App
    In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.

    Try adjusting the hyperparameters!
    """
)


# creating sidebar

with st.sidebar.header("1. Upload your CSV data"):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown(
        """
            [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
        """
    )

with st.sidebar.header("2. Set Parameters"):
    split_size = st.sidebar.slider(
        "Data split ratio (% for Training Set)", 10, 90, 80, 5
    )

    st.sidebar.subheader("2.1 Learning Parameters")
    params_n_estimators = st.sidebar.number_input(
        "Number of estimators", 100, 1000, 100, 100
    )
    parameter_random_state = st.sidebar.number_input("Random state", 0, 100, 42, 1)
    parameter_criterion = st.sidebar.select_slider(
        "Criterion",
        options=["squared_error", "absolute_error"],
    )

    st.sidebar.subheader("2.2 Complexity Parameters")
    parameter_min_samples_split = st.sidebar.number_input(
        "Min samples split", 1, 10, 2, 1
    )
    parameter_min_samples_leaf = st.sidebar.number_input(
        "Min samples leaf", 1, 10, 1, 1
    )
    parameter_max_features = st.sidebar.number_input(
        "Max features", 1, len(boston.columns), 1, 1
    )


    st.sidebar.subheader("2.3 Bootstrapping")
    parameter_bootstrap = st.sidebar.radio(
        "Bootstrap samples when building trees", [True, False]
    )
    parameter_oob_score = st.sidebar.radio(
        "Whether to use out-of-bag samples to estimate the R^2 on unseen data",
        [True, False],
    )
    parameter_n_jobs = st.sidebar.radio(
        "The number of jobs to run in parallel", [1, -1]
    )


st.write('Data set')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    model(df)

else:
    st.info("Awaiting for CSV file to be uploaded.")
    if st.toggle("Use Example Dataset"):
        st.write("Using example dataset")
        df = boston
        st.write(df)
        model(df)