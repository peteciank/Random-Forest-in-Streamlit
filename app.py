# Import necessary libraries
import streamlit as st  # Import Streamlit library for creating web application
import pandas as pd  # Import Pandas library for data manipulation
import numpy as np  # Import NumPy library for numerical operations
from sklearn.model_selection import train_test_split  # Import train_test_split function for splitting data
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestRegressor for building regression model
from sklearn.metrics import mean_squared_error, r2_score  # Import mean_squared_error and r2_score for model evaluation
import altair as alt  # Import Altair library for data visualization
import time  # Import time library for adding delays
import zipfile  # Import zipfile library for creating zip files

# Set page configuration and title
st.set_page_config(page_title='Random Forest Model Testing', page_icon='🤖')  # Set page title and icon
st.title('Random Forest Model Testing using Streamlit')  # Display title on the web app


# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('1.1. Input data')

    st.markdown('**1. Use custom data**')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])  # Widget to upload a CSV file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)  # Read the uploaded CSV file
    
    # Download example data
    @st.cache_data
    def convert_df(input_df):
        return input_df.to_csv(index=False).encode('utf-8')

    example_csv = pd.read_csv('https://raw.githubusercontent.com/peteciank/public_files/main/Datasets/diabetes.csv')
    csv = convert_df(example_csv)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='diabetes.csv',
        mime='text/csv',
    )

    # Select example data
    st.markdown('**1.2. Use Sample Data**')
    example_data = st.toggle('Load Sample data')
    if example_data:
        df = pd.read_csv('https://raw.githubusercontent.com/peteciank/public_files/main/Datasets/diabetes.csv')

    st.header('2. Set Parameters')
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)  # Slider widget for selecting data split ratio

    st.subheader('2.1. Set Learning Parameters')
    with st.expander('See parameters'):
        parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)  # Slider widget for selecting number of estimators
        parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])  # Dropdown widget for selecting max features
        parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)  # Slider widget for selecting minimum samples split
        parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)  # Slider widget for selecting minimum samples leaf

    st.subheader('2.2. Set General Parameters')
    with st.expander('See parameters', expanded=False):
        parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)  # Slider widget for selecting random state
        parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse'])  # Dropdown widget for selecting performance measure
        parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])  # Dropdown widget for selecting bootstrap
        parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])  # Dropdown widget for selecting oob score

    sleep_time = st.slider('Sleep time', 0, 3, 0)  # Slider widget for selecting sleep time

# Initiate the model building process
if uploaded_file or example_data:
    with st.status("Running ...", expanded=True) as status:
        st.write("Loading data ...")
        time.sleep(sleep_time)  # Adding delay to simulate loading time

        st.write("Preparing data ...")
        time.sleep(sleep_time)  # Adding delay to simulate data preparation time
        X = df.iloc[:,:-1]  # Extracting features
        y = df.iloc[:,-1]   # Extracting target variable

        st.write("Splitting data ...")
        time.sleep(sleep_time)  # Adding delay to simulate data splitting time
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)  # Splitting data into training and testing sets

        st.write("Model training ...")
        time.sleep(sleep_time)  # Adding delay to simulate model training time

        if parameter_max_features == 'all':
            parameter_max_features = None
            parameter_max_features_metric = X.shape[1]  # Set max features metric if 'all' is selected

        rf = RandomForestRegressor(
            n_estimators=parameter_n_estimators,
            max_features=parameter_max_features,
            min_samples_split=parameter_min_samples_split,
            min_samples_leaf=parameter_min_samples_leaf,
            random_state=parameter_random_state,
            criterion=parameter_criterion,
            bootstrap=parameter_bootstrap,
            oob_score=parameter_oob_score)  # Initialize Random Forest Regressor model with selected parameters
        rf.fit(X_train, y_train)  # Train the model

        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)  # Adding delay to simulate prediction time
        y_train_pred = rf.predict(X_train)  # Predict on training data
        y_test_pred = rf.predict(X_test)    # Predict on testing data

        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)  # Adding delay to simulate evaluation time
        train_mse = mean_squared_error(y_train, y_train_pred)  # Calculate training Mean Squared Error
        train_r2 = r2_score(y_train, y_train_pred)            # Calculate training R-squared
        test_mse = mean_squared_error(y_test, y_test_pred)    # Calculate testing Mean Squared Error
        test_r2 = r2_score(y_test, y_test_pred)              # Calculate testing R-squared

        st.write("Displaying performance metrics ...")
        time.sleep(sleep_time)  # Adding delay to simulate display time
        parameter_criterion_string = ' '.join([x.capitalize() for x in parameter_criterion.split('_')])  # Format criterion string
        rf_results = pd.DataFrame(['Random forest', train_mse, train_r2, test_mse, test_r2]).transpose()  # Create DataFrame for results
        rf_results.columns = ['Method', f'Training {parameter_criterion_string}', 'Training R2', f'Test {parameter_criterion_string}', 'Test R2']  # Set column names
        for col in rf_results.columns:
            rf_results[col] = pd.to_numeric(rf_results[col], errors='ignore')  # Convert objects to numerics
        rf_results = rf_results.round(3)  # Round to 3 digits

    status.update(label="Status", state="complete", expanded=False)  # Update status to complete

    # Display data info
    st.header('Input data', divider='rainbow')
    col = st.columns(4)
    col[0].metric(label="No. of samples", value=X.shape[0], delta="")
    col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
    col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
    col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")

    # Display initial dataset, train split, and test split
    with st.expander('Initial dataset', expanded=True):
        st.dataframe(df, height=210, use_container_width=True)
    with st.expander('Train split', expanded=False):
        train_col = st.columns((3,1))
        with train_col[0]:
            st.markdown('**X**')
            st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
        with train_col[1]:
            st.markdown('**y**')
            st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
    with st.expander('Test split', expanded=False):
        test_col = st.columns((3,1))
        with test_col[0]:
            st.markdown('**X**')
            st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
        with test_col[1]:
            st.markdown('**y**')
            st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

    # Zip dataset files for download
    df.to_csv('dataset.csv', index=False)
    X_train.to_csv('X_train.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    list_files = ['dataset.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
    with zipfile.ZipFile('dataset.zip', 'w') as zipF:
        for file in list_files:
            zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

    with open('dataset.zip', 'rb') as datazip:
        btn = st.download_button(
            label='Download ZIP',
            data=datazip,
            file_name="dataset.zip",
            mime="application/octet-stream"
            )

    # Display model parameters
    st.header('Model parameters', divider='rainbow')
    parameters_col = st.columns(3)
    parameters_col[0].metric(label="Data split ratio (% for Training Set)", value=parameter_split_size, delta="")
    parameters_col[1].metric(label="Number of estimators (n_estimators)", value=parameter_n_estimators, delta="")
    parameters_col[2].metric(label="Max features (max_features)", value=parameter_max_features_metric, delta="")

    # Display feature importance plot
    importances = rf.feature_importances_
    feature_names = list(X.columns)
    forest_importances = pd.Series(importances, index=feature_names)
    df_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'value'})

    bars = alt.Chart(df_importance).mark_bar(size=40).encode(
       x='value:Q',
       y=alt.Y('feature:N', sort='-x')
      ).properties(height=250)

    performance_col = st.columns((2, 0.2, 3))
    with performance_col[0]:
        st.header('Model performance', divider='rainbow')
        st.dataframe(rf_results.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'}))
    with performance_col[2]:
        st.header('Feature importance', divider='rainbow')
        st.altair_chart(bars, theme='streamlit', use_container_width=True)

    # Prediction results
    st.header('Prediction results', divider='rainbow')
    s_y_train = pd.Series(y_train, name='actual').reset_index(drop=True)
    s_y_train_pred = pd.Series(y_train_pred, name='predicted').reset_index(drop=True)
    df_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
    df_train['class'] = 'train'

    s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
    s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
    df_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
    df_test['class'] = 'test'

    df_prediction = pd.concat([df_train, df_test], axis=0)

    prediction_col = st.columns((2, 0.2, 3))

    # Display dataframe
    with prediction_col[0]:
        st.dataframe(df_prediction, height=320, use_container_width=True)

    # Display scatter plot of actual vs predicted values
    with prediction_col[2]:
        scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
            x='actual',
            y='predicted',
            color='class'
         )
        st.altair_chart(scatter, theme='streamlit', use_container_width=True)

# Prompt user to upload CSV if none is detected
else:
    st.warning('👈 Upload a CSV file or click *"Load example data"* to get started!')
