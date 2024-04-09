# Building a Machine Learning Model with Streamlit

**Overview**

This Streamlit app provides an interactive end-to-end workflow for building a machine learning (ML) model with a specific focus on Random Forests. It empowers users to experiment with different parameters and instantly see how they affect model performance.

**Features**

* **Data Upload:**  Upload your own CSV dataset or use the provided drug solubility example.
* **Parameter Customization:** Tune the Random Forest model using various slider widgets in the sidebar.
* **Real-time Model Building:** See model training in progress and instantly get performance metrics when adjustments are made.
* **Visualizations:** Understand model results with feature importance bar charts and actual vs. predicted scatter plots.
* **Model & Data Download:** Download your trained model as a ZIP file containing the necessary data splits.

**Libraries**

* **Streamlit:**  Creates the dynamic web-based user interface.
* **Pandas:**  Data manipulation and loading the CSV file.
* **Scikit-learn:** Building the Random Forest regressor model.
* **Altair:**  Generating charts for visualizations.

**Code Breakdown**

**1. Imports, Page Setup, and Explainers**

* Imports necessary libraries
* Uses `st.set_page_config` to set the app's title and icon.
* Expanders (`st.expander`) provide clear explanations of the app's purpose and functionality.

**2. Sidebar: User Input**

* **Data Loading**
   * Allows file upload (`st.file_uploader`) or use of an example dataset.
   * Includes download button for the example CSV.
* **Model Parameters**
   * Sliders (`st.slider`) dynamically control:
     *  Data split ratio
     *  Random Forest hyperparameters (number of trees, max features, etc.)

**3.  Model Building Workflow**

* **Status Display:**  A `st.status` widget provides feedback ("Running...", "Complete") during model building.
* **Data Loading and Preparation**
   * `pd.read_csv` loads the data.
   * Data is split into training and testing sets.
* **Model Training**
   *  A Random Forest Regressor (`rf`)  is created with parameters from sidebar inputs.
   * `rf.fit` trains the model on the training data.
* **Prediction and Evaluation**
   * Makes predictions on training and testing sets.
   * Calculates performance metrics (MSE, R2).

**4. Results Display**

* **Tabs (Optional):** Consider using `st.tabs` to organize results.
* **Data Info:** Displays the number of samples, features, and split sizes.
* **Model Parameters:** Shows chosen parameters.
* **Performance Metrics:** Shows MSE and R2 in a clear table format.
* **Feature Importance:** Bar chart displays feature importance. 
* **Prediction Results:**
   * Table shows actual vs. predicted values.
   * Scatter plot visualizes actual vs. predicted values.

**5. Download Links**

* Creates a ZIP file containing the dataset and splits.
* Provides a `st.download_button` for download.

**How to Run the App**

1. Install libraries: `pip install streamlit pandas scikit-learn altair`
2. Save the code as a Python file (e.g., `app.py`) 
3. From your terminal, run: `streamlit run app.py`
