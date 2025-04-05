# Personal Fitness Tracker 🏃‍♂️

This project is a web application built with Streamlit that predicts the number of calories burned during exercise based on user-provided information. It uses a machine learning model (Random Forest Regressor) trained on exercise and calorie data.

## Description

The Personal Fitness Tracker allows users to input their personal metrics such as age, gender, Body Mass Index (BMI), exercise duration, heart rate, and body temperature. Based on these inputs, the application predicts the estimated kilocalories burned. It also provides context by showing similar results from the dataset and comparing the user's metrics (age, duration, heart rate, body temp) against the dataset population using percentiles.

## Features

- **User Input:** Interactive sliders and radio buttons for entering Age, BMI, Duration, Heart Rate, Body Temperature, and Gender.
- **Calorie Prediction:** Displays the predicted kilocalories burned based on the input parameters using a trained Random Forest model.
- **Similar Results:** Shows anonymized data entries from the dataset with similar calorie burn values.
- **Percentile Comparison:** Calculates and displays the user's percentile rank for Age, Duration, Heart Rate, and Body Temperature compared to the dataset.
- **Data Caching:** Uses Streamlit's caching for efficient data loading and model training.

## How It Works

1.  **Data Loading:** The application loads data from `exercise.csv` and `calories.csv`, merging them based on `User_ID`.
2.  **Data Preprocessing:** Calculates BMI and prepares the data for model training (one-hot encoding for gender, splitting into train/test sets).
3.  **Model Training:** A Random Forest Regressor model is trained on the preprocessed data (`X_train`, `y_train`).
4.  **User Input:** The sidebar collects user parameters.
5.  **Prediction:** The trained model predicts calories burned based on the user's input.
6.  **Display:** Results, including the prediction, similar entries, and percentile comparisons, are displayed in the main panel.

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    Make sure you have a `requirements.txt` file (you might need to create one based on the imports in `app.py`). A possible `requirements.txt` would include:
    ```
    streamlit
    numpy
    pandas
    scikit-learn
    matplotlib
    seaborn
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ensure data files are present:**
    Place `calories.csv` and `exercise.csv` in the same directory as `app.py`.

## Usage

To run the application, navigate to the project directory in your terminal and run:

```bash
streamlit run app.py
```

This will open the application in your default web browser. Use the sidebar to input your parameters and view the predicted calories burned and other insights on the main page.

## Dependencies

- Python (3.7+)
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Data

The application relies on two CSV files:

- `calories.csv`: Contains `User_ID` and corresponding `Calories` burned.
- `exercise.csv`: Contains `User_ID`, `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, and `Body_Temp`.

These files must be present in the root directory of the project for the application to function correctly.
