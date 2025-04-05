import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time
import warnings

# Configure warnings properly
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set page configuration
st.set_page_config(
    page_title="Personal Fitness Tracker",
    page_icon="🏃‍♂️",
    layout="wide"
)

# Function to load and preprocess data
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    
    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)
    
    # Add BMI column
    exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
    exercise_df["BMI"] = round(exercise_df["BMI"], 2)
    
    return exercise_df

# Function to prepare training and test data
@st.cache_data
def prepare_model_data(exercise_df):
    exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
    
    # Add BMI column to both training and test sets
    for data in [exercise_train_data, exercise_test_data]:
        data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
        data["BMI"] = round(data["BMI"], 2)
    
    # Prepare the training and testing sets
    exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
    exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)
    
    # Separate features and labels
    X_train = exercise_train_data.drop("Calories", axis=1)
    y_train = exercise_train_data["Calories"]
    
    X_test = exercise_test_data.drop("Calories", axis=1)
    y_test = exercise_test_data["Calories"]
    
    return X_train, y_train, X_test, y_test

# Function to train model
@st.cache_resource
def train_model(X_train, y_train):
    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6, random_state=42)
    random_reg.fit(X_train, y_train)
    return random_reg

# Function to collect user input
def user_input_features():
    with st.sidebar:
        st.header("User Input Parameters")
        
        age = st.slider("Age:", 10, 100, 30)
        bmi = st.slider("BMI:", 15, 40, 20)
        duration = st.slider("Duration (min):", 0, 35, 15)
        heart_rate = st.slider("Heart Rate:", 60, 130, 80)
        body_temp = st.slider("Body Temperature (°C):", 36, 42, 38)
        gender_button = st.radio("Gender:", ("Male", "Female"))
        
        gender = 1 if gender_button == "Male" else 0
        
        # Use column names to match the training data
        data_model = {
            "Age": age,
            "BMI": bmi,
            "Duration": duration,
            "Heart_Rate": heart_rate,
            "Body_Temp": body_temp,
            "Gender_male": gender  # Gender is encoded as 1 for male, 0 for female
        }
        
        features = pd.DataFrame(data_model, index=[0])
    return features

# Function to display progress bar
def show_progress():
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)

# Main application
def main():
    st.title("Personal Fitness Tracker")
    st.write("In this WebApp you will be able to observe your predicted calories burned in your body. Pass your parameters such as `Age`, `Gender`, `BMI`, etc., into this WebApp and then you will see the predicted value of kilocalories burned.")
    
    # Get user input
    df_user = user_input_features()
    
    # Load data
    try:
        exercise_df = load_data()
        X_train, y_train, X_test, y_test = prepare_model_data(exercise_df)
        model = train_model(X_train, y_train)
        
        # Display user parameters
        st.write("---")
        st.header("Your Parameters:")
        show_progress()
        st.write(df_user)
        
        # Align prediction data columns with training data
        df_user = df_user.reindex(columns=X_train.columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(df_user)
        
        # Display prediction
        st.write("---")
        st.header("Prediction:")
        show_progress()
        st.metric("Calories Burned", f"{round(prediction[0], 2)} kilocalories")
        
        # Display similar results
        st.write("---")
        st.header("Similar Results:")
        show_progress()
        
        # Find similar results based on predicted calories
        calorie_range = [prediction[0] - 10, prediction[0] + 10]
        similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & 
                                  (exercise_df["Calories"] <= calorie_range[1])]
        
        if not similar_data.empty:
            st.write(similar_data.sample(min(5, len(similar_data))))
        else:
            st.write("No similar results found in the dataset.")
        
        # Display general information
        st.write("---")
        st.header("General Information:")
        
        # Boolean logic for age, duration, etc., compared to the user's input
        boolean_age = (exercise_df["Age"] < df_user["Age"].values[0]).tolist()
        boolean_duration = (exercise_df["Duration"] < df_user["Duration"].values[0]).tolist()
        boolean_body_temp = (exercise_df["Body_Temp"] < df_user["Body_Temp"].values[0]).tolist()
        boolean_heart_rate = (exercise_df["Heart_Rate"] < df_user["Heart_Rate"].values[0]).tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Age Percentile", f"{round(sum(boolean_age) / len(boolean_age) * 100, 1)}%", 
                     help="Percentage of people in the dataset who are younger than you")
            st.metric("Heart Rate Percentile", f"{round(sum(boolean_heart_rate) / len(boolean_heart_rate) * 100, 1)}%",
                     help="Percentage of people in the dataset with lower heart rate during exercise")
        
        with col2:
            st.metric("Duration Percentile", f"{round(sum(boolean_duration) / len(boolean_duration) * 100, 1)}%",
                     help="Percentage of people in the dataset with shorter exercise duration")
            st.metric("Body Temperature Percentile", f"{round(sum(boolean_body_temp) / len(boolean_body_temp) * 100, 1)}%",
                     help="Percentage of people in the dataset with lower body temperature during exercise")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please make sure the data files 'calories.csv' and 'exercise.csv' are in the same directory as this app.")

if __name__ == "__main__":
    main()