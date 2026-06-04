import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page title
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# Load model
@st.cache_resource
def load_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# Sidebar for inputs
st.sidebar.header("Input Student Data")

def user_input_features():
    study_hours = st.sidebar.slider("Study Hours", 0.0, 10.0, 5.0)
    attendance = st.sidebar.slider("Attendance (%)", 0.0, 100.0, 75.0)
    prev_marks = st.sidebar.slider("Previous Marks", 0.0, 100.0, 65.0)
    sleep_hours = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0)
    
    data = {
        'Study_Hours': study_hours,
        'Attendance': attendance,
        'Previous_Marks': prev_marks,
        'Sleep_Hours': sleep_hours
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main page
st.title("Student Performance Prediction System")
st.markdown("""
This app predicts the **Final Marks** of a student based on their study habits and academic factors.
""")

if model is not None:
    # Prediction
    prediction = model.predict(input_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Input Parameters")
        st.write(input_df)
        
    with col2:
        st.subheader("Prediction Result")
        st.success(f"Estimated Final Marks: **{prediction[0]:.2f}**")

    # Visualizations
    st.divider()
    st.subheader("Data Visualizations")
    
    if os.path.exists('students.csv'):
        df = pd.read_csv('students.csv')
        
        tab1, tab2, tab3 = st.tabs(["Correlation", "Distributions", "Study vs Marks"])
        
        with tab1:
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap='viridis', ax=ax)
            st.pyplot(fig)
            
        with tab2:
            st.write("### Feature Distributions")
            feature = st.selectbox("Select Feature", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, ax=ax)
            st.pyplot(fig)
            
        with tab3:
            st.write("### Study Hours vs Final Marks")
            fig, ax = plt.subplots()
            sns.scatterplot(x='Study_Hours', y='Final_Marks', data=df, ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Please run 'train_model.py' first to generate data and train the model.")
else:
    st.error("Model file 'model.pkl' not found. Please run 'train_model.py' first.")
