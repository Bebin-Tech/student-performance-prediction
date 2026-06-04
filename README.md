# Student Performance Prediction System

A complete end-to-end Machine Learning project to predict a student's final exam marks based on study habits and academic factors.

## Project Objective
Predict final exam marks using features like Study Hours, Attendance, Previous Marks, and Sleep Hours.

## Features
- **Data Collection**: Synthetic dataset generation with 200 records (`students.csv`).
- **Data Preprocessing**: Handling nulls and statistical analysis.
- **EDA**: Visualizations including correlation heatmaps and histograms.
- **Model**: Linear Regression algorithm.
- **UI**: Interactive Streamlit web application.

## Project Structure
- `students.csv`: The dataset.
- `train_model.py`: Script to preprocess data, perform EDA, train the model, and save it.
- `app.py`: Streamlit application for predictions.
- `model.pkl`: Saved trained model.
- `requirements.txt`: Project dependencies.

## Installation & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   ```bash
   python train_model.py
   ```
   This will generate `students.csv`, perform EDA (saving plots), and create `model.pkl`.

3. **Run the Web App**:
   ```bash
   streamlit run app.py
   ```

## Model Evaluation Metrics
The model is evaluated using:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R² Score**: Coefficient of Determination

## Author
Senior ML Engineer
