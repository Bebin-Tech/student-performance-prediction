# Student Performance Prediction System

A complete end-to-end Machine Learning project to predict a student's final exam marks based on study habits and academic factors. This project includes data generation, model training, evaluation, and an interactive web interface.

## Project Objective
Predict final exam marks using features like:
- **Study_Hours**: Total hours spent studying.
- **Attendance**: Percentage of classes attended.
- **Previous_Marks**: Marks obtained in previous exams.
- **Sleep_Hours**: Average daily sleep duration.

## Project Structure
- `students.csv`: The generated dataset.
- `train_model.py`: Script to generate data, perform EDA, train the Linear Regression model, and save it.
- `app.py`: Streamlit application for real-time predictions and visualizations.
- `test_project.py`: Script to verify the model and file structure.
- `requirements.txt`: List of Python dependencies.
- `*.png`: EDA visualizations generated during training.

## Local Setup & Execution

### 1. Prerequisites
Ensure you have Python installed (3.8+ recommended).

### 2. Install Dependencies
Open your terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Run the training script to generate the dataset and train the ML model:
```bash
python train_model.py
```
This will:
- Create `students.csv`.
- Save EDA plots: `histograms.png`, `heatmap.png`, `scatter_study.png`.
- Output model evaluation metrics (MAE, MSE, RMSE, R²).
- Save the trained model as `model.pkl`.

### 4. Run the Web Application
Launch the interactive Streamlit UI:
```bash
streamlit run app.py
```

### 5. Run Tests
Verify the project functionality:
```bash
python test_project.py
```

## Model Evaluation
The system uses **Linear Regression**, providing high interpretability and performance for academic factor analysis. Evaluation metrics are printed during the training phase.

## Author
Senior ML Engineer
