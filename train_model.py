import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle
import os

# 1. Data Collection
def generate_data():
    np.random.seed(42)
    n_samples = 200
    study_hours = np.random.uniform(2, 10, n_samples)
    attendance = np.random.uniform(60, 100, n_samples)
    previous_marks = np.random.uniform(40, 100, n_samples)
    sleep_hours = np.random.uniform(5, 9, n_samples)
    
    # Realistic relationship
    final_marks = (0.4 * (study_hours * 10) + 
                   0.2 * attendance + 
                   0.3 * previous_marks + 
                   0.1 * (sleep_hours * 10) + 
                   np.random.normal(0, 2, n_samples))
    final_marks = np.clip(final_marks, 0, 100)
    
    df = pd.DataFrame({
        'Study_Hours': np.round(study_hours, 1),
        'Attendance': np.round(attendance, 1),
        'Previous_Marks': np.round(previous_marks, 1),
        'Sleep_Hours': np.round(sleep_hours, 1),
        'Final_Marks': np.round(final_marks, 1)
    })
    df.to_csv('students.csv', index=False)
    print("Dataset 'students.csv' generated.")
    return df

if not os.path.exists('students.csv'):
    df = generate_data()
else:
    # 2. Data Preprocessing
    df = pd.read_csv('students.csv')

print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Statistical Summary ---")
print(df.describe())

# 3. Exploratory Data Analysis (EDA)
def perform_eda(df):
    plt.figure(figsize=(12, 8))
    
    # Histograms
    df.hist(figsize=(10, 8))
    plt.tight_layout()
    plt.savefig('histograms.png')
    plt.close()
    
    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('heatmap.png')
    plt.close()
    
    # Scatter plot: Study Hours vs Final Marks
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Study_Hours', y='Final_Marks', data=df)
    plt.title('Study Hours vs Final Marks')
    plt.savefig('scatter_study.png')
    plt.close()

perform_eda(df)
print("\nEDA plots saved: histograms.png, heatmap.png, scatter_study.png")

# 4. Feature Engineering
X = df[['Study_Hours', 'Attendance', 'Previous_Marks', 'Sleep_Hours']]
y = df['Final_Marks']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Machine Learning Model - Linear Regression
# Linear Regression fits a linear model with coefficients to minimize the residual sum of squares 
# between the observed targets in the dataset, and the targets predicted by the linear approximation.
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained using Linear Regression.")

# 7. Model Evaluation
y_pred = model.predict(X_test)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Interpretation:
# MAE: Average magnitude of the errors in a set of predictions.
# MSE: Average of the squares of the errors.
# RMSE: Square root of MSE, penalizes larger errors more.
# R²: Proportion of the variance for the dependent variable that's explained by independent variables.

# 8. Predictions for new students
sample_data = pd.DataFrame({
    'Study_Hours': [8, 5],
    'Attendance': [95, 70],
    'Previous_Marks': [85, 60],
    'Sleep_Hours': [7, 6]
})
predictions = model.predict(sample_data)
print("\n--- Sample Predictions ---")
for i, pred in enumerate(predictions):
    print(f"Student {i+1} Predicted Marks: {pred:.2f}")

# 9. Model Saving
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved as 'model.pkl'.")

# Test loading
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
print("Loaded model and verified functionality.")
