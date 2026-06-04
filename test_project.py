import os
import pickle
import pandas as pd
import numpy as np

def test_files_exist():
    assert os.path.exists('students.csv'), "Dataset missing"
    assert os.path.exists('model.pkl'), "Model file missing"
    print("✅ Essential files exist.")

def test_model_prediction():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Test data
    test_data = pd.DataFrame({
        'Study_Hours': [8.0],
        'Attendance': [90.0],
        'Previous_Marks': [80.0],
        'Sleep_Hours': [7.0]
    })
    
    prediction = model.predict(test_data)
    assert 0 <= prediction[0] <= 100, f"Prediction {prediction[0]} out of range"
    print(f"✅ Model prediction test passed. Prediction: {prediction[0]:.2f}")

if __name__ == "__main__":
    print("Running tests...")
    try:
        test_files_exist()
        test_model_prediction()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
