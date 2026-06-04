@echo off
echo ===================================================
echo Student Performance Prediction System - Local Runner
echo ===================================================

echo [1/3] Installing/Updating dependencies...
pip install -r requirements.txt

echo [2/3] Training model and generating data...
python train_model.py

echo [3/3] Running tests...
python test_project.py

echo.
echo Setup Complete! Starting Streamlit Web App...
streamlit run app.py

pause
