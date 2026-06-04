@echo off
echo ===================================================
echo Student Performance Prediction System - Local Runner
echo ===================================================

echo [1/3] Installing/Updating dependencies...
py -m pip install -r requirements.txt

echo [2/3] Training model and generating data...
py train_model.py

echo [3/3] Running tests...
py test_project.py

echo.
echo Setup Complete! Starting Streamlit Web App...
py -m streamlit run app.py

pause
