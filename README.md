Urban Air Quality Index (AQI) Prediction System is a machine learning-based web application that predicts air quality categories using environmental and atmospheric parameters. Built using Logistic Regression, the system processes pollutant and weather data to provide real-time AQI classification along with prediction confidence.

Urban Air Quality Index (AQI) Prediction System
📌 Project Overview

The Urban Air Quality Index (AQI) Prediction System is a machine learning-based web application that predicts the AQI category based on real-time environmental and atmospheric parameters.

The system uses a trained Logistic Regression model to classify air quality levels into predefined categories (e.g., Low, Medium, High) and displays prediction confidence.

🎯 Objective

To build a reliable and interactive system that:

Analyzes environmental pollutant data

Predicts AQI category accurately

Provides confidence score for prediction

Offers a user-friendly web interface for real-time input

🧠 Machine Learning Approach

Model Used: Logistic Regression

Training Platform: Google Colab

Feature Engineering: Structured environmental parameters

Prediction Type: Multi-class classification

Deployment: Web-based interface using Gradio

📊 Input Parameters

The model takes the following environmental inputs:

PM2.5

PM10

NO₂

SO₂

CO

O₃

Temperature (°C)

Humidity (%)

Wind Speed (m/s)

Atmospheric Pressure (hPa)

Rainfall (mm)

Hour of the Day

Day of the Week

Month

⚙️ System Architecture

User inputs environmental parameters via web UI

Inputs are structured into a DataFrame

Pre-trained Logistic Regression model processes data

AQI category is predicted

Confidence score is calculated

Result is displayed with visual emphasis

💻 Technologies Used

Python

Pandas

Scikit-learn

Gradio

Joblib

🚀 Key Features

Real-time AQI category prediction

Clean and interactive web interface

Confidence percentage display

Proper feature name alignment for production safety

Deployment-ready architecture

🔐 License

All Rights Reserved.
Unauthorized copying, modification, or distribution is prohibited.
