# 🌫 Urban Air Quality Index (AQI) Prediction System

---

## 📌 Project Overview

The **Urban Air Quality Index (AQI) Prediction System** is a machine learning-based web application designed to predict air quality categories using environmental and atmospheric parameters.

Built using a **Logistic Regression** model, the system processes pollutant and weather data to provide real-time AQI classification along with prediction confidence.

---
## 📂 Project Files

- `app.py` – Gradio web application for AQI prediction  
- `logistic_regression_model.joblib` – Trained Logistic Regression model  
- `label_encoder.joblib` – Saved label encoder used for encoding AQI categories  
- `AirPollutionPrediction.ipynb` – Model training notebook (Google Colab)  
- `requirements.txt` – Project dependencies required for deployment
---

## 🎯 Objective

To develop a reliable and interactive system that:

- Analyzes environmental pollutant data  
- Predicts AQI category accurately  
- Provides prediction confidence score  
- Offers a user-friendly web interface for real-time input  

---

## 🧠 Machine Learning Approach

- **Model Used:** Logistic Regression  
- **Training Platform:** Google Colab  
- **Prediction Type:** Multi-class classification  
- **Interface:** Gradio
- **Deployment & Hosting:** Hugging Face Spaces

---

## 📊 Input Parameters

The model accepts the following environmental inputs:

- PM2.5  
- PM10  
- NO₂  
- SO₂  
- CO  
- O₃  
- Temperature (°C)  
- Humidity (%)  
- Wind Speed (m/s)  
- Atmospheric Pressure (hPa)  
- Rainfall (mm)  
- Hour of the Day  
- Day of the Week  
- Month  

---

## ⚙️ System Architecture

1. User inputs environmental parameters through the web interface  
2. Inputs are structured into a DataFrame  
3. The trained Logistic Regression model processes the data  
4. AQI category is predicted  
5. Confidence score is calculated  
6. Results are displayed with visual emphasis  

---

## 💻 Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- Gradio  
- Joblib  

---

## 🚀 Key Features

- Real-time AQI category prediction  
- Interactive and responsive user interface  
- Confidence percentage display  
- Production-safe feature alignment  
- Deployment-ready architecture  

---

## 🔐 License

**All Rights Reserved.**  
Unauthorized copying, modification, or distribution is prohibited.

---

# 🚀 LIVE DEPLOYMENT LINK

## 🔗 https://huggingface.co/spaces/syedmuskan03/air-pollution-aqi-predictor
