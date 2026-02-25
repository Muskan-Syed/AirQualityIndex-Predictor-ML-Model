import gradio as gr
import pandas as pd
import joblib

# Load trained model
model = joblib.load("logistic_regression_model.joblib")

# Mappings
day_of_week_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4,
    'Saturday': 5, 'Sunday': 6
}

month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def predict_aqi(
    PM2_5, PM10, NO2, SO2, CO, O3,
    Temp_C, Humidity_percent,
    Wind_Speed_mps, Pressure_hPa,
    Rain_mm, Hour, DayOfWeek, Month
):
    try:
        input_df = pd.DataFrame([[ 
            PM2_5, PM10, NO2, SO2, CO, O3,
            Temp_C, Humidity_percent,
            Wind_Speed_mps, Pressure_hPa,
            Rain_mm, Hour,
            day_of_week_map[DayOfWeek],
            month_map[Month]
        ]],
        columns=[
            'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
            'Temp_C', 'Humidity_%',
            'Wind_Speed_mps', 'Pressure_hPa',
            'Rain_mm', 'Hour', 'DayOfWeek', 'Month'
        ])

        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            confidence = round(max(probs) * 100, 2)
        else:
            confidence = "N/A"

        color = "#00e5ff"
        if "Medium" in str(prediction):
            color = "#ffcc00"
        if "High" in str(prediction):
            color = "#ff3366"

        return f"""
        <div style="
            font-size:38px;
            font-weight:900;
            text-align:center;
            padding:60px;
            border-radius:30px;
            border:4px solid {color};
            box-shadow:0 0 40px {color};
            color:{color};
            background:rgba(255,255,255,0.08);
            backdrop-filter: blur(12px);
        ">
        🌌 PREDICTED AQI CATEGORY <br><br>
        {prediction}
        <br><br>
        🔥 Confidence: {confidence}%
        </div>
        """

    except Exception as e:
        return f"<div style='color:red; font-size:22px;'>Error: {str(e)}</div>"


cosmic_css = """
body {
    background: linear-gradient(135deg, #c6f1ff, #89f7fe, #66a6ff, #b3ecff);
    background-size: 400% 400%;
    animation: gradientFlow 15s ease infinite;
}

@keyframes gradientFlow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

h1 {
    font-size: 52px !important;
    font-weight: 900 !important;
    text-align: center;
    color: #003366;
}

h2 {
    font-size: 28px !important;
    font-weight: 800 !important;
    text-align: center;
    color: #002244;
}

label {
    font-size: 20px !important;
    font-weight: 800 !important;
    color: #001a33 !important;
}
"""

with gr.Blocks(css=cosmic_css) as demo:

    gr.Markdown("# 🌫 **Urban Air Quality Index Predictor**")
    gr.Markdown("## **Enter environmental parameters to predict AQI category.**")

    PM2_5 = gr.Slider(0, 200, 50, step=0.1, label="PM2.5")
    PM10 = gr.Slider(0, 300, 100, step=0.1, label="PM10")
    NO2 = gr.Slider(0, 100, 25, step=0.1, label="NO2")
    SO2 = gr.Slider(0, 50, 5, step=0.1, label="SO2")
    CO = gr.Slider(0, 10, 1, step=0.01, label="CO")
    O3 = gr.Slider(0, 150, 40, step=0.1, label="O3")
    Temp_C = gr.Slider(-10, 40, 20, step=0.1, label="Temperature (°C)")
    Humidity_percent = gr.Slider(0, 100, 60, step=0.1, label="Humidity (%)")
    Wind_Speed_mps = gr.Slider(0, 20, 3, step=0.1, label="Wind Speed (m/s)")
    Pressure_hPa = gr.Slider(950, 1050, 1010, step=0.1, label="Pressure (hPa)")
    Rain_mm = gr.Slider(0, 50, 0, step=0.1, label="Rainfall (mm)")
    Hour = gr.Slider(0, 23, 12, step=1, label="Hour of Day")
    DayOfWeek = gr.Dropdown(list(day_of_week_map.keys()), value="Wednesday", label="Day of Week")
    Month = gr.Dropdown(list(month_map.keys()), value="July", label="Month")

    predict_button = gr.Button("🚀 PREDICT AQI")
    output = gr.HTML()

    predict_button.click(
        predict_aqi,
        inputs=[
            PM2_5, PM10, NO2, SO2, CO, O3,
            Temp_C, Humidity_percent,
            Wind_Speed_mps, Pressure_hPa,
            Rain_mm, Hour, DayOfWeek, Month
        ],
        outputs=output
    )

demo.launch()
