import pandas as pd
import streamlit as st
import joblib

model = joblib.load("model/traffic_model.pkl")
model_columns  = joblib.load("model/model_columns.pkl")

st.title("Smart City Traffic Prediction")

st.write("This app predicts the number of vehicles based on junction and date-time.")

junction = st.number_input("Enter junction number ",min_value=1,max_value=4) 

date_time = st.text_input("Enter date time ","2016-01-01 09:00:00")

if st.button("Predict Traffic"):
    date_time = pd.to_datetime(date_time)

    new_data = {
        "junction" : junction,
        "year" : date_time.year,
        "hour" : date_time.hour,
        "day" : date_time.day,
        "month" : date_time.month,
        "day_of_week" : date_time.dayofweek,
    }

    input_df = pd.DataFrame([new_data])

    input_df = input_df.reindex(columns = model_columns,fill_value = 0)
   
    prediction = model.predict(input_df)
    result = round(prediction[0])

    st.success(f"Predicted Number of Vehicles : {result}")
    st.subheader("Traffic Pattern Visualizations ")
    st.image("reports/average_traffic_by_hour.png")
    st.image("reports/average_traffic_by_junction.png")