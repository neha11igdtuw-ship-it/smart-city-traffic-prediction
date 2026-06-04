import pandas as pd
import joblib

model = joblib.load("model/traffic_model.pkl")
print("\n model loaded successfully ")

model_columns = joblib.load("model/model_columns.pkl")
print("\n model loaded successfully ")

junction = int(input("Enter junction number : "))
date_time = input("Enter DateTime (. example : 2016-01-01 08:00:00) : ")

date_time = pd.to_datetime(date_time)


newData = {
    "Junction": junction,
    "year":date_time.year,
    "hour":date_time.hour,
    "day":date_time.day,
    "month":date_time.month,
    "day_of_week":date_time.dayofweek
}

input_df = pd.DataFrame([newData])

input_df = input_df.reindex(columns = model_columns,fill_value = 0)

#predict 
prediction = model.predict(input_df)

results = round(prediction[0])
print("\n predicted no. of vehicles : ", results)

file = open("reports/traffic_prediction_result.txt","w")
file.write("Predicted no. of vehicles :" +str(results))
file.close()
print("\n result saved successfully")
