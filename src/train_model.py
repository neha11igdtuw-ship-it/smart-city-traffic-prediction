import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib 

#collect data

df = pd.read_csv("data/data/train_aWnotuB.csv")

# cleaning data 

print("\n first 5 rows : ")
print(df.head())

print("\n shape : ")
print(df.shape)

print("\n info :")
print(df.info())

print("\n columns :")
print(df.columns)

print("\n Missing ")
print(df.isnull().sum())


#exploring data 

df['DateTime'] = pd.to_datetime(df["DateTime"])
df['year'] = df['DateTime'].dt.year
df["hour"] = df["DateTime"].dt.hour
df["day"] = df["DateTime"].dt.day
df["month"] = df["DateTime"].dt.month
df["day_of_week"] = df["DateTime"].dt.dayofweek


 
hourly_traffic = df.groupby("hour")["Vehicles"].mean()

plt.figure(figsize=(8, 5))
plt.plot(hourly_traffic.index, hourly_traffic.values, marker="o")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Number of Vehicles")
plt.title("Average Traffic by Hour")
plt.grid(True)
plt.savefig("reports/average_traffic_by_hour.png")
plt.show()

junction_traffic = df.groupby("Junction")["Vehicles"].mean()

plt.figure(figsize=(8, 5))
plt.bar(junction_traffic.index, junction_traffic.values)
plt.xlabel("Junction")
plt.ylabel("Average Number of Vehicles")
plt.title("Average Traffic by Junction")
plt.xticks(junction_traffic.index)
plt.grid(True)
plt.savefig("reports/average_traffic_by_junction.png")
plt.show()

df = df.drop(["DateTime","ID"], axis = 1)

print(df.head())
print(df.info())
print(df.columns)
print(df.tail())

#analyzing data

target = "Vehicles"

x = df.drop(columns=[target])        
y = df[target]

print ("\n Input x : ", x.head())

print("\n target y :", y.head())
print("\n X shape : ", x.shape)
print("\n Y shape : ", y.shape)

x_train,  x_test, y_train,  y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)

print("\n x_train_shape : ", x_train.shape)
print("x_test shape:", x_test.shape)

print("\n y_train_shape : ", y_train.shape)
print("y_test shape:", y_test.shape)

#model selection 

model = RandomForestRegressor(random_state =42)
model.fit(x_train,y_train)

print("\n Model trained successfully")

# prediction

y_pred = model.predict(x_test)

print("\n Predicted values :", y_pred[:10])
print("\n Actual values :", y_test.values[:10])

#Analyzing 

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
rmse = np.sqrt(mse)

print("\n R2 score :", r2)
print("\n mse :", mse)
print("\n mae :", mae )
print("\n rmse : ", rmse )

#saving model

joblib.dump(model,"model/traffic_model.pkl")
print("\n model saved successfully!")

joblib.dump(x.columns.tolist(),"model/model_columns.pkl")
print("\n Column model saved successfully!")

