 # Smart City Traffic Prediction

This project predicts the number of vehicles at different smart city traffic junctions using historical traffic data. The model uses date-time based features and junction information to estimate traffic volume.

## Project Overview

Traffic management is an important part of smart city planning. This project uses machine learning to analyze traffic patterns and predict vehicle count for a given junction and date-time.

The project includes:
- Data loading and exploration
- DateTime feature extraction
- Traffic pattern visualization
- Model training using Random Forest Regressor
- Model evaluation
- Saved model prediction using user input
- Prediction result saved in a report file

## Dataset

The dataset contains traffic records with the following columns:

- `DateTime`: Date and time of traffic record
- `Junction`: Junction number
- `Vehicles`: Number of vehicles
- `ID`: Unique record ID

The target column is:

```text
Vehicles
```

## Project Structure

```text
smart_city_traffic_prediction/
  data/
    data/
      train_aWnotuB.csv
      datasets_8494_11879_test_BdBKkAj.csv

  model/
    traffic_model.pkl
    model_columns.pkl

  reports/
    average_traffic_by_hour.png
    average_traffic_by_junction.png
    traffic_prediction_result.txt

  src/
    train_model.py
    predict.py

  README.md
  requirements.txt
  .gitignore
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Steps Performed

### 1. Data Loading

The training dataset is loaded using Pandas.

```python
df = pd.read_csv("data/data/train_aWnotuB.csv")
```

### 2. Data Exploration

Basic checks are performed using:

```python
df.head()
df.shape
df.info()
df.columns
df.isnull().sum()
```

These checks help understand the dataset structure and missing values.

### 3. DateTime Feature Extraction

The `DateTime` column is converted into a proper datetime object.

```python
df["DateTime"] = pd.to_datetime(df["DateTime"])
```

New features are extracted:

- Year
- Hour
- Day
- Month
- Day of week

These features help the model understand traffic behavior based on time.

### 4. Data Visualization

Two graphs are created and saved in the `reports` folder:

1. Average Traffic by Hour
2. Average Traffic by Junction

These graphs help understand traffic patterns.

### 5. Feature Selection

The unnecessary columns `DateTime` and `ID` are removed after useful features are extracted.

```python
df = df.drop(["DateTime", "ID"], axis=1)
```

The target column is:

```python
target = "Vehicles"
```

Input and output are separated:

```python
x = df.drop(columns=[target])
y = df[target]
```

### 6. Train-Test Split

The data is divided into training and testing parts.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)
```

### 7. Model Training

A Random Forest Regressor model is used for prediction.

```python
model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)
```

### 8. Model Evaluation

The model is evaluated using:

- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error
- R2 Score

The final R2 score achieved is approximately:

```text
0.964
```

This shows that the model performs well in predicting traffic volume.

### 9. Model Saving

The trained model and input column names are saved using Joblib.

```python
joblib.dump(model, "model/traffic_model.pkl")
joblib.dump(x.columns.tolist(), "model/model_columns.pkl")
```

### 10. Prediction

The `predict.py` file loads the saved model and takes user input:

- Junction number
- DateTime

Example input:

```text
Enter junction number: 1
Enter DateTime: 2015-11-01 01:00:00
```

The model predicts the number of vehicles and saves the result in:

```text
reports/traffic_prediction_result.txt
```

## Sample Output

```text
Predicted number of vehicles: 13
```

## Graph Insights

From the hourly traffic graph:
- Traffic is lowest during early morning hours.
- Traffic increases during daytime.
- Peak traffic is seen during evening hours.

From the junction-wise graph:
- Junction 1 has the highest average traffic.
- Junction 4 has the lowest average traffic.
- Junction location plays an important role in traffic prediction.

## How to Run the Project

### 1. Clone the repository

```bash
git clone <repository-link>
```

### 2. Move into the project folder

```bash
cd smart_city_traffic_prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train_model.py
```

### 5. Run prediction

```bash
python src/predict.py
```
# Streamlit App Screenshot

![Streamlit App Screenshot] (images/streamlit_app.png)

## Result

The model successfully predicts traffic volume for a given junction and date-time. After adding the `year` feature, the model performance improved and achieved an R2 score of approximately 0.964.

## Learning Outcome

Through this project, I learned:

- How to handle time-based data
- How to extract useful features from a DateTime column
- How to visualize traffic patterns
- How to train a Random Forest regression model
- How to evaluate model performance
- How to save and reuse a trained model
- How to create a prediction script using user input

## Author

Neha

## Domain

Data Science and Machine Learning