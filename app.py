from flask import Flask, render_template, request
from joblib import load
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Instatiate the flask web application
app = Flask(__name__)

# Load the model
with open("DiabetesModel.pkl", "rb") as file:
    model = load(file)

# Define gender and smoking_history mappings
gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
smoking_history_mapping = {'No Info': 0, 'never': 1, 'former': 2, 'current': 3, 'not current': 4, 'ever': 5}

# Load the dataset to calculate average values for HbA1c and Blood Glucose
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Calculate average for HbA1c and Blood Glucose
average_hba1c = df['HbA1c_level'].mean()
average_blood_glucose = df['blood_glucose_level'].mean()

# Function to preprocess user input
def preprocess_input(user_input):
    # Preprocess gender and smoking history
    user_input['gender'] = gender_mapping[user_input['gender']]
    user_input['smoking_history'] = smoking_history_mapping[user_input['smoking_history']]

    # Replace don't know answer with average values
    if user_input['HbA1c_level'].lower() == "dk":
        user_input['HbA1c_level'] = average_hba1c
    else:
        user_input['HbA1c_level'] = float(user_input['HbA1c_level'])

    if user_input['blood_glucose_level'].lower() == "dk":
        user_input['blood_glucose_level'] = average_blood_glucose
    else:
        user_input['blood_glucose_level'] = float(user_input['blood_glucose_level'])

    return user_input

# Function to make predictions
def predict_diabetes(user_input):
    # Preprocess user input
    preprocessed_input = preprocess_input(user_input)

    # Create a numpy array from preprocessed input
    input_array = np.array([[preprocessed_input['gender'], preprocessed_input['age'], preprocessed_input['hypertension'],
                             preprocessed_input['heart_disease'], preprocessed_input['smoking_history'], preprocessed_input['bmi'],
                             preprocessed_input['HbA1c_level'], preprocessed_input['blood_glucose_level']]])

    # Scale the input array as this was done when creating the model
    scaler = MinMaxScaler()
    scaled_input = scaler.fit_transform(input_array)

    # Make the prediction
    prediction = model.predict(scaled_input)

    return prediction[0]

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction result page
@app.route('/predict', methods=['POST'])
def predict():
    user_input = {
        'gender': request.form['gender'],
        'age': float(request.form['age']),
        'hypertension': int(request.form['hypertension']),
        'heart_disease': int(request.form['heart_disease']),
        'smoking_history': request.form['smoking_history'],
        'bmi': float(request.form['bmi']),
        'HbA1c_level': request.form['hba1c_level'],
        'blood_glucose_level': request.form['blood_glucose_level']
    }

    # Make prediction
    prediction = predict_diabetes(user_input)

    # Output prediction
    if prediction == 0:
        result = "Based on the information provided, it is predicted that you don't have diabetes."
    else:
        result = "Based on the information provided, it is predicted that you have diabetes."

    return render_template('result.html', result=result)

# Run flask application
if __name__ == '__main__':
    app.run(debug=True)
