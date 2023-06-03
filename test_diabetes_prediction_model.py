from joblib import load
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Load the trained model from the pickle file
with open("DiabetesModel.pkl", "rb") as file:
    model = load(file)

# Define mappings for gender and smoking_history
gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
smoking_history_mapping = {'No Info': 0, 'never': 1, 'former': 2, 'current': 3, 'not current': 4, 'ever': 5}

# Load the dataset to calculate average values
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Calculate average HbA1c level and Blood Glucose level
average_hba1c = df['HbA1c_level'].mean()
average_blood_glucose = df['blood_glucose_level'].mean()

# Function to preprocess user input and make a prediction
def preprocess_input(user_input):
    # Preprocess gender and smoking history
    user_input['gender'] = gender_mapping[user_input['gender']]
    user_input['smoking_history'] = smoking_history_mapping[user_input['smoking_history']]

    # Replace "don't know" with average values
    if user_input['HbA1c_level'].lower() == "dk":
        user_input['HbA1c_level'] = average_hba1c
    else:
        user_input['HbA1c_level'] = float(user_input['HbA1c_level'])

    if user_input['blood_glucose_level'].lower() == "dk":
        user_input['blood_glucose_level'] = average_blood_glucose
    else:
        user_input['blood_glucose_level'] = float(user_input['blood_glucose_level'])

    return user_input

# Function to make a prediction
def predict_diabetes(user_input):
    # Preprocess user input
    preprocessed_input = preprocess_input(user_input)

    # Create a numpy array from preprocessed input
    input_array = np.array([[preprocessed_input['gender'], preprocessed_input['age'], preprocessed_input['hypertension'],
                             preprocessed_input['heart_disease'], preprocessed_input['smoking_history'], preprocessed_input['bmi'],
                             preprocessed_input['HbA1c_level'], preprocessed_input['blood_glucose_level']]])

    # Scale the input array
    scaler = MinMaxScaler()
    scaled_input = scaler.fit_transform(input_array)

    # Make the prediction
    prediction = model.predict(scaled_input)

    return prediction[0]

# User input
user_input = {
    'gender': input("Enter your gender (Male/Female/Other): "),
    'age': float(input("Enter your age: ")),
    'hypertension': int(input("Do you have hypertension? (0 for No, 1 for Yes): ")),
    'heart_disease': int(input("Do you have a heart disease? (0 for No, 1 for Yes): ")),
    'smoking_history': input("Enter your smoking history (No Info/never/former/current/not current/ever): "),
    'bmi': float(input("Enter your BMI: ")),
    'HbA1c_level': input("Enter your HbA1c level (or enter dk for 'don't know'): "),
    'blood_glucose_level': input("Enter your blood glucose level (or enter dk for 'don't know'): ")
}

# Make the prediction
prediction = predict_diabetes(user_input)

# Output the prediction
if prediction == 0:
    print("Based on the provided information, it is predicted that you do not have diabetes.")
else:
    print("Based on the provided information, it is predicted that you have diabetes.")