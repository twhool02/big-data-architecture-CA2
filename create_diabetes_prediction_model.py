#!/usr/bin/env python

# Import required libraries
import pandas as pd
import pickle
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import shapiro

# Import the dataset into a pandas dataframe
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Make a copy of the dataframe
df_diabetes = df.copy()

# Drop any rows with missing values
df_diabetes.dropna(axis=0, inplace=True)

# Define mappings for gender and smoking_history
gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
smoking_history_mapping = {'No Info': 0, 'never': 1, 'former': 2, 'current': 3, 'not current': 4, 'ever': 5}

# Convert the words to numbers
df_diabetes['gender'] = df_diabetes['gender'].map(gender_mapping)
df_diabetes['smoking_history'] = df_diabetes['smoking_history'].map(smoking_history_mapping)

### Split Data into Feature and Label

# Create label and features dataframes
label = df_diabetes['diabetes']
features = df_diabetes.drop(['diabetes'],axis=1)

# split data into train and test at 80/20 ratio
x_train, x_test, y_train, y_test = train_test_split(features.values, label.values, test_size=0.2, random_state=0)

# scale the values for features
from sklearn.preprocessing import MinMaxScaler

# scaler = StandardScaler()
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# use ensemble learning i.e. random forests to make a prediction
# instantiate the random forest classifier
# class weight is adjusted to account for lower number of diabetes cases
clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', class_weight='balanced')

# Train the model
clf = clf.fit(x_train, y_train)

# Save the model as a pickle file
dump(clf , "DiabetesModel.pkl")

