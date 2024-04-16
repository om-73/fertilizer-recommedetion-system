import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load data
crop = pd.read_csv("crop_recommendation.csv")

# Data exploration
print(crop.shape)
print(crop.info())
print(crop.isnull().sum())
print(crop.duplicated().sum())
print(crop.describe())

# Data preprocessing
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7, 'apple': 8,
    'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
    'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
crop['crop_num'] = crop['label'].map(crop_dict)

X = crop.drop(['crop_num', 'label'], axis=1)
y = crop['crop_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Model evaluation
ypred = rfc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, ypred))

# Save the model
with open('model_without_scaling.pkl', 'wb') as file:
    pickle.dump(rfc, file)
