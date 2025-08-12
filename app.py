import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load trained model, scaler, and encoders

model=tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('onehot_encoder_func.pkl', 'rb') as file:
    onehot_encoder_func = pickle.load(file)

with open('onehot_encoder_ST.pkl', 'rb') as file:
    onehot_encoder_ST = pickle.load(file)

with open('onehot_encoder_TN.pkl', 'rb') as file:
    onehot_encoder_TN = pickle.load(file)



## streamlit app
st.title('Defect Prediction')

# User input
Functionality = st.selectbox('Functionality', onehot_encoder_func.categories_[0])
Story_Type = st.selectbox('Story Type', onehot_encoder_ST.categories_[0])
Core_Function = st.selectbox('Core Function', label_encoder.classes_)
Affects_Workflow = st.selectbox('Affects Workflow', label_encoder.classes_)
Has_Workaround = st.selectbox('Has Workaround', label_encoder.classes_)  
Tester_Name = st.selectbox('Tester Name', onehot_encoder_TN.categories_[0])
Severity_Score = st.number_input('Severity Score')
Time_Taken_Minutes = st.number_input('Time Taken (Minutes)')




# Prepare the input data
input_data = pd.DataFrame({
    'Core_Function': [label_encoder.transform([Core_Function])[0]],
    'Affects_Workflow': [label_encoder.transform([Affects_Workflow])[0]],
    'Has_Workaround': [label_encoder.transform([Has_Workaround])[0]],
    'Severity_Score': [Severity_Score],
    'Time_Taken_Minutes': [Time_Taken_Minutes]
})

# One-hot encode
func_encoded = onehot_encoder_func.transform([[Functionality]]).toarray()
func_encoded_df = pd.DataFrame(func_encoded, columns=onehot_encoder_func.get_feature_names_out(['Functionality']))

ST_encoded = onehot_encoder_ST.transform([[Story_Type]]).toarray()
ST_encoded_df = pd.DataFrame(ST_encoded, columns=onehot_encoder_ST.get_feature_names_out(['Story_Type']))

TN_encoded = onehot_encoder_TN.transform([[Tester_Name]]).toarray()
TN_encoded_df = pd.DataFrame(TN_encoded, columns=onehot_encoder_TN.get_feature_names_out(['Tester_Name']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), func_encoded_df], axis=1)
input_data = pd.concat([input_data.reset_index(drop=True), ST_encoded_df], axis=1)
input_data = pd.concat([input_data.reset_index(drop=True), TN_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

predicted_class = np.argmax(prediction)  # index of highest prob
class_names = ["No Issues", "P1", "P2", "P3"]  # your label order

st.write(f"Predicted class: {class_names[predicted_class]}")
st.write(f"Probability: {prediction[0][predicted_class]:.2f}")
