#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
import streamlit as st
from PIL import Image
# Loading the saved model
#loaded_model = pickle.load(open('C:/Users/sohel/Dropbox/Ariana/Interview_Data/Python_model/Diabetes/trained_model.sav', 'rb'))
with open("random_forest_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)
# Creating a function for Prediction
def diabetes_prediction(input_data):
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # Giving a title
    st.title('Diabetes Prediction Web App')
    st.markdown("<span style='font-size: 20px;'><strong>By Sohel Ahmed</strong></span>", unsafe_allow_html=True)
    image = Image.open("Diabetes.png")
    st.image(image, use_column_width=True)
    # Getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # Code for Prediction
    diagnosis = ''

    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()

