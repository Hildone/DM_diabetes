import streamlit as st 
import pickle
import numpy as np



load_model = pickle.load(open('diabetes_model.pkl','rb'))

st.title("Prediksi Diabetes Bagi Wanita")

#input
col1,col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input("Pregnancies")
    Glucose = st.number_input(" Glucose")
    BloodPressure = st.number_input(" BloodPressure")
    SkinThickness= st.number_input(" SkinThickness")

with col2:
    Insulin = st.number_input(" Insulin")
    BMI = st.number_input(" BMI (Body Mass Index)")
    DiabetesPedigreeFunction = st.number_input(" Diabetes Pedigree Function")
    Age = st.number_input(" Ag")

#olah array
input_sample = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]

input_np_array = np.asarray(input_sample)
input_np_array_reshaped = input_np_array.reshape(1,-1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
std_data = sc.fit_transform(input_np_array_reshaped)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# std_data = sc.fit_transform(input_sample)

#predict
diagnosis=''

if st.button("Test Prediksi Diabetes"):
    diabetes_pred = load_model.predict(std_data)
    
    if(diabetes_pred[0]==0):
        diagnosis = "Pasien Tidak Terkena Diabetes"
    else :
        diagnosis = "Pasien Terkena Diabetes"
    
    st.success(diagnosis)
