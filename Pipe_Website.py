# Import Package 
import streamlit as st

import pandas as pd
import numpy as np

import joblib

# Import Model 
model = joblib.load('final_pipeline.pkl')

# Judul Website
st.title('Model Prediksi Resiko Gagal Bayar (Pipeline)')

# Predict via inputer
# Imputer
income = st.number_input('Income')
age = st.number_input('Age')
experience = st.number_input('Experience')
profession = st.text_input('Profession')
city = st.text_input('City')
state = st.text_input('State')
current_job_yrs = st.number_input('Curent Job (years)')
current_house_yrs = st.number_input('Curent House(years)')

# Tombol Imputer
input_data = pd.DataFrame({
    'income': [income],
    'age': [age],
    'experience': [experience],
    'profession': [profession],
    'city': [city],
    'state': [state],
    'current_job_yrs': [current_job_yrs],
    'current_house_yrs': [current_house_yrs]
})

if st.button('Prediksi Input Data'):
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.write('Tidak Gagal Bayar')
    else:
        st.write('Gagal Bayar')

# **Upload File CSV**
uploaded_file = st.file_uploader('Masukkan file CSV', type=['csv'])

# Pastikan file tersimpan di session_state agar tidak hilang setelah klik tombol
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Cek apakah kolom sudah sesuai
    expected_columns = ['income', 'age', 'experience', 'profession', 'city', 'state', 'current_job_yrs', 'current_house_yrs']
    if list(df.columns) != expected_columns:
        st.error(f"Kolom dalam CSV tidak sesuai! Diharapkan: {expected_columns}, tetapi ditemukan: {list(df.columns)}")
    else:
        st.session_state.df = df
        st.write('Upload Sukses!')

# **Tombol Prediksi**
if 'df' in st.session_state:  # Pastikan ada file yang diunggah
    if st.button('Prediksi File'):
        df = st.session_state.df  # Ambil DataFrame dari session_state
        
        # Prediksi menggunakan seluruh DataFrame
        prediction = model.predict(df)

        # Show Prediction
        for i, pred in enumerate(prediction):
            if pred == 0:
                st.write(f'Baris {i+1}: Tidak Gagal Bayar')
            else:
                st.write(f'Baris {i+1}: Gagal Bayar')




