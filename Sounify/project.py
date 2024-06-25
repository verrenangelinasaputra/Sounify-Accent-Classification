import streamlit as st
from joblib import load
import numpy as np
import soundfile as sf
from scipy.signal import resample
import librosa
import pandas as pd
import os
import tensorflow as tf 

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

st.title('Sounify')
audio_file = st.file_uploader('Upload Audio', type=["wav", "mp3", "m4a"])

target_sample_rate = 44100


def load_model(model_name):
    if model_name.endswith('.h5'):
        model = tf.keras.models.load_model(f'./{model_name}')
    else:
        model = load(f'./{model_name}.joblib')
    return model

@st.cache_data
def load_label_encoder():
    label_encoder = load('./label_encoder.joblib')
    return label_encoder

if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

import base64

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_encoded = get_image_base64('./logos.png')  

st.sidebar.markdown("""
    <div style="margin-top: -50px; margin-bottom: 50px;  display: flex; justify-content: center;">
        <img src="data:image/png;base64,{}" style="width: 50%;">
    </div>
    """.format(image_encoded), unsafe_allow_html=True)

# gunakan model yang skenario modelnya menghasilkan akurasi terbaik
model_options = {
    'K-Nearest Neighbours (KNN)': 'knn_model_5',
    'Decision Tree (DT)': 'dt_model_5',
    'Random Forest (RF)': 'randf_model_5',
    'Convolutional Neural Network (CNN)': 'cnn'
}

selected_model_desc = st.sidebar.selectbox("Select Model", list(model_options.keys()))
selected_model = model_options[selected_model_desc]


if st.sidebar.button("Load Model"):
    st.session_state.model = load_model(selected_model)
    st.session_state.label_encoder = load_label_encoder()
    st.sidebar.success(f"{selected_model} Loaded")

def extract_features(audio_bytes, sr=44100, n_mfcc=13):
    with open('temp_audio_file.wav', 'wb') as f:
        f.write(audio_bytes)
    
    y, sr = sf.read('temp_audio_file.wav')
    
    if sr != target_sample_rate:
        y = resample(y, int(len(y) * target_sample_rate / sr))
        sr = target_sample_rate 

    samples_10_sec = target_sample_rate * 10  
    y_10_sec = y[:samples_10_sec]

    mfccs = librosa.feature.mfcc(y=y_10_sec, sr=sr, n_mfcc=n_mfcc)

    mfccs_normalized = (mfccs - np.mean(mfccs)) / np.std(mfccs)

    return mfccs

if st.sidebar.button("Classify the Accent"):
    if audio_file is not None:
        if st.session_state.model is None or st.session_state.label_encoder is None:
            st.sidebar.error('Please load the model and label encoder first.')
        else:
            st.sidebar.success("Processing the Classification")
            audio_bytes = audio_file.read()
            
            features = extract_features(audio_bytes)
            
            mfcc_features = {f'MFCC_{i+1}': np.mean(features[i]) for i in range(features.shape[0])}
            input_features = pd.DataFrame([mfcc_features])
            
            prediction = st.session_state.model.predict(input_features)
            
            
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown(f'<div style="background-color: green; color: white; padding: 10px; border-radius: 5px;"><strong>The predicted accent is: {prediction}</strong></div>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)  
            st.audio(audio_bytes)
    else:
        st.sidebar.error('Please upload an audio file')

