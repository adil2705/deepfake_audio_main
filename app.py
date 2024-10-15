import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import io
from feedbacks import feedback  # Import your feedback function

# Global constant
max_length = 204

# Feature Extraction Process
def extract_features(audio_file, sr=16000):
    audio, _ = librosa.load(audio_file, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    # Padding or truncation for consistent shape
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > max_length:
        mfccs = mfccs[:, :max_length]

    return mfccs

# Load the trained model
model = tf.keras.models.load_model("Model/ann.h5")

# Function to detect a fake voice
def detect_fake_voice(audio_file, model):
    mfccs = extract_features(audio_file, sr=16000)
    prediction = model.predict(np.expand_dims(mfccs, axis=0))
    return prediction

# Streamlit app
st.title("DeepFake Audio Detection")
st.write("desicoder's ")


uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Read the uploaded file as a bytes-like object
    audio_bytes = uploaded_file.read()
    
    # Create a BytesIO object for librosa to load
    audio_file = io.BytesIO(audio_bytes)

    # Run detection
    prediction = detect_fake_voice(audio_file, model)

    # Generate feedback using the imported function
    feedback_message = feedback(np.max(prediction))  # Pass the maximum prediction score to your feedback function

    # Display results
    st.write("Prediction Score: ", prediction)
    st.write("Feedback: ", feedback_message)