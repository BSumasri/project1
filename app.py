import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# ---------------------------

# Load trained model
# ---------------------------
model = tf.keras.models.load_model(r"D:\AInML\IMDB-RNN_Project\model1.h5")

# ---------------------------
# Parameters (must match training)
# ---------------------------
maxlen = 500
word_index = imdb.get_word_index()

# ---------------------------
# Text encoder
# ---------------------------
def encode_review(text):
    words = text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words]
    padded = sequence.pad_sequences([encoded], maxlen=maxlen)
    return padded

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Movie Review Sentiment Analyzer")

user_input = st.text_area("Enter a movie review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text.")
    else:
        encoded = encode_review(user_input)
        prediction = model.predict(encoded)[0][0]

        if prediction > 0.5:
            st.success(f"Positive review ({prediction:.2f})")
        else:
            st.error(f"Negative review ({prediction:.2f})")