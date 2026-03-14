import streamlit as st
import numpy as np
import cv2
import joblib
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = joblib.load("digit_model.pkl")

st.title("Handwritten Digit Recognizer")

st.write("Draw a digit (0–9)")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:

    img = canvas_result.image_data

    # Convert to grayscale
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)

    # Resize to dataset size
    img = cv2.resize(img, (8, 8))

    # Invert colors
    img = cv2.bitwise_not(img)

    # Normalize
    img = img / 16.0

    # Flatten image
    img = img.flatten().reshape(1, -1)

    prediction = model.predict(img)

    st.subheader(f"Predicted Digit: {prediction[0]}")
