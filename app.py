import streamlit as st
import numpy as np
import joblib
from PIL import Image
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
    img = Image.fromarray(img.astype("uint8")).convert("L")

    # Resize to dataset size
    img = img.resize((8, 8))

    # Convert to numpy
    img_array = np.array(img)

    # Invert colors
    img_array = 255 - img_array

    # Normalize
    img_array = img_array / 16.0

    # Flatten
    img_array = img_array.flatten().reshape(1, -1)

    prediction = model.predict(img_array)

    st.subheader(f"Predicted Digit: {prediction[0]}")
