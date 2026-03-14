import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = tf.keras.models.load_model("digit_cnn_model.h5")

st.title("🧠 Handwritten Digit Recognition")

st.write("Draw a digit (0-9) in the box below")

# Canvas
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

    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img,(28,28))

    img = img / 255.0

    img = img.reshape(1,28,28,1)

    prediction = model.predict(img)

    digit = np.argmax(prediction)

    confidence = np.max(prediction)

    st.subheader(f"Prediction: {digit}")
    st.write(f"Confidence: {confidence:.2f}")