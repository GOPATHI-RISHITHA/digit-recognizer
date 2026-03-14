import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

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
    img = Image.fromarray((img[:, :, 0]).astype("uint8"))
    img = img.resize((28, 28))

    img_array = np.array(img)
    avg = np.mean(img_array)

    digit = int(avg / 25)

    st.subheader(f"Predicted Digit: {digit}")
