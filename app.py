import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Same model
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

model = NeuralNet()
model.load_state_dict(torch.load("digit_nn.pth", map_location="cpu"))
model.eval()

st.title("Digit Recognizer (Backpropagation NN)")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=25,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
)

if canvas_result.image_data is not None:

    img = canvas_result.image_data

    img = Image.fromarray(img.astype("uint8")).convert("L")
    img = img.resize((28, 28))

    img = np.array(img)
    img = 255 - img
    img = img / 255.0

    img = img.reshape(1, 1, 28, 28)
    img = torch.tensor(img, dtype=torch.float32)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, 1).item()

    st.subheader(f"Prediction: {pred}")
