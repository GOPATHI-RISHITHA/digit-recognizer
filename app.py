import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*5*5,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1,64*5*5)
        return self.fc(x)

model = CNN()
model.load_state_dict(torch.load("digit_cnn.pth", map_location="cpu"))
model.eval()

st.title("AI Digit Recognizer")

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
    img = img.resize((28,28))

    img = np.array(img)
    img = 255 - img
    img = img / 255.0

    img = img.reshape(1,1,28,28)
    img = torch.tensor(img, dtype=torch.float32)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output,1).item()

    st.subheader(f"Prediction: {pred}")
