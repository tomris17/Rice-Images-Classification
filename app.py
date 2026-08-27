import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("rice_model.keras")
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

def process_image(img):
    img = img.resize((128, 128))
    img = np.array(img)
    
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
        
    img = np.expand_dims(img, axis=0)
    return img

st.title("Rice Species Classification")
st.write("Upload a rice image to predict its variety.")

file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    processed_img = process_image(img)
    predictions = model.predict(processed_img)
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    
    st.subheader(f"Prediction: {predicted_class_name}")
    st.write(f"Confidence: %{confidence:.2f}")