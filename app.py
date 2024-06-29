import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Function to preprocess the image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Rescale to [0, 1]
    return image

# Load the saved model
def load_saved_model():
    model_path = 'models/model.keras'
    model = load_model(model_path)
    return model

model = load_saved_model()

# Define classes for classification (replace with your actual class names)
classes = ['Asian-Green-Bee-Eater', 'Brown-Headed-Barbet', 'Brown-Headed-Barbet', 'Brown-Headed-Barbet', 'Brown-Headed-Barbet']

# Streamlit app
st.title('Image Classification Demo')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    height=256
    width=256
    processed_image = preprocess_image(image, target_size=(height, width))

    # Make predictions
    prediction = model.predict(processed_image)
    predicted_class = classes[np.argmax(prediction)]

    # Display prediction
    st.write(f'Prediction: {predicted_class}')
    st.write(f'Confidence: {round(100 * np.max(prediction), 2)} %')
