import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load the pre-trained model
model = tf.keras.models.load_model('imageclassifier.h5')

# Streamlit app
st.title("Image Classifier App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    #
    #     # Make a prediction using the loaded model
    predictions = model.predict(image)
    rounded_predictions = np.round(predictions)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Prediction probabilities:", rounded_predictions.tolist())

    # class_labels = ["Cat", "Dog"]  # Replace with your actual class labels
    #
    # predicted_label = class_labels[np.argmax(predictions)]
    # st.write("Predicted Label:", predicted_label)
