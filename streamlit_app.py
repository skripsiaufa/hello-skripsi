import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Page title
st.set_page_config(page_title='ML Model Building', page_icon='🤖')
st.title('🤖 ML Model Building')

# Function to load and preprocess image
def preprocess_image(image):
    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load your model
model = tf.keras.models.load_model('https://github.com/skripsiaufa/hello-skripsi/blob/master/MobileNet.h5')

# Define your label names
label_names = ['asoka', 'kecubung', 'krokot', 'periwinkle', 'telang', 'zinnia']   # Update with your actual class names

# Streamlit app
def main():
    st.title("Image Classification App")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image and make prediction
        image_array = preprocess_image(uploaded_file)
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)

        # Display prediction
        st.subheader("Prediction:")
        st.write(f"Predicted class: {label_names[predicted_class]}")
        st.write(f"Confidence: {prediction[0][predicted_class]*100:.2f}%")
        st.write("All Predictions:")
        st.write(prediction)

if __name__ == '__main__':
    main()

