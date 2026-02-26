import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# 1. Load the Model
# We use st.cache_resource to load the model only once, speeding up the app
@st.cache_resource
def load_model():
    # Ensure 'model_bone.h5' is in the same directory or provide full path
    model = tf.keras.models.load_model('model_bone.h5')
    return model

# 2. Define Preprocessing Function (Must match training exactly)
def preprocess_image(image):
    # Convert PIL Image to NumPy array
    img_array = np.array(image)

    # Check if image is RGB, if so, convert to Grayscale
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Resize to the input size the model expects
    img_array = cv2.resize(img_array, (224, 224))

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_array = clahe.apply(img_array)

    # Normalize to 0-1 range (matching the training generator)
    img_array = img_array.astype(np.float32) / 255.0

    # Add Batch and Channel dimensions: Shape becomes (1, 224, 224, 1)
    img_array = np.expand_dims(img_array, axis=-1) # Add channel
    img_array = np.expand_dims(img_array, axis=0)  # Add batch
    
    return img_array

# 3. Build the UI
st.title("🦴 Bone Fracture Detection")
st.write("Upload an X-ray image to detect if a fracture is present.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray', use_container_width=True)
    
    st.write("Processing...")
    
    try:
        # Load model
        model = load_model()
        
        # Preprocess
        processed_img = preprocess_image(image)
        
        # Predict
        prediction = model.predict(processed_img)
        
        # The model output is [Prob_Class0, Prob_Class1]
        # Assuming Class 0 = Fractured, Class 1 = Normal (Check your class_indices!)
        # We will assume indices based on typical alphabetical sorting: 
        # 0: Fractured, 1: Normal (Verify this with your train_gen.class_indices)
        
        class_names = ['Fractured', 'Normal'] # Update order if necessary
        predicted_class_index = np.argmax(prediction)
        confidence = prediction[0][predicted_class_index]
        
        predicted_label = class_names[predicted_class_index]
        
        # Result Display
        st.divider()
        if predicted_label == 'Fractured':
            st.error(f"**Prediction: {predicted_label}**")
        else:
            st.success(f"**Prediction: {predicted_label}**")
            
        st.info(f"Confidence: {confidence * 100:.2f}%")
        
        # Optional: Show probabilities for both
        st.write("Class Probabilities:")
        st.progress(float(prediction[0][0]), text=f"Fractured: {prediction[0][0]:.2f}")
        st.progress(float(prediction[0][1]), text=f"Normal: {prediction[0][1]:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")