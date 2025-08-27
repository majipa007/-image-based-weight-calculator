import streamlit as st
from PIL import Image
import os
from segementer import segment_image

st.title("Goat Weight Detection - Image Segmenter")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open(os.path.join("temp_image.jpg"), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="Original Image", use_container_width=True)
    st.write("")
    st.write("Segmenting...")

    # Perform segmentation
    segmented_img = segment_image("temp_image.jpg")

    if segmented_img:
        st.image(segmented_img, caption="Segmented Image", use_container_width=True)
    else:
        st.write("No goats detected or an error occurred during segmentation.")
    
    # Clean up the temporary file
    os.remove("temp_image.jpg")
