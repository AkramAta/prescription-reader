<<<<<<< HEAD
import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Ensure the Tesseract path is correctly set for your system
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Improved preprocessing for handwritten OCR
def preprocess_image(uploaded_file):
    image = np.array(Image.open(uploaded_file).convert("L"))  # Convert to grayscale
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Increase resolution
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian blur to remove noise
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=50)  # Enhance contrast
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding with more tuning
    return image

# Extract text from the processed image using Tesseract optimized for handwriting
def extract_text_from_image(image):
    pil_image = Image.fromarray(image)
    config = r'--oem 1 --psm 11'  # Optimized for sparse text with word segmentation
    text = pytesseract.image_to_string(pil_image, lang='eng', config=config)
    return text

# Streamlit UI
st.title("🧠 Handwritten Prescription Reader")

# File uploader to upload prescription image
uploaded_file = st.file_uploader("Upload a handwritten prescription image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Prescription", use_container_width=True)

    # Preprocess the image
    processed_image = preprocess_image(uploaded_file)

    # Extract text from the processed image
    text = extract_text_from_image(processed_image)

    # Display the extracted text
    st.subheader("📋 Extracted Prescription:")
    st.text_area("Text Extracted", text, height=300)

    # Optional: Allow users to download the text as a .txt file
    if st.button("Download Extracted Text"):
        st.download_button(label="Download Text", data=text, file_name="handwritten_prescription_text.txt", mime="text/plain")
=======
import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np



# Improved preprocessing for handwritten OCR
def preprocess_image(uploaded_file):
    image = np.array(Image.open(uploaded_file).convert("L"))  # Convert to grayscale
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Increase resolution
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian blur to remove noise
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=50)  # Enhance contrast
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding with more tuning
    return image

# Extract text from the processed image using Tesseract optimized for handwriting
def extract_text_from_image(image):
    pil_image = Image.fromarray(image)
    config = r'--oem 1 --psm 11'  # Optimized for sparse text with word segmentation
    text = pytesseract.image_to_string(pil_image, lang='eng', config=config)
    return text

# Streamlit UI
st.title("🧠 Handwritten Prescription Reader")

# File uploader to upload prescription image
uploaded_file = st.file_uploader("Upload a handwritten prescription image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Prescription", use_container_width=True)

    # Preprocess the image
    processed_image = preprocess_image(uploaded_file)

    # Extract text from the processed image
    text = extract_text_from_image(processed_image)

    # Display the extracted text
    st.subheader("📋 Extracted Prescription:")
    st.text_area("Text Extracted", text, height=300)

    # Optional: Allow users to download the text as a .txt file
    if st.button("Download Extracted Text"):
        st.download_button(label="Download Text", data=text, file_name="handwritten_prescription_text.txt", mime="text/plain")
>>>>>>> 61b9cae3cc7d0d0fa50966749b932e99799d9b21
