import cv2
import numpy as np
import streamlit as st
from PIL import Image

def adjust_contrast(image, alpha):
    """Adjust the contrast of the image."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def adjust_brightness(image, beta):
    """Adjust the brightness of the image."""
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)

def smooth_image(image):
    """Smooth the image using Gaussian Blur."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def sharpen_image(image):
    """Sharpen the image using a kernel."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_mask(image):
    """Apply a circular mask to the image."""
    mask = np.zeros(image.shape[:2], dtype="uint8")
    center = (image.shape[1] // 2, image.shape[0] // 2)
    radius = min(center[0], center[1], image.shape[1] - center[0], image.shape[0] - center[1]) // 2
    cv2.circle(mask, center, radius, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)

# Streamlit UI
st.title("Image Enhancement App")
st.write("Upload an image and apply filters to enhance it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Original Image", use_column_width=True)

    # Contrast Adjustment
    contrast_image = adjust_contrast(image, alpha=1.5)
    st.image(contrast_image, caption="Contrast Adjusted", use_column_width=True)

    # Brightness Adjustment
    brightness_image = adjust_brightness(image, beta=50)
    st.image(brightness_image, caption="Brightness Adjusted", use_column_width=True)

    # Smoothing
    smoothed_image = smooth_image(image)
    st.image(smoothed_image, caption="Smoothed Image", use_column_width=True)

    # Sharpening
    sharpened_image = sharpen_image(image)
    st.image(sharpened_image, caption="Sharpened Image", use_column_width=True)

    # Masking
    masked_image = apply_mask(image)
    st.image(masked_image, caption="Masked Image", use_column_width=True)
