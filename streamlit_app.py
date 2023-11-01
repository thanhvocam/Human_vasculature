import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import gdown
import os
import time

url = "https://drive.google.com/uc?id=1xZL5HjrVFsp4Zu6osnc_btM1NfiBcoop"
model_path = "best.pt"
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

def main():
    st.title("Human Vasculature Image Segmentation")
    
    # Choose img to test
    val_path = "val"
    image_files = os.listdir(val_path)
    st.header("Select an image to see the prediction or upload an image below")
    selected_image = st.selectbox("", image_files)
    image_path = os.path.join(val_path, selected_image)
    image = Image.open(image_path)
    st.image(image, caption="Test image", use_column_width=True)
    predict(model_path=model_path, image=image)

    st.header("Upload an image to get prediction")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "tif"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image before predict', use_column_width=True)      
        predict(model_path=model_path, image=image)

def predict(model_path, image):
    start = time.time()                
    # Load YOLO model
    model = YOLO(model_path)            
    # Perform object detection
    history = model.predict(image)[0]
    image_after_pred = history.plot()
    
    # Save and Display the image
    save_path = 'data_cus/result/image_pred.png'
    cv2.imwrite(save_path, image_after_pred)
    end = time.time()
    pred_time = end - start
    with st.spinner("Running"):
        time.sleep(pred_time)
        st.image(image_after_pred, caption="Prediction result", use_column_width=True)

if __name__ == "__main__":
    main()