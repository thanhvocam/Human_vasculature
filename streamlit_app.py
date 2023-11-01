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
    col1, col2 = st.columns(2)
    with col1:
        # Choose img to test
        val_path = "val"
        image_files = os.listdir(val_path)
        selected_image = st.selectbox("Choose Image to see the Result before upload your Image", image_files)
        image_path = os.path.join(val_path, selected_image)
        image_val = Image.open(image_path)
        st.image(image_val, caption="Image_Test", use_column_width=True)
        
        start1 = time.time()
        # Load YOLO model
        model = YOLO(model_path)

        # Perform object detection
        history = model.predict(image_val)[0]
        image_val_pred = history.plot()
        
        # Save and Display the image
        save_path = 'data_cus/result/image_pred.png'
        cv2.imwrite(save_path, image_val_pred)
        end1 = time.time()
        process_pred1 = end1 - start1
        with st.spinner("Running"):
            time.sleep(process_pred1)
            st.image(image_val_pred, caption="Segmentation Objects", use_column_width=True)
            st.success("Image saved successfully.")
        
        
    with col2:
        uploaded_file = st.file_uploader("Upload an image to get prediction", type=["jpg", "jpeg", "png", "tif"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image before predict', use_column_width=True)      

            start2 = time.time()                
            # Load YOLO model
            model = YOLO(model_path)            
            # Perform object detection
            history = model.predict(image)[0]
            image_after_pred = history.plot()
            
            # Save and Display the image
            save_path = 'data_cus/result/image_pred.png'
            cv2.imwrite(save_path, image_after_pred)
            end2 = time.time()
            process_pred2 = end2 - start2
            with st.spinner("Running"):
                time.sleep(process_pred2)
                st.image(image_after_pred, caption="Segmentation Objects", use_column_width=True)
                st.success("Image saved successfully.")

if __name__ == "__main__":
    main()