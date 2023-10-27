import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import gdown

url = "https://drive.google.com/uc?id=1xZL5HjrVFsp4Zu6osnc_btM1NfiBcoop"
output = "best.pt"

def main():
    st.title("Human Vasculature Image Segmentation")  
    
    uploaded_file = st.file_uploader("Upload an image to get prediction", type=["jpg", "jpeg", "png", "tif"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image before predict', use_column_width=True)      

                        
        # Load YOLO model
        model = YOLO("best.pt")            
        # Perform object detection
        history = model.predict(image)[0]
        image_after_pred = history.plot()
        
        # Save and Display the image
        save_path = 'data_cus/result/image_pred.png'
        cv2.imwrite(save_path, image_after_pred)
        
        st.image(image_after_pred, caption="Segmentation Objects", use_column_width=True)
        st.success("Image saved successfully.")

if __name__ == "__main__":
    main()