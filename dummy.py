import streamlit as st
import os
import cv2
from ultralytics import YOLO

def main():
    st.title("YOLO Object Detection")
    
    # Load YOLO model
    model = YOLO("HuBMAP/yolov8x-seg2/weights/best.pt")
    
    # Select image from directory
    dirlist = os.listdir("data_cus/test")
    selected_image = st.selectbox("Select an image:", dirlist)
    
    # Perform object detection
    history = model.predict("data_cus/test/" + selected_image)[0]
    image = history.plot()
    
    # Save and display the image
    save_path = 'data_cus/result/image.png'
    cv2.imwrite(save_path, image)
    st.image(image, caption="Detected Objects", use_column_width=True)
    st.success("Image saved successfully.")

if __name__ == "__main__":
    main()