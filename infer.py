import cv2
from ultralytics import YOLO


model = YOLO("HuBMAP/yolov8x-seg2/weights/best.pt")
history = model.predict("data_cus/test/72e40acccadf.tif")[0]
image = history.plot()

save_path = 'data_cus/result/image.png'
cv2.imwrite(save_path, image)
