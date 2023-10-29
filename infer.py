import cv2
from ultralytics import YOLO


model = YOLO("best.pt")
history = model.predict("72e40acccadf.tif")[0]
image = history.plot()

save_path = 'after_pred.png'
cv2.imwrite(save_path, image)
