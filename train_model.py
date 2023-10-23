from pathlib import Path
import json
import torchvision.transforms as T
import os
import numpy as np
import shutil
import random
from ultralytics import YOLO
from PIL import Image
import torch
import zlib
import base64
from pycocotools import _mask as coco_mask
import matplotlib.pyplot as plt
import pandas as pd
import cv2



class DatasetHub:
    def __init__(self, img_path: Path, labels_path: Path):
        self.img_path = img_path
        self.labels_path = labels_path
        self.type_class = {"blood_vessel": 0, "glomerulus": 1, "unsure": 2}
        self.image_files = os.listdir(img_path)
        self.length = len(self.image_files)

        self.dataset_dirpath = os.path.join(os.getcwd(), "data_cus")
        os.makedirs(self.dataset_dirpath, exist_ok=True)

        self.train_dirpath = os.path.join(self.dataset_dirpath, "train")
        os.makedirs(self.train_dirpath, exist_ok=True)
        self.img_train = os.path.join(self.train_dirpath, "images")
        os.makedirs(self.img_train, exist_ok=True)
        self.labels_train = os.path.join(self.train_dirpath, "labels")
        os.makedirs(self.labels_train, exist_ok=True)

        self.val_dirpath = os.path.join(self.dataset_dirpath, "val")
        os.makedirs(self.val_dirpath, exist_ok=True)
        self.img_val = os.path.join(self.val_dirpath, "images")
        os.makedirs(self.img_val, exist_ok=True)
        self.labels_val = os.path.join(self.val_dirpath, "labels")
        os.makedirs(self.labels_val, exist_ok=True)

        self.config_path = os.path.join(self.dataset_dirpath, "coco.yaml")
        os.makedirs(self.labels_val, exist_ok=True)

    def __len__(self):
        return len(self.image_files)

    def transform_img(self):
        image = Image.open(self.img_path)
        image_array = np.array(image)
        tensor = torch.from_numpy(image_array)
        return tensor

    def get_labels(self, labels_path):
        with open(labels_path, "r") as jsonl_file:
            for line in jsonl_file:
                data = json.loads(line)
                id = data["id"]
                annotations = data["annotations"]
                file_name = id + ".txt"
                with open(f"labels_all_data/{file_name}", "w") as file:
                    for annotation in annotations:
                        type_cl = self.type_class[annotation["type"]]
                        coordi = np.array(annotation["coordinates"]) / 512.0
                        flattened_coordinates = [
                            str(coord)
                            for sublist in coordi
                            for subsublist in sublist
                            for coord in subsublist
                        ]
                        coordinates = " ".join(flattened_coordinates)
                        file.write(f"{type_cl} {coordinates}\n")

    def copy_data(self, dataset_dirpath: Path, labels_path: Path):
        all_images = os.listdir(dataset_dirpath)
        all_labels = os.listdir(labels_path)

        img_train = round(self.length * 0.8)
        img_val = self.length - img_train

        random_images_train = random.sample(all_images, img_train)
        random_images_val = random.sample(all_images, img_val)

        for image in random_images_train:
            image_path = os.path.join(dataset_dirpath, image)
            shutil.copy(image_path, self.img_train)

        for image in random_images_val:
            image_path = os.path.join(dataset_dirpath, image)
            shutil.copy(image_path, self.img_val)

        for img in random_images_train:
            for label in all_labels:
                if label[:12] == img[:12]:
                    label_path = os.path.join(labels_path, label)
                    shutil.copy(label_path, self.labels_train)

        for img in random_images_val:
            for label in all_labels:
                if label[:12] == img[:12]:
                    label_path = os.path.join(labels_path, label)
                    shutil.copy(label_path, self.labels_val)

    def get_config(self):
        with open(self.config_path, mode="w") as file:
            file.write(
                f"train: {self.train_dirpath}\n val: {self.val_dirpath}\n nc: {[i for i in self.type_class]}"
            )


def main():
    model = YOLO("yolov8x-seg.pt")
    model.train(
        # Project
        project="HuBMAP",
        name="yolov8x-seg",

        # Random Seed parameters
        deterministic=True,
        seed=43,

        # Data & model parameters
        data="data_cus/coco.yaml", 
        save=True,
        save_period=1,
        pretrained=True,
        imgsz=512,

        # Training parameters
        epochs=20,
        batch=16,
        workers=8,
        val=True,
        device=0,

        # Optimization parameters
        lr0=0.018,
        patience=3,
        optimizer="SGD",
        momentum=0.947,
        weight_decay=0.0005,
        close_mosaic=3,
    )

if __name__ == '__main__':
    main()

dirlist = os.listdir("data_cus/val/images")
print(dirlist[:5])

model = YOLO("HuBMAP/yolov8x-seg22/weights/best.pt")   
history = model.predict("data_cus/test/72e40acccadf.tif")[0]
image = history.plot()

save_path = 'data_cus/result/image.png'
cv2.imwrite(save_path, image)
print("Hình ảnh đã được lưu thành công dưới định dạng PNG.")

plt.imshow(image)
plt.show()


F1_curve = Image.open("HuBMAP/yolov8x-seg22/results.png")
plt.figure(figsize=(15,20))
plt.imshow(F1_curve)
plt.show()

P_curve = Image.open("HuBMAP/yolov8x-seg22/train_batch1.jpg")
plt.imshow(P_curve)
plt.show()

class BestYolo:
    def __init__(self, conf: float = 0.05):
        self.model_path = "HuBMAP/yolov8x-seg23/weights/best.pt"
        self.model = self.get_model()
        self.conf = conf
    
    def get_model(self) -> YOLO:
        return YOLO(self.model_path)
    
    def __call__(self, source) -> list[dict]:
        sublist = []
        result = self.model(source)[0]
        if result.masks:
            for i in range(len(result.masks.data)):
                conf = round(float(result.boxes.conf[i]), 2)
                mask = np.expand_dims(result.masks.data[i].cpu().numpy(), axis=0).transpose(1,2,0)
            
                if int(result.boxes.cls[i]) == 0 and conf >= self.conf:
                    sublist.append({"mask": mask, "confidence": conf})
                else:
                    continue
            return sublist
        else:
            return None


__TEST_PATH = "data_cus/test"
model = BestYolo()