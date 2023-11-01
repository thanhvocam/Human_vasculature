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
from pycocotools import _mask as coco_mask
import matplotlib.pyplot as plt
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
