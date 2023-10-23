from itertools import chain
import json
import os
import shutil
from tqdm import tqdm
from colorama import Fore
import yaml
import numpy as np
import shutil

import os
import sys
from colorama import Fore
from pycocotools import _mask as coco_mask 
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import base64
import numpy as np
import torch
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import pandas as pd
import torchvision.transforms as T




class COCODataset:
    def __init__(self, images_dirpath: str, annotations_filepath: str, length: int = 1633):
        self.train_size = None
        self.val_size = None
        self.length = length
        self.classes = None
        self.labels_counter = None
        self.normalize = None
        
        self.images_dirpath = images_dirpath
        self.annotations_filepath = annotations_filepath
        self.dataset_dirpath = os.path.join(os.getcwd(), "dataset")
        self.train_dirpath =  os.path.join(self.dataset_dirpath, "train")
        self.val_dirpath =  os.path.join(self.dataset_dirpath, "val")
        self.config_path = os.path.join(self.dataset_dirpath, "coco.yaml")

        self.samples = self.parse_jsonl(annotations_filepath)
        self.classes_dict = {
            "blood_vessel": 0,
            "glomerulus": 1,
            "unsure": 2,
        }

    def __prepare_dirs(self) -> None:
        if not os.path.exists(self.dataset_dirpath):
            os.makedirs(os.path.join(self.train_dirpath, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.train_dirpath, "labels"), exist_ok=True)
            os.makedirs(os.path.join(self.val_dirpath, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.val_dirpath, "labels"), exist_ok=True)
        else:
            raise RuntimeError("Dataset already exists!")

    def __define_splitratio(self) -> None:
        self.train_size = round(self.length * self.train_size)
        self.val_size = self.length - self.train_size
        assert self.train_size + self.val_size == self.length

    def parse_jsonl(self, path: str) -> list[dict]:
        with open(path, 'r') as json_file:
            jsonl_samples = [
                json.loads(line)
                for line in tqdm(
                    json_file, desc="Processing polygons", total=self.length
                )
            ]
        return jsonl_samples

    def __define_paths(self, i: int) -> dict:
        data_path = self.val_dirpath
        if i < self.train_size:
            data_path = self.train_dirpath
        return {
            "images": os.path.join(data_path, "images"),
            "labels": os.path.join(data_path, "labels")
        }

    @staticmethod
    def __get_label_path(paths_dict: dict, identifier: str) -> str:
        return os.path.join(
            paths_dict["labels"],
            f"{identifier}.txt"
        )

    @staticmethod
    def __get_image_path(paths_dict: dict, identifier: str) -> str:
        return os.path.join(
            paths_dict["images"],
            f"{identifier}.tif"
        )

    def __copy_image(self, dst_path: str, identifier: str) -> str:
        shutil.copyfile(
            os.path.join(self.images_dirpath, f"{identifier}.tif"),
            dst_path
        )

    def __copy_label(self, annotations: list, dst_path: str) -> None:
        with open(dst_path, "w") as file:
            for annotation in annotations:
                coordinates = annotation["coordinates"][0]
                label = self.classes_dict[annotation["type"]]
                if label in self.classes:
                    if coordinates:
                        if self.normalize:
                            coordinates = np.array(coordinates) / 512.0
                        coordinates = " ".join(map(str, chain(*coordinates)))
                        file.write(f"{label} {coordinates}\n")
                        self.labels_counter += 1

    def __splitfolders(self):
        for i, line in tqdm(
                enumerate(self.samples),
                desc="Dataset creation", total=self.length
        ):
            self.labels_counter = 0
            identifier = line["id"]
            annotations = line["annotations"]
            paths_dict = self.__define_paths(i)

            dst_image_path = self.__get_image_path(paths_dict, identifier)
            dst_label_path = self.__get_label_path(paths_dict, identifier)

            self.__copy_image(dst_image_path, identifier)
            self.__copy_label(annotations, dst_label_path)

            if self.labels_counter == 0:
                os.remove(dst_image_path)
                os.remove(dst_label_path)

    def __count_dataset(self) -> dict:
        train_images = len(os.listdir(os.path.join(self.train_dirpath, "images")))
        train_labels = len(os.listdir(os.path.join(self.train_dirpath, "labels")))
        val_images = len(os.listdir(os.path.join(self.val_dirpath, "images")))
        val_labels = len(os.listdir(os.path.join(self.val_dirpath, "labels")))
        return {
            "train_images": train_images,
            "train_labels": train_labels,
            "val_images": val_images,
            "val_labels": val_labels
        }

    @staticmethod
    def __check_sanity(count_dict: dict) -> None:
        assert count_dict["train_images"] == count_dict["train_labels"]
        assert count_dict["val_images"] == count_dict["val_labels"]

    def __finalizing(self, count_dict: dict) -> None:
        assert os.path.exists(self.dataset_dirpath)

        example_structure = [
            "dataset",
            "train", "labels", "images",
            "val", "labels", "images"
        ]

        dir_bone = (
            dirname.split("/")[-1]
            for dirname, _, filenames in os.walk(self.dataset_dirpath)
            if dirname.split("/")[-1] in example_structure
        )

        try:
            print("\n~ HuBMAP Dataset Structure ~\n")
            print(
            f"""
          ├── {next(dir_bone)}
          │   │
          │   ├── {next(dir_bone)}
          │   │   └── {next(dir_bone)}
          │   │   └── {next(dir_bone)}
          │   │
          │   ├── {next(dir_bone)}
          │   │   └── {next(dir_bone)}
          │   │   └── {next(dir_bone)}
            """
            )
        except StopIteration as e:
            print(e)
        else:
            print(Fore.GREEN + "-> Success")
            print(Fore.GREEN + f"Train dataset: {count_dict['train_images']}\nVal dataset: {count_dict['val_images']}")

    def get_config(self) ->dict:
        names = ["blood_vessel", "glomerulus", "unsure"]
        return {
            "train": str(self.train_dirpath),
            "val": str(self.val_dirpath),
            "names": [names[i] for i in self.classes]
        }

    @staticmethod
    def display_config(config: dict) -> None:
        print(Fore.BLACK + "\n~ HuBMAP Config Structure ~\n")
        print(
        f"""
      │   │
      │   ├── train
      │   │   └── {config['train']}/images
      │   │
      │   │
      │   ├── val
      │   │   └── {config['val']}/images
      │   │
      │   │
      │   ├── names
      │   │   └── {' '.join(config['names'])}
        """
        )
        print(Fore.GREEN + "-> Success")
        print(Fore.GREEN + f"Number of classes: {len(config['names'])}"
                           f"\nClasses: {' '.join(config['names'])}" 
              )

    def write_config(self, config: dict) -> None:
        with open(self.config_path, mode="w") as f:
            yaml.safe_dump(stream=f, data=config)

    def __call__(self, train_size: float,
                 classes: list[int],
                 make_config: bool = True,
                 normalize: bool = True
                ) -> None:
        
        self.train_size = train_size
        self.classes = classes
        self.normalize = normalize
        
        self.__define_splitratio()
        self.__prepare_dirs()
        self.__splitfolders()
        count_dict = self.__count_dataset()
        self.__check_sanity(count_dict)
        self.__finalizing(count_dict)
        
        if make_config:
            config = self.get_config()
            self.write_config(config)
            self.display_config(config)

coco = COCODataset(
    annotations_filepath="data/polygons.jsonl",
    images_dirpath="data/train",
)
coco(train_size=0.80, classes=[0, 1, 2])


class SetupPipline:
    def __init__(self, display: bool = True):
        self.pycocotools = self.__pycocotools()
        self.ultralytics = self.__ultralytics()
        
    @staticmethod
    def __ultralytics() -> str:
        sys.path.append("/kaggle/input/hubmap-tools-ultralytics-and-pycocotools/ultralytics/ultralytics") 
        return "successfully"
        
    @staticmethod
    def __pycocotools() -> str:
        if not os.path.exists("/kaggle/working/packages"):
            shutil.copytree("/kaggle/input/hubmap-tools-ultralytics-and-pycocotools/pycocotools/pycocotools", "/kaggle/working/packages")
            os.chdir("/kaggle/working/packages/pycocotools-2.0.6/")
            os.system("python setup.py install")
            os.system("pip install . --no-index --find-links /kaggle/working/packages/")
            os.chdir("/kaggle/working")
            return "successfully"
    
    def display(self) -> None:
        print(Fore.GREEN+f"\nPycocotools was installed {self.pycocotools}")
        print(f"Ultralytics was installed {self.ultralytics}"+Fore.WHITE)

pipline = SetupPipline()

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
        data="dataset/coco.yaml", 
        save=True,
        save_period=5,
        pretrained=True,
        imgsz=512,

        # Training parameters
        epochs=20,
        batch=4,
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


dirlist = os.listdir("dataset/val/images")
print(dirlist[:5])

model = YOLO("/kaggle/working/HuBMAP/yolov8x-seg/weights/best.pt")
history = model.predict("../working/dataset/val/images/ed6a92a9410c.tif")[0]
image = history.plot()
plt.imshow(image)
plt.show()


F1_curve = Image.open("/kaggle/working/HuBMAP/yolov8x-seg/results.png")
plt.figure(figsize=(15,20))
plt.imshow(F1_curve)
plt.show()

P_curve = Image.open("/kaggle/working/HuBMAP/yolov8x-seg/train_batch5561.jpg")
plt.imshow(P_curve)
plt.show()

class EncodeBinaryMask:
    @staticmethod
    def __checking_mask(mask: np.ndarray) -> np.ndarray:
        if mask.dtype != np.bool:
            raise ValueError(
                "expects a binary mask, received dtype == %s" %
                mask.dtype
            )
        return mask

    @staticmethod
    def __convert_mask(mask: np.ndarray):
        mask_to_encode = mask.astype(np.uint8)
        mask_to_encode = np.asfortranarray(mask_to_encode)
        return mask_to_encode

    @staticmethod
    def __compress_encode(encoded_mask) -> t.Text:
        binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
        base64_str = base64.b64encode(binary_str)
        return base64_str

    def __call__(self, mask: np.ndarray) -> t.Text:
        mask = self.__checking_mask(mask)
        mask_to_encode = self.__convert_mask(mask)
        encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]
        base64_str = self.__compress_encode(encoded_mask)
        return base64_str
    
class Submission:
    def __init__(self, dirpath: str, model: torch.nn.Module):
        self.__eval_transforms = self.get_transforms()
        self.__model = model
        
        self.__encoder = EncodeBinaryMask()
        self.__dirpath = dirpath
        self.__filenames = os.listdir(dirpath)
        self.height = 512
        self.width = 512
        
        self.__submission_dict = {
            "id": [],
            "height": [],
            "width": [],
            "prediction_string": []
        }
        
        self.submission = None
    
    @staticmethod
    def get_transforms():
        return T.Compose([
            T.ToTensor(),
            T.Resize(size=(512, 512)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.__filenames)

    def __get_columns(self) -> None:
        for filename in self.__filenames:
            path = self.__get_image_path(filename)
            masks = self.__forward(path)
            identifier, height, width, prediction_string = self.__get_cells(filename, masks)
            self.__update_columns(identifier, height, width, prediction_string)

    def __update_columns(self, identifier: str, height: int, width: int, prediction_string: str) -> None:
        self.__submission_dict["id"].append(identifier)
        self.__submission_dict["height"].append(height)
        self.__submission_dict["width"].append(width)
        self.__submission_dict["prediction_string"].append(prediction_string)

    def __get_cells(self, filename: str, masks: list):
        prediction_string = ""
        prediction_string = self.__get_prediction_string(masks, prediction_string)
        identifier = filename.split(".")[0]
        return identifier, self.height, self.width, prediction_string

    def __get_prediction_string(self, masks: list, prediction_string: str) -> str:
        if masks:
            for outputs in masks:
                mask = outputs["mask"]
                mask = np.where(mask > 0.5, 1, 0).astype(np.bool)
                base64_str = self.__encoder(mask)
                confidence = outputs["confidence"]
                prediction_string += f"0 {confidence} {base64_str.decode('utf-8')} "
        else:
            return ""
        return prediction_string

    def __get_image_path(self, filename: str) -> str:
        return os.path.join(
            self.__dirpath, filename
        )

    def __get_image(self, path: str) -> torch.Tensor:
        image = Image.open(path)
        image = np.asarray(image)
        image = self.__eval_transforms(image)
        return image

    def __forward(self, image: torch.tensor) -> list:
        masks = self.__model(image) 
        return masks 

    def submit(self) -> None:
        if not self.submission:
            self.__get_columns()
            self.submission = pd.DataFrame(self.__submission_dict)
            self.submission = self.submission.set_index('id')
            self.submission.to_csv("submission.csv")

class BestYolo:
    def __init__(self, conf: float = 0.05):
        self.model_path = "/kaggle/working/HuBMAP/yolov8x-seg/weights/best.pt"
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


__TEST_PATH = "/kaggle/input/hubmap-hacking-the-human-vasculature/test"
model = BestYolo()
sub = Submission(dirpath=__TEST_PATH, model=model)
sub.submit()

sub.submission.head()