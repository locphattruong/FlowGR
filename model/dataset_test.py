import os
import random
import json
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor

from typing import Literal, Tuple
import torch.utils.data as data
import torchvision.transforms.functional as TF
class VitonHDTestDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
    ):
        super(VitonHDTestDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.toTensor = transforms.ToTensor()

        with open(
            os.path.join(dataroot_path, phase, "vitonhd_" + phase + "_tagged.json"), "r"
        ) as file1:
            data1 = json.load(file1)

        annotation_list = [
            "sleeveLength",
            "neckLine",
            "item",
        ]

        self.annotation_pair = {}
        for k, v in data1.items():
            for elem in v:
                annotation_str = ""
                for template in annotation_list:
                    for tag in elem["tag_info"]:
                        if (
                            tag["tag_name"] == template
                            and tag["tag_category"] is not None
                        ):
                            annotation_str += tag["tag_category"]
                            annotation_str += " "
                self.annotation_pair[elem["file_name"]] = annotation_str

        self.order = order
        self.toTensor = transforms.ToTensor()

        im_names = []
        c_names = []
        dataroot_names = []


        if phase == "train":
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")

        with open(filename, "r") as f:
            for line in f.readlines():
                if phase == "train":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == "paired":
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot_path)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.clip_processor = CLIPImageProcessor()
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        if c_name in self.annotation_pair:
            cloth_annotation = self.annotation_pair[c_name]
        else:
            cloth_annotation = "shirts"
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name)).resize((self.width,self.height))

        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width,self.height))
        image = self.transform(im_pil_big)

        mask = Image.open(os.path.join(self.dataroot, self.phase, "agnostic-mask-bin", im_name.replace('.jpg','.jpg'))).resize((self.width,self.height))
        mask = self.toTensor(mask)
        mask = mask[:1]
        mask = 1-mask
        im_mask = image * mask
 
        pose_img = Image.open(
            os.path.join(self.dataroot, self.phase, "image-densepose", im_name)
        ).resize((self.width,self.height))
        pose_img = self.transform(pose_img)  # [-1,1]
 
        result = {}               # an instance (same file name): <cloth, person wearing that cloth>
        result["c_name"] = c_name #unpaired: im name of a person and c name of another person
        result["im_name"] = im_name #paired: c name and im name are the same instance  
        result["image"] = image
        result["cloth_pure"] = self.transform(cloth)
        result["cloth"] = self.clip_processor(images=cloth, return_tensors="pt").pixel_values
        result["inpaint_mask"] =1-mask
        result["im_mask"] = im_mask
        result["caption_cloth"] = "a photo of " + cloth_annotation
        result["caption"] = "model is wearing a " + cloth_annotation
        result["pose_img"] = pose_img

        return result

    def __len__(self):
        # model images + cloth image
        return len(self.im_names)