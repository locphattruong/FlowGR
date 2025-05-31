import os
import random
import json
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from diffusers.image_processor import VaeImageProcessor

from typing import Literal, Tuple
import torch.utils.data as data
import torchvision.transforms.functional as TF

class VitonHDDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
    ):
        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size


        self.norm = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.transform2D = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.toTensor = transforms.ToTensor()
        
        self.to_pil = transforms.Compose([
                        transforms.Normalize(mean=[-1], std=[2]),  # [-1,1] → [0,1]
                        transforms.ToPILImage()
                    ])

        with open(
            os.path.join(dataroot_path, phase, "vitonhd_" + phase + "_tagged.json"), "r"
        ) as file1:
            data1 = json.load(file1)

        annotation_list = [
            # "colors",
            # "textures",
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
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        self.clip_processor = CLIPImageProcessor()
        self.mask_processor = VaeImageProcessor()
        
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        # subject_txt = self.txt_preprocess['train']("shirt")
        if c_name in self.annotation_pair:
            cloth_annotation = self.annotation_pair[c_name]
        else:
            cloth_annotation = "shirts"
        
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name)).resize((self.width,self.height))
        cloth = self.transform(cloth)
        
        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width,self.height))

        image = self.transform(im_pil_big)
        # load parsing image

        mask = Image.open(os.path.join(self.dataroot, self.phase, "agnostic-mask-bin", im_name.replace('.jpg','.jpg'))).resize((self.width,self.height))
        crops_coords = self.mask_processor.get_crop_region(mask, image.shape[2], image.shape[1], pad=10)

        mask = self.toTensor(mask)
        mask = mask[:1]
        
        densepose_name = im_name
        densepose_map = Image.open(
            os.path.join(self.dataroot, self.phase, "image-densepose", densepose_name)
        ).resize((self.width,self.height))
        pose_img = self.toTensor(densepose_map)  # [-1,1]

        if self.phase == "train":
            if random.random() > 0.5:
                cloth = self.flip_transform(cloth)
                mask = self.flip_transform(mask)
                image = self.flip_transform(image)
                pose_img = self.flip_transform(pose_img)

            if random.random()>0.5:
                color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.5)
                fn_idx, b, c, s, h = transforms.ColorJitter.get_params(color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,color_jitter.hue)
                
                image = TF.adjust_contrast(image, c)
                image = TF.adjust_brightness(image, b)
                image = TF.adjust_hue(image, h)
                image = TF.adjust_saturation(image, s)

                cloth = TF.adjust_contrast(cloth, c)
                cloth = TF.adjust_brightness(cloth, b)
                cloth = TF.adjust_hue(cloth, h)
                cloth = TF.adjust_saturation(cloth, s)

              
            if random.random() > 0.5:
                scale_val = random.uniform(0.8, 1.2)
                # image = transforms.functional.affine(
                #     image, angle=0, translate=[0, 0], scale=scale_val, shear=0
                # )
                # mask = transforms.functional.affine(
                #     mask, angle=0, translate=[0, 0], scale=scale_val, shear=0
                # )
                # pose_img = transforms.functional.affine(
                #     pose_img, angle=0, translate=[0, 0], scale=scale_val, shear=0
                # )
                cloth = transforms.functional.affine(
                    cloth, angle=0, translate=[0, 0], scale=scale_val, shear=0
                )



            if random.random() > 0.5:
                shift_valx = random.uniform(-0.2, 0.2)
                shift_valy = random.uniform(-0.2, 0.2)
                # image = transforms.functional.affine(
                #     image,
                #     angle=0,
                #     translate=[shift_valx * image.shape[-1], shift_valy * image.shape[-2]],
                #     scale=1,
                #     shear=0,
                # )
                # mask = transforms.functional.affine(
                #     mask,
                #     angle=0,
                #     translate=[shift_valx * mask.shape[-1], shift_valy * mask.shape[-2]],
                #     scale=1,
                #     shear=0,
                # )
                # pose_img = transforms.functional.affine(
                #     pose_img,
                #     angle=0,
                #     translate=[
                #         shift_valx * pose_img.shape[-1],
                #         shift_valy * pose_img.shape[-2],
                #     ],
                #     scale=1,
                #     shear=0,
                # )
                cloth = transforms.functional.affine(
                    cloth,
                    angle=0,
                    translate=[
                        shift_valx * pose_img.shape[-1],
                        shift_valy * pose_img.shape[-2],
                    ],
                    scale=1,
                    shear=0,
                )

        mask = 1-mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        im_mask = image * mask
        pose_img =  self.norm(pose_img)

        cloth_trim = image * (1-mask)
        cloth_trim = self.clip_processor(images=self.to_pil(cloth_trim).crop(crops_coords), return_tensors="pt").pixel_values
        
        result = {}
        result["c_name"] = c_name
        result["image"] = image #VAE condition
        result["cloth_trim"] = cloth_trim #IP-Adapter
        result["cloth_pure"] = cloth #VAE generator
        result["inpaint_mask"] = 1-mask #VAE condition
        result["im_mask"] = im_mask #VAE condition
        result["caption"] = "model is wearing " + cloth_annotation
        result["caption_cloth"] = "a photo of " + cloth_annotation
        result["annotation"] = cloth_annotation
        result["pose_img"] = pose_img #VAE condition (later phase)

        return result

    def __len__(self):
        return len(self.im_names)