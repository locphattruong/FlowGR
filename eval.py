import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
from cleanfid import fid
from DISTS_pytorch import DISTS
import pyiqa

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class VitonhdEvalDataset(Dataset):
    def __init__(self, gt_dir: str, pred_dir: str, resize_to: tuple[int, int] = (1024, 768)):
        logging.info(f"Predictions dir: {pred_dir}.")

        if not check_directory_contents(gt_dir, pred_dir):
            logging.warning("Proceeding with the available predictions despite mismatch in contents.")

        self.pred_files = sorted(glob(f"{pred_dir}/*.[jp][pn]g"))
        self.gt_files = [
            os.path.join(gt_dir, os.path.basename(f).replace("_pred", ""))
            for f in self.pred_files
        ]

        self.gt_transform = None
        self.pred_transform = None
        print(resize_to)
        self.set_transforms(resize_to)

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt = self.gt_transform(self.pil_to_tensor(self.gt_files[idx]))
        pred = self.pred_transform(self.pil_to_tensor(self.pred_files[idx]))
        return (gt, pred), self.gt_files[idx]

    def pil_to_tensor(self, path):
        if isinstance(path, str):
            image = Image.open(path).convert("RGB")
            return pil_to_tensor(image)
        elif isinstance(path, Image.Image):
            return pil_to_tensor(path.convert("RGB"))
        else:
            raise TypeError(f"Unsupported type for path: {type(path)}. Expected str or PIL Image.")
        

    def set_transforms(self, resize_to):
        h, w = self.pil_to_tensor(self.pred_files[0]).shape[1:]

        self.gt_transform = transforms.Compose([
            transforms.ToDtype(dtype=torch.float32, scale=True),
            transforms.Resize(resize_to)
        ])

        if resize_to == (1024, 768):
            if (h, w) in [(256, 256), (256, 176)]:
                logging.warning(f"Resizing predictions ({h}x{w}) to 1024x768.")
                self.pred_transform = transforms.Compose([
                    transforms.ToDtype(dtype=torch.float32, scale=True),
                    transforms.Resize(resize_to)
                ])
            elif (h, w) in [(512, 512), (128, 128)]:
                logging.warning(f"Resizing predictions ({h}x{w}) to 1024x1024 and cropping to 1024x768.")
                self.pred_transform = transforms.Compose([
                    transforms.ToDtype(dtype=torch.float32, scale=True),
                    transforms.Resize((1024, 1024)),
                    transforms.CenterCrop(resize_to)
                ])
            elif (h, w) == (1024, 1024):
                logging.warning(f"Cropping predictions ({h}x{w}) to 1024x768.")
                self.pred_transform = transforms.Compose([
                    transforms.ToDtype(dtype=torch.float32, scale=True),
                    transforms.CenterCrop(resize_to)
                ])
            elif (h, w) == (1024, 768):
                self.pred_transform = transforms.ToDtype(dtype=torch.float32, scale=True)
            elif (h, w) == (512, 384):
                self.pred_transform = transforms.Compose([
                    transforms.ToDtype(dtype=torch.float32, scale=True),
                    transforms.Resize(resize_to)
                ])
            else:
                raise ValueError(f"Unexpected image size: ({h}, {w})")

        elif resize_to == (341, 256) or resize_to == (512, 384):
            self.pred_transform = transforms.Compose([
                transforms.ToDtype(dtype=torch.float32, scale=True),
                transforms.Resize(resize_to)
            ])
        else:
            raise ValueError(f"Unsupported resize dimensions: {resize_to}")


def check_directory_contents(gt_dir: str, pred_dir: str) -> bool:
    num_gt_files = len(os.listdir(gt_dir))
    num_pred_files = len(os.listdir(pred_dir))

    if num_gt_files != num_pred_files:
        logging.warning(f"Mismatch in directory contents:\n - GT: {num_gt_files} files\n - Pred: {num_pred_files} files")
        return False
    return True


def print_results(metric_names: list[str], metric_values: list[float], source: str = None):
    if source:
        logging.info(source)
    print("   Metric   |   Value  ")
    print("------------|----------")
    for name, value in zip(metric_names, metric_values):
        print(f"{name:<11} | {value:.4f}")
    print("-----------------------")


class PYIQAEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics, self.metric_names = self._initialize_metrics()
        self._reset_state()

    def _initialize_metrics(self):
        metrics = [
            pyiqa.create_metric("ssim"),
            pyiqa.create_metric("ms_ssim"),
            pyiqa.create_metric("cw_ssim"),
            pyiqa.create_metric("lpips")
        ]
        names = ["↑ SSIM", "↑ MS-SSIM", "↑ CW-SSIM", "↓ LPIPS"]
        return [m.to(self.device) for m in metrics], names

    def _reset_state(self):
        self.metric_values = torch.zeros(len(self.metrics), device=self.device)
        self.total = 0

    def update(self, gt, pred):
        for i, metric in enumerate(self.metrics):
            self.metric_values[i] += metric(gt, pred).sum()
        self.total += gt.size(0)

    def compute(self):
        return (self.metric_values / self.total).cpu().tolist()

    def reset(self):
        self._reset_state()


def compute_cleanfid(gt_dir, pred_dir):
    names = ["↓ FID", "↓ CLIP-FID", "↓ KID"]
    values = [
        fid.compute_fid(gt_dir, pred_dir, num_workers=8, verbose=False),
        fid.compute_fid(gt_dir, pred_dir, num_workers=8, verbose=False, model_name="clip_vit_b_32"),
        fid.compute_kid(gt_dir, pred_dir, num_workers=8, verbose=False)
    ]
    print_results(names, values, source="`clean-fid`")


@torch.no_grad()
def compute_dists(gt_dir, pred_dir, batch_size, num_workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = DISTS().to(device)
    dataset = VitonhdEvalDataset(gt_dir, pred_dir, resize_to=(341, 256))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    values = []
    for (gt_batch, pred_batch), _ in tqdm(dataloader, desc="Evaluating DISTS"):
        gt_batch, pred_batch = gt_batch.to(device), pred_batch.to(device)
        values.append(metric(gt_batch, pred_batch).mean().item())

    print_results(["↓ DISTS"], [np.mean(values)], source="`DISTS_pytorch`")


def main(gt_dir, pred_dir, width, height, batch_size=32, num_workers=4):
    evaluator = PYIQAEvaluator()
    dataset = VitonhdEvalDataset(gt_dir, pred_dir, resize_to=(height,width))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    for (gt_batch, pred_batch), _ in tqdm(dataloader, desc="Evaluating PYIQA"):
        gt_batch = gt_batch.to(evaluator.device)
        pred_batch = pred_batch.to(evaluator.device)
        evaluator.update(pred_batch, gt_batch)

    print_results(evaluator.metric_names, evaluator.compute(), source="`pyiqa`")
    compute_cleanfid(gt_dir, pred_dir)
    compute_dists(gt_dir, pred_dir, batch_size, num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="Path to ground-truth directory")
    parser.add_argument("--pred_dir", required=True, help="Path to predictions directory")
    parser.add_argument("--width", type=int, default=384, help="Width for resizing images")
    parser.add_argument("--height", type=int, default=512, help="Height for resizing images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()
    main(args.gt_dir, args.pred_dir, args.width, args.height ,args.batch_size, args.num_workers)
