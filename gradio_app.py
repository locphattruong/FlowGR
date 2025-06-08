import argparse
import gradio as gr
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
import pyiqa
from torch.utils.data import DataLoader

from eval import VitonhdEvalDataset
import os


class ImageEvaluator:
    def __init__(self, gt_dir, pred_dir, width, height):
        self.dataset = VitonhdEvalDataset(gt_dir, pred_dir, resize_to=(height, width))
        self.dataloader = iter(DataLoader(self.dataset, batch_size=1, shuffle=True))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gt_dir = gt_dir
        self.metrics = [
            pyiqa.create_metric("ssim").to(self.device),
            pyiqa.create_metric("ms_ssim").to(self.device),
            pyiqa.create_metric("cw_ssim").to(self.device),
            pyiqa.create_metric("lpips").to(self.device),
            pyiqa.create_metric("dists").to(self.device),
        ]
        self.metric_names = [
            "↑ SSIM",
            "↑ MS-SSIM",
            "↑ CW-SSIM",
            "↓ LPIPS",
            "↓ DISTS",
        ]

    @torch.no_grad()
    def next_sample(self):
        try:
            (gt, pred), gt_name = next(self.dataloader)
            gt = gt.to(self.device)
            pred = pred.to(self.device)

            # Construct paths for person and binary mask
            cloth = gt_name[0]
            person_path = cloth.replace("cloth", "image")
            mask_path = cloth.replace("cloth", "agnostic-mask-bin")

            # Load side images
            person_img = Image.open(person_path).convert("RGB").resize((256, 341))
            mask_img = Image.open(mask_path).convert("RGB").resize((256, 341))

            # Convert tensors to PIL for display
            to_pil = transforms.ToPILImage()
            gt_img = to_pil(gt.squeeze().cpu()).resize((768, 1024))
            pred_img = to_pil(pred.squeeze().cpu()).resize((768, 1024))

            results = {
                name: f"{metric(pred, gt).item():.4f}"
                for name, metric in zip(self.metric_names, self.metrics)
            }

            return gt_img, pred_img, person_img, mask_img, results
        except StopIteration:
            return None, None, None, None, {"Done": "No more samples"}


def build_interface(gt_dir, pred_dir, width, height):
    evaluator = ImageEvaluator(gt_dir, pred_dir, width, height)

    def update():
        gt_img, pred_img, person_img, mask_img, metrics = evaluator.next_sample()
        return gt_img, person_img, mask_img, pred_img, "\n".join(f"{k}: {v}" for k, v in metrics.items())

    with gr.Blocks() as demo:
        gr.Markdown("## Image Quality Evaluation Viewer")
        with gr.Row():
            with gr.Column():
                gt_img = gr.Image(label="Ground Truth", type="pil", height=512)
                with gr.Row():
                    person_img = gr.Image(label="Person", type="pil", width=128, height=170)
                    mask_img = gr.Image(label="Mask", type="pil", width=128, height=170)
            pred_img = gr.Image(label="Prediction", type="pil", height=512)
        metrics_box = gr.Textbox(label="Metrics", lines=6)
        next_btn = gr.Button("Next")
        next_btn.click(fn=update, outputs=[gt_img, person_img, mask_img, pred_img, metrics_box])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="Path to ground truth images")
    parser.add_argument("--pred_dir", required=True, help="Path to predicted images")
    parser.add_argument("--width", required=True, help="")
    parser.add_argument("--height", required=True, help="")
    args = parser.parse_args()

    demo = build_interface(args.gt_dir, args.pred_dir, int(args.width), int(args.height))
    demo.launch()
