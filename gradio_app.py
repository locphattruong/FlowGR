import argparse
import gradio as gr
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
import pyiqa
from torch.utils.data import DataLoader

from eval import VitonhdEvalDataset


class ImageEvaluator:
    def __init__(self, gt_dir, pred_dir):
        self.dataset = VitonhdEvalDataset(gt_dir, pred_dir, resize_to=(512, 384))
        self.dataloader = iter(DataLoader(self.dataset, batch_size=1, shuffle=True))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
            cloth = gt_name
            person = gt_name.replace("cloth", "image")
            mask_bin = gt_name.replace("cloth", "agnostic-mask-bin")
            results = {
                name: f"{metric(pred, gt).item():.4f}"
                for name, metric in zip(self.metric_names, self.metrics)
            }

            # Convert tensors to PIL for display
            to_pil = transforms.ToPILImage()
            gt_img = to_pil(gt.squeeze().cpu()).resize((768, 1024))
            pred_img = to_pil(pred.squeeze().cpu()).resize((768, 1024))
            return gt_img, pred_img, results
        except StopIteration:
            return None, None, {"Done": "No more samples"}


def build_interface(gt_dir, pred_dir):
    evaluator = ImageEvaluator(gt_dir, pred_dir)

    def update():
        gt_img, pred_img, metrics = evaluator.next_sample()
        return gt_img, pred_img, "\n".join(f"{k}: {v}" for k, v in metrics.items())

    with gr.Blocks() as demo:
        gr.Markdown("## Image Quality Evaluation Viewer")
        with gr.Row():
            gt_img = gr.Image(label="Ground Truth", type="pil")
            pred_img = gr.Image(label="Prediction", type="pil")
        metrics_box = gr.Textbox(label="Metrics", lines=6)
        next_btn = gr.Button("Next")
        next_btn.click(fn=update, outputs=[gt_img, pred_img, metrics_box])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="Path to ground truth images")
    parser.add_argument("--pred_dir", required=True, help="Path to predicted images")
    args = parser.parse_args()

    demo = build_interface(args.gt_dir, args.pred_dir)
    demo.launch()
