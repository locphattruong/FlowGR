
CUDA_VISIBLE_DEVICES=0 python gradio_app.py \
    --width 384 --height 512 \
    --gt_dir /home/ptruong/FlowGR/DATA/VITON-HD/test/cloth \
    --pred_dir /home/ptruong/FlowGR/new_result/img_ckpt_370000/512x384