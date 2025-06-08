CUDA_VISIBLE_DEVICES=1 python eval.py \
    --gt_dir /home/ptruong/FlowGR/DATA/VITON-HD/test/cloth \
    --pred_dir /home/ptruong/FlowGR/new_result/img_ckpt_370000/512x384 \
    --width 384 --height 512 \
    --batch_size 16 --num_workers 8
