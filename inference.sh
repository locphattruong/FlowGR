CUDA_VISIBLE_DEVICES=1 accelerate launch inference_tryoff.py --gradient_checkpointing --use_8bit_adam \
    --pretrained_garmentnet_path=/home/ptruong/FlowGR/result/20250603-211028/checkpoint-370000 \
    --output_dir=/home/ptruong/FlowGR/new_result/img_ckpt_370000/1024x768 --data_dir=/home/ptruong/FlowGR/DATA/VITON-HD \
    --width 768 --height 1024 --test_batch_size=4 --mixed_precision bf16