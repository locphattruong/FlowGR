CUDA_VISIBLE_DEVICES=0 accelerate launch train_tryoff.py --gradient_checkpointing --use_8bit_adam \
    --output_dir=result --train_batch_size=4 --data_dir=/home/ptruong/IDM-VTON/DATA/VITON-HD \
    --width 384 --height 512 --test_batch_size=1 --mixed_precision bf16 --adam_epsilon 1e-4 \
    --checkpointing_epoch 5000 