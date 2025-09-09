python -W ignore inference.py \
    --volume_path /data/users/wjy/C/dataset/CelebAMask-HQ/Images \
    --lora_ckpt /data/users/wjy/cpc_sam/output/brow4_auto_first_img256_20240513-072623/best.pth \
    --gpu_id 1 \
    --module sam_lora_mask_decoder \
    --dataset brow \
    --num_classes 1 \