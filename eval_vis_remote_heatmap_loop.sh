export CUDA_VISIBLE_DEVICES=3
export NO_ALBUMENTATIONS_UPDATE=1
export NCCL_P2P_DISABLE=1
export HF_ENDPOINT="https://hf-mirror.com"

setsid nohup python eval_vis_remote_heatmap_loop.py > log_1.txt 2>&1 &
