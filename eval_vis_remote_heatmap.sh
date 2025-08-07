export CUDA_VISIBLE_DEVICES=0
export NO_ALBUMENTATIONS_UPDATE=1
export NCCL_P2P_DISABLE=1
export HF_ENDPOINT="https://hf-mirror.com"

python eval_vis_remote_heatmap.py
