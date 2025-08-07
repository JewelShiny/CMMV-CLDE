export CUDA_VISIBLE_DEVICES=3,4,5
export NO_ALBUMENTATIONS_UPDATE=1
export NCCL_P2P_DISABLE=1
export HF_ENDPOINT="https://hf-mirror.com"

setsid nohup python train_mvcv.py > log_2.txt 2>&1 &