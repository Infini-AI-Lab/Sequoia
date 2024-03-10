CUDA_VISIBLE_DEVICES=2 python run_sequoia.py --model  chargoddard/internlm2-7b-llama \
    --target chargoddard/internlm2-20b-llama  --T 0.6 --P 1.0 --staylayer 20 \
    --M 1024 \
    --growmap  ../L40_growmaps/L40-CNN-7b-70b-stochastic.pt --Mode baseline --seed 17 --vocab 92544
