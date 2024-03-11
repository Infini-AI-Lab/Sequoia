CUDA_VISIBLE_DEVICES=0 python run_sequoia.py --model  meta-llama/Llama-2-7b-chat-hf \
    --target meta-llama/Llama-2-70b-chat-hf  --T 0.6 --P 1.0 --staylayer 11 \
    --M 1024 \
    --growmap  ../L40_growmaps/L40-CNN-7b-70b-stochastic.pt --Mode baseline --seed 17 --vocab 32000
