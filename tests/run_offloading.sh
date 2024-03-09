CUDA_VISIBLE_DEVICES=0 python run_sequoia.py --model  TinyLlama/TinyLlama-1.1B-Chat-v1.0\
    --target meta-llama/Llama-2-13b-chat-hf  --T 0.6 --P 1.0 --staylayer 28\
    --M 1024 \
    --growmap  ../L40_growmaps/L40-CNN-7b-70b-stochastic.pt --Mode spec --seed 17
