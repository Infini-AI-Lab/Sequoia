CUDA_VISIBLE_DEVICES=8 python run_sequoia.py --model  meta-llama/Llama-2-7b-chat-hf\
    --target meta-llama/Llama-2-70b-chat-hf  --T 0.6 --P 1.0 \
    --M 1536 \
    --growmap  ../L40_growmaps/L40-CNN-7b-70b-stochastic.pt
