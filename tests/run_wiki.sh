CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode greedy --dataset wikimqa --S 128 
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 512 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode greedy --dataset wikimqa --S 256
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 640 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode greedy --dataset wikimqa --S 384
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 768 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode greedy --dataset wikimqa --S 512
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 896 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode greedy --dataset wikimqa --S 640
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 1024 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode greedy --dataset wikimqa --S 768
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 1152 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode greedy --dataset wikimqa --S 896
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 1280 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode greedy --dataset wikimqa --S 1024

CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode greedy --dataset wikimqa --S 128 
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 512 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode greedy --dataset wikimqa --S 256
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 640 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode greedy --dataset wikimqa --S 384
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 768 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode greedy --dataset wikimqa --S 512
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 896 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode greedy --dataset wikimqa --S 640
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 1024 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode greedy --dataset wikimqa --S 768
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 1152 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode greedy --dataset wikimqa --S 896
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 1280 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode greedy --dataset wikimqa --S 1024




CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 10 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode baseline --dataset wikimqa --S 128 
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 10 --M 512 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode baseline --dataset wikimqa --S 256
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 10 --M 640 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode baseline --dataset wikimqa --S 384
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 10 --M 768 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode baseline --dataset wikimqa --S 512
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 10 --M 896 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode baseline --dataset wikimqa --S 640
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 10 --M 1024 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode baseline --dataset wikimqa --S 768
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 10 --M 1152 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode baseline --dataset wikimqa --S 896
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 10 --M 1280 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode baseline --dataset wikimqa --S 1024

CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 10 --M 384 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode baseline --dataset wikimqa --S 128 
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 10 --M 512 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode baseline --dataset wikimqa --S 256
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 10 --M 640 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode baseline --dataset wikimqa --S 384
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 10 --M 768 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode baseline --dataset wikimqa --S 512
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 10 --M 896 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode baseline --dataset wikimqa --S 640
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 10 --M 1024 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode baseline --dataset wikimqa --S 768
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 10 --M 1152 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode baseline --dataset wikimqa --S 896
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 10 --M 1280 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode baseline --dataset wikimqa --S 1024













