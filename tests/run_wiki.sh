CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/1.3b_33b/growmaps/A100-CNN-1.3b-33b-greedy.pt  --Mode greedy --dataset wikimqa --S 128 
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 1152 --growmap ../A100_growmaps/1.3b_33b/growmaps/A100-CNN-1.3b-33b-greedy.pt  --Mode greedy --dataset wikimqa --S 512
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 1280 --growmap ../A100_growmaps/1.3b_33b/growmaps/A100-CNN-1.3b-33b-greedy.pt  --Mode greedy --dataset wikimqa --S 1024
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 2048 --growmap ../A100_growmaps/1.3b_33b/growmaps/A100-CNN-1.3b-33b-greedy.pt  --Mode greedy --dataset wikimqa --S 1664


CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 10 --M 384 --growmap ../A100_growmaps/1.3b_33b/growmaps/A100-CNN-1.3b-33b-greedy.pt  --Mode greedy --dataset wikimqa --S 128 
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 10 --M 1152 --growmap ../A100_growmaps/1.3b_33b/growmaps/A100-CNN-1.3b-33b-greedy.pt  --Mode greedy --dataset wikimqa --S 512
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 10 --M 1280 --growmap ../A100_growmaps/1.3b_33b/growmaps/A100-CNN-1.3b-33b-greedy.pt  --Mode greedy --dataset wikimqa --S 1024
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 10 --M 2048 --growmap ../A100_growmaps/1.3b_33b/growmaps/A100-CNN-1.3b-33b-greedy.pt  --Mode greedy --dataset wikimqa --S 1664













