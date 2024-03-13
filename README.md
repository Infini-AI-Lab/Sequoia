# Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding
[[paper](https://arxiv.org/abs/2402.12374)]
## Environment Set Up
We recommend the following commands to set up the environment

    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.36.2
    pip install accelerate==0.26.1
    pip install datasets==2.16.1
    pip install einops
    pip install protobuf
    pip install sentencepiece
    pip install typing-extensions

## Evaluations
To reproduce the main results

    cd tests
    bash run_L40.sh

or `bash run_A100.sh`
    
A command should be in the format like

    python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  \
    --T 0.6 --P 1.0  --start 0 --end 200 --M 384 \
    --growmap ../A100_growmaps/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt \
    --Mode greedy --dataset cnn

`testbed.py` is for stochastic decoding. `testbed_greedy.py` is for greedy decoding. `test_specinfer.py` is for specinfer sampling. `test_greedyS.py` is for Top-k/greedy sampling. `test_accept.py` is for preparing the accepting rate vector.

`--model` specifies the draft and `--target` spefifies the target. Currently, only Llama models are supported (including Llama2, Sheared-LLaMA, Vicuna and TinyLlama).

`--T` specifies the temperature and `--P` spefifies the top-p for generation. 

`--dataset` should be in `cnn, openwebtext, c4`.  `--start` and `--end` decides how many examples will be evaluated. `--seed` is for adjusting random seeds. To precisely reproduce the results, seed is set to be 17 by default.

`--growmap` specifies the tree structure. We have prepared some growmaps in `A100_growmaps` and `L40_growmaps`. 

`--M` should be set at least `#tree + 256`. 384 is enough for all the experiments except offloading. To run offloading, we need the command like the following

    CUDA_VISIBLE_DEVICES=0 python testbed.py --model meta-llama/Llama-2-7b-hf \
    --target meta-llama/Llama-2-70b-hf  --T 0.6 --P 1.0 \
    --start 0 --end 100 --Mode greedy  --M 1024 \
    --growmap  ../L40_growmaps/L40-CNN-7b-70b-stochastic.pt  --offloading --dataset cnn

## How to obtain acceptance rate vector
To obtain the acceptance rate vector, which is used in `tree_search.py`, we need the following command

    python test_accept.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  \
    --T 0.6 --P 1.0  --start 0 --end 200 --M 288 --W 32\
    --ALG stochastic --dataset cnn \

`--ALG` is stochastic or greedy. `--W` is the maximum width. `--M` should be set at least `--W + 256`.

To statically obtain the acceptance rate vector (which is much faster if the target model needs offloading)

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python fast_test.py --model meta-llama/Llama-2-7b-hf  \
    --target meta-llama/Llama-2-70b-hf --T 1.1 --P 1.0 --DP 1.1 --W 32 --start 0 --end 200

The acceptance rate vector will be printed and will be saved to `--dst` (`../acceptance-rate-vector.pt` by default).

## How to generate growmaps

We use the following command

    python tree_search.py --config demo-config.json

We can modify the content of demo-config.json to generate different growmaps. The growmaps for experiments in the paper in prepared in `L40_growmaps` and `A100_growmaps`. 

## TODOs
- [ ] Support other open source models.
- [ ] Support multi-round dialogue.
- [ ] Support INT4/8 quantization.
- [ ] Support multi-GPU. 
## Citation

If you find Sequoia useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{chen2024sequoia,
  title={Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding},
  author={Chen, Zhuoming and May, Avner and Svirschevski, Ruslan and Huang, Yuhsun and Ryabinin, Max and Jia, Zhihao and Chen, Beidi},
  journal={arXiv preprint arXiv:2402.12374},
  year={2024}
}
```












