import sys
sys.path.append("..")
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
import torch
import numpy as np 
from datasets import load_from_disk
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn.functional import softmax
from accelerate import Accelerator
import argparse
from data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset_eval
import argparse
from Tree.SpecTree import SpecTree
import time
from utils import get_sampling_logits, _make_causal_mask, cuda_graph_for_residual, cuda_graph_for_sampling_without_replacement
from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG
from Engine.offload_engine import OffloadEngine
import random
import json
import os.path as osp
import ssl
import urllib.request
import os
from time import sleep
def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict

def simulation_fast(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, prompts :list[str], tokenizer: AutoTokenizer,T=0.6, top_p=0.9,
            max_length=512, residual_graph=None, grow_map=None, sampling_callables = None,
            sample_gather_indices = None, vocab_size=32000):

    num_eval_steps = len(prompts)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer =  None
    parents_buffer =  None
    position_ids = torch.zeros(max_length).long().to('cuda:0')
    data_id = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(prompts), total=num_eval_steps):
            batch = "[INST]" + batch + "[/INST]" + "\n\nASSISTANT:"
            input_ids = tokenizer(batch, return_tensors="pt").input_ids.to('cuda:0')
            if input_ids.shape[1] > 200:
                continue
            data_id  = data_id + 1
            input_text = (
                    tokenizer.decode(
                    input_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                )
                .strip()
                .split(" ")
                )
            print(" ".join(input_text), end=" ", flush=True)
            terminate = False
            
            draft_kv_len = 0
            target_kv_len = 0
            attn_mask.fill_(torch.finfo(dtype).min)
            spectree = SpecTree(prefix=input_ids.squeeze(0), device='cuda:0', temperature=T,
                                    top_p=top_p,
                                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length, grow_map=grow_map,
                                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer, 
                                    parents_buffer = parents_buffer, 
                                    position_ids = position_ids,
                                    residual_graph = residual_graph,
                                    sampling_callables=sampling_callables,
                                    sample_gather_indices = sample_gather_indices, vocab_size=vocab_size)
            torch.cuda.synchronize()
            t1 = time.time()
            pos = 0
            generated_ids = []
            while input_ids.shape[1] < 256 and terminate == False:
                spectree.construct_grow_map()
                valid_tokens, draft_kv_len, target_kv_len, terminate = spectree.verify()
                
                num_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
                input_begin_pos = input_ids.shape[1]
                num_large_model_steps += 1
                input_ids = valid_tokens.unsqueeze(0)
                
                if (input_ids[0][-1] == 2) or (input_ids[0][-1] == 0): terminate = True
                
                generated_ids.extend(input_ids[0][input_begin_pos:].tolist())

                generated_text = (
                    tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                )
                .strip()
                .split(" ")
                )
                now = len(generated_text) - 1
                if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now

            print(" ".join(generated_text[pos:]), flush=True)
            
            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            draft_model.clear_kv()
            target_model.clear_kv()
            if num_large_model_steps > 0:
                print("Data ID: {} total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}, {}".format(data_id, total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps, num_decoding_steps / num_large_model_steps), flush=True)
    print("Data ID: {} total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}, {}".format(data_id, total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps, num_decoding_steps / num_large_model_steps), flush=True)
    return num_decoding_steps / num_large_model_steps

def simulation_baseline(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, prompts :list[str], tokenizer: AutoTokenizer,T=0.6, top_p=0.9,
            max_length=512, residual_graph=None, grow_map=None, sampling_callables = None,
            sample_gather_indices = None, vocab_size=32000):

    num_eval_steps = len(prompts)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    
    position_ids = torch.zeros(max_length).long().to('cuda:0')
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(prompts), total=num_eval_steps):
            batch = "[INST]" + batch + "[/INST]" + "\n\nASSISTANT:"
            input_ids = tokenizer(batch, return_tensors="pt").input_ids.to('cuda:0')
            initial_len = input_ids.shape[1]
            input_text = (
                    tokenizer.decode(
                    input_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                )
                .strip()
                .split(" ")
                )
            print(" ".join(input_text), end=" ", flush=True)
            terminate = False
            
            position_ids = torch.arange(max_length).to('cuda:0').unsqueeze(0)
            storage_ids = torch.arange(max_length).to('cuda:0')
            attn_mask = _make_causal_mask((max_length, max_length), target_model.dtype, target_model.device)
            torch.cuda.synchronize()
            t1 = time.time()
            inner_decoding_step = 0
            start_length = 0
            pos = 0
            generated_ids = []
            while inner_decoding_step < initial_len + 256 and terminate == False:
                if inner_decoding_step == 0:
                    start_length = input_ids.shape[1]
                    logits = target_model.inference(input_ids = input_ids, storage_ids=storage_ids[:start_length],
                                                    position_ids = position_ids[..., :start_length], 
                                                    attn_mask=attn_mask[:start_length, :start_length][None, None, :, :])[0][-1]
                    
                else:
                    logits = target_model.inference(input_ids = input_ids, storage_ids=storage_ids[start_length + inner_decoding_step-1 : start_length + inner_decoding_step],
                                                    position_ids = position_ids[..., start_length + inner_decoding_step-1 : start_length + inner_decoding_step], 
                                                    attn_mask=attn_mask[start_length + inner_decoding_step-1 : start_length + inner_decoding_step, :start_length + inner_decoding_step][None, None, :, :])[0][-1]
                
                logits = get_sampling_logits(logits=logits, top_p=top_p, T=T)
                
                p = softmax(logits / T, dim=-1)
                new_token = p.multinomial(num_samples=1).unsqueeze(0)
                input_ids = new_token
                num_decoding_steps += 1
                inner_decoding_step += 1
                
                generated_ids.extend(input_ids[0][-1:].tolist())

                generated_text = (
                    tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                )
                .strip()
                .split(" ")
                )
                now = len(generated_text) - 1
                if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now

            print(" ".join(generated_text[pos:]), flush=True)
            
            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            target_model.clear_kv()
            if num_decoding_steps > 0:
                print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps), flush=True)
    print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps))
    return num_decoding_steps / num_large_model_steps
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    target_model = OffloadEngine(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, device="cuda:0", stay_layers=args.staylayer)
    if args.Mode == 'spec':
        draft_model = GraphInferenceEngine(max_length=args.M, model_name_or_path = args.model, dtype = torch.float16, device="cuda:0")
        residual_graph = cuda_graph_for_residual(dim=args.vocab)
        path = args.growmap
        grow_map = torch.load(path)
        tree_size = grow_map["size"]
        print(tree_size)
        idx_lists = grow_map["roots"]

        branch_lists = grow_map['branches']
        draft_step = len(grow_map["roots"])
        
        if args.cudagraph:
            graph_capture_list = [sum(x) for x in branch_lists]
            
            graph_capture_list.append(1)
            draft_model.initialize_cuda_graph(graph_capture_list)

        sampling_callables = {}
        sample_gather_indices = {}
        for i in range(draft_step - 1):
            idx_len = len(idx_lists[i])
            num_samples = max(branch_lists[i])
            sampling_callables[i] = cuda_graph_for_sampling_without_replacement(
                max_length=args.M, idx_len=idx_len, num_samples=num_samples,
                temperature=args.T, tree_size=tree_size, dim=args.vocab) 
        for i in range(draft_step - 1):
            ith_gather_list = []
            max_num_samples = max(branch_lists[i])
            for j, branch in enumerate(branch_lists[i]):
                branch_index = torch.arange(branch, device="cuda:0", dtype=torch.long)
                branch_index = branch_index + j * max_num_samples
                ith_gather_list.append(branch_index)
            ith_gather_list = torch.cat(ith_gather_list)
            sample_gather_indices[i] = ith_gather_list
    
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts.append(sample["turns"][0])
    if args.Mode == 'spec':
        simulation_fast(target_model=target_model, draft_model=draft_model, prompts=prompts,tokenizer=tokenizer, T=args.T, top_p=args.P,
                                        max_length=args.M, residual_graph = residual_graph, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices, vocab_size=args.vocab)
    else:
        simulation_baseline(target_model=target_model, draft_model=None, prompts=prompts[:4],tokenizer=tokenizer, T=args.T, top_p=args.P,
                                        max_length=args.M, residual_graph = None, grow_map = None, sampling_callables=None, sample_gather_indices = None, vocab_size=args.vocab)
    
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model', default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--target', type=str, help='target model', default="meta-llama/Llama-2-70b-hf")
    parser.add_argument("--data_root", type=str, default="dataset/")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument('--growmap', type=str, default="../L40_growmaps/L40-CNN-7b-70b-stochastic.pt", help='growmap path')
    parser.add_argument('--T', type=float, default=0.6, help='temperature')
    parser.add_argument('--P', type=float, default=0.9, help='top_p')
    parser.add_argument('--cudagraph', action='store_true')
    parser.add_argument('--seed', type=int, default=17, help='random seed')
    parser.add_argument('--Mode', type=str, default="spec", help='tree mode')
    parser.add_argument('--staylayer', type=int, default=0, help='layers on chip')
    parser.add_argument('--vocab', type=int, default=32000, help='vocab size')
    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)
