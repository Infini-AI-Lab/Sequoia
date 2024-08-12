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
from data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset_eval, convert_wikimqa_dataset
import argparse
from Tree.SpecTree import SpecTree
import time
from utils import get_sampling_logits, _make_causal_mask, cuda_graph_for_residual, cuda_graph_for_sampling_without_replacement
from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG
from Engine.offload_engine import OffloadEngine
import random
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--target', type=str, help='target model')
parser.add_argument('--dataset', type=str, default="../dataset/c4_small.json", help='dataset path')
parser.add_argument('--growmap', type=str, default="../growmaps/68m_7b-64.pt", help='growmap path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=0.9, help='top_p')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--seed', type=int, default=17, help='random seed')
parser.add_argument('--Mode', type=str, default="greedy", help='tree mode')
parser.add_argument('--offloading', action='store_true')
args = parser.parse_args()
print(args)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(args.seed)



def simulation_fast(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, dataloader: DataLoader, T=0.6, top_p=0.9,
            max_length=512, residual_graph=None, grow_map=None, sampling_callables = None,
            sample_gather_indices = None):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_length).long().to('cuda:0')
    parents_buffer =  torch.zeros(max_length).long().to('cuda:0')
    position_ids = torch.zeros(max_length).long().to('cuda:0')
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            draft_kv_len = 0
            target_kv_len = 0
            attn_mask.fill_(torch.finfo(dtype).min)
            spectree = SpecTree(prefix=input_ids.squeeze(0), device='cuda:0', temperature=T,
                                    top_p=top_p,
                                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length, max_target_seq=max_length, grow_map=grow_map,
                                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer, 
                                    parents_buffer = parents_buffer, 
                                    position_ids = position_ids,
                                    residual_graph = residual_graph,
                                    sampling_callables=sampling_callables,
                                    sample_gather_indices = sample_gather_indices)
            torch.cuda.synchronize()
            t1 = time.time()
            while input_ids.shape[1] < 256 and terminate == False:
                spectree.construct_grow_map()
                valid_tokens, draft_kv_len, target_kv_len, terminate = spectree.verify()
                
                num_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
                num_large_model_steps += 1
                input_ids = valid_tokens.unsqueeze(0)
                if (input_ids[0][-1] == 2) or (input_ids[0][-1] == 0): terminate = True
            
            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            draft_model.clear_kv()
            target_model.clear_kv()
    print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}, {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps, num_decoding_steps / num_large_model_steps))
    return num_decoding_steps / num_large_model_steps



def simulation_baseline(target_model : GraphInferenceEngineTG, dataloader: DataLoader, T=0.6, top_p=0.9, max_length=256):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    total_time = 0.0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            position_ids = torch.arange(max_length).to('cuda:0').unsqueeze(0)
            storage_ids = torch.arange(max_length).to('cuda:0')
            attn_mask = _make_causal_mask((max_length, max_length), target_model.dtype, target_model.device)
            torch.cuda.synchronize()
            t1 = time.time()
            inner_decoding_step = 0
            start_length = 0
            while inner_decoding_step < 32 and terminate == False:
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
                if input_ids[0][-1] == 2: 
                    terminate = True
            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            target_model.clear_kv()
            
    print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps))
    return num_decoding_steps
def simulation_benchmark(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, dataloader: DataLoader, T=0.6, top_p=0.9, 
                max_length=512, residual_graph=None, grow_map=None, sampling_callables = None,
                sample_gather_indices = None):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    initialize_time = 0.0
    speculate_time = 0.0
    verify_time = 0.0
    large_model_run = 0.0
    accept_loop = 0.0
    kv_select = 0.0
    sample_time = 0.0
    small_model_compute = 0.0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_length).long().to('cuda:0')
    parents_buffer =  torch.zeros(max_length).long().to('cuda:0')
    position_ids = torch.zeros(max_length).long().to('cuda:0')
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
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
                                        sample_gather_indices = sample_gather_indices)
            while input_ids.shape[1] < 256 and terminate == False:
                torch.cuda.synchronize()
                t1 = time.time()
                torch.cuda.synchronize()
                t2 = time.time()
                a, b = spectree.construct_grow_map(benchmark=True)
                torch.cuda.synchronize()
                t3 = time.time()
                valid_tokens, draft_kv_len, target_kv_len, x, y, z, terminate = spectree.verify(benchmark=True)
                torch.cuda.synchronize()
                t4 = time.time()
                initial_size = input_ids.shape[1]
                input_ids = valid_tokens.unsqueeze(0)
                
                if (input_ids[0] == 2)._is_any_true() or (input_ids[0] == 0)._is_any_true() or input_ids.shape[1] >= 256: 
                    terminate = True
                if not terminate:
                    sample_time += a
                    small_model_compute += b
                    large_model_run += x
                    accept_loop += y
                    kv_select += z
                    initialize_time += (t2 - t1)
                    speculate_time += (t3 - t2)
                    verify_time += (t4 - t3)
                    num_decoding_steps += (valid_tokens.shape[0] - initial_size)
                    num_large_model_steps += 1
            draft_model.clear_kv()
            target_model.clear_kv()
            if num_large_model_steps > 0:
                print(num_decoding_steps / num_large_model_steps)
    print("total decoding steps: {}".format(num_decoding_steps), "large model steps: {}".format(num_large_model_steps), "avg decoding step: {}".format(num_decoding_steps / num_large_model_steps))
    print("initialization time:{}".format(initialize_time / num_large_model_steps), "speculate time: {}".format(speculate_time / num_large_model_steps),  "verify time: {}".format(verify_time / num_large_model_steps))
    print("large model run: {}".format(large_model_run / num_large_model_steps) , "accept loop: {}".format(accept_loop / num_large_model_steps), "kv select: {}".format(kv_select / num_large_model_steps))
    print("small model run: {}".format(small_model_compute / num_large_model_steps) , "sample time: {}".format(sample_time / num_large_model_steps))
    return num_decoding_steps / num_large_model_steps



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
eval_list = list(range(200, 2000))
import random
random.shuffle(eval_list)

if args.dataset == 'openwebtext':
    tokenized_dataset_eval = load_from_disk("../dataset/openwebtext_eval").select(eval_list[args.start :args.end])
elif args.dataset == 'wiki':
    tokenized_dataset_eval = convert_wiki_dataset(tokenizer=tokenizer).select(eval_list[args.start :args.end])
elif args.dataset == 'cnn':
    tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(eval_list[args.start :args.end])
elif args.dataset == 'wikimqa':
    tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(eval_list[args.start :args.end])
else:
    tokenized_dataset_eval = convert_c4_dataset_eval(tokenizer=tokenizer).select(eval_list[args.start :args.end])

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)


if args.Mode == 'baseline':
    if args.offloading:
        target_model = OffloadEngine(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, device="cuda:0")
    else:
        target_model =  GraphInferenceEngineTG(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, device="cuda:0")
else:
    draft_model = GraphInferenceEngine(max_length=args.M, model_name_or_path = args.model, dtype = torch.float16, device="cuda:0")
    if args.offloading:
        target_model = OffloadEngine(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, device="cuda:0")
    else:
        target_model =  GraphInferenceEngineTG(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, device="cuda:0")
    
    residual_graph = cuda_graph_for_residual()
    path = args.growmap
    grow_map = torch.load(path)

    tree_size = grow_map["size"]
    print(tree_size)
    idx_lists = grow_map["roots"]
    branch_lists = grow_map['branches']
    draft_step = len(grow_map["roots"])
    
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
            temperature=args.T, tree_size=tree_size) 
    for i in range(draft_step - 1):
        ith_gather_list = []
        max_num_samples = max(branch_lists[i])
        for j, branch in enumerate(branch_lists[i]):
            branch_index = torch.arange(branch, device="cuda:0", dtype=torch.long)
            branch_index = branch_index + j * max_num_samples
            ith_gather_list.append(branch_index)
        ith_gather_list = torch.cat(ith_gather_list)
        sample_gather_indices[i] = ith_gather_list
    

accelerator = Accelerator()
dataloader = accelerator.prepare(dataloader)

if args.Mode == 'benchmark':
    simulation_benchmark(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P, 
                                               max_length=args.M, residual_graph = residual_graph, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices)
elif args.Mode == 'baseline':
    simulation_baseline(target_model=target_model, dataloader=dataloader, T=args.T, top_p=args.P, max_length=args.M)
elif args.Mode == 'greedy':
    simulation_fast(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P,
                                     max_length=args.M, residual_graph = residual_graph, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices)