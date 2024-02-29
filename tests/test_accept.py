import sys
sys.path.append("..")
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
import torch
from datasets import load_from_disk
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
import argparse
from data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset_eval, convert_dataset
import argparse
from Tree.SpecTree import SpecTreeTest
from Tree.GreedyTree import GreedyTreeTest
from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--target', type=str, help='target model')
parser.add_argument('--dataset', type=str, default="../dataset/c4_small.json", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=0.9, help='top_p')
parser.add_argument('--ALG', type=str, default="stochastic", help='algorithm')
parser.add_argument('--D', type=int, default=1, help='depth')
parser.add_argument('--W', type=int, default=16, help='max width')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--Mode', type=str, default="greedy", help='tree mode')
parser.add_argument('--offloading', action='store_true')
parser.add_argument('--dst', type=str, default="../acceptance-rate-vector.pt", help='destination for accepetance rate vector')
args = parser.parse_args()
print(args)




def simulation_stochastic(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, dataloader: DataLoader, T=0.6, top_p=0.9, w=4, max_length=512):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_length).long().to('cuda:0')
    parents_buffer =  torch.zeros(max_length).long().to('cuda:0')
    position_ids = torch.zeros(max_length).long().to('cuda:0')
    branch_prob = torch.zeros(w + 1).to('cuda:0')
    output_branch_prob = torch.zeros(w + 2).to('cuda:0')
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            draft_kv_len = 0
            target_kv_len = 0
            while input_ids.shape[1] < 256 and terminate == False:
                attn_mask.fill_(torch.finfo(dtype).min)
                spectree = SpecTreeTest(prefix=input_ids.squeeze(0), device='cuda:0', temperature=T,
                                    top_p=top_p, 
                                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length,
                                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer, 
                                    parents_buffer = parents_buffer, 
                                    position_ids = position_ids, max_width=w)
                
                
                valid_tokens, draft_kv_len, target_kv_len,  b, terminate = spectree.verify(benchmark=True)
                branch_prob[b] += 1
                
                
                num_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
                num_large_model_steps += 1
                input_ids = valid_tokens.unsqueeze(0)
                if (input_ids[0] == 2)._is_any_true() or (input_ids[0] == 0)._is_any_true(): terminate = True
                
            draft_model.clear_kv()
            target_model.clear_kv()
            if num_large_model_steps > 0:
                print(num_decoding_steps / num_large_model_steps)
    print("total decoding steps: {}".format(num_decoding_steps), "large model steps: {}".format(num_large_model_steps), "avg decoding step: {}".format(num_decoding_steps / num_large_model_steps))
    branch_prob = branch_prob / branch_prob.sum(dim=-1) 
    accumated_prob = branch_prob.cumsum(dim=-1)
    output_branch_prob[1:] = branch_prob
    print(output_branch_prob)
    torch.save(output_branch_prob, args.dst)
    return num_decoding_steps / num_large_model_steps

def simulation_greedy(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, dataloader: DataLoader, T=0.6, top_p=0.9, w=4, max_length=512):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_length).long().to('cuda:0')
    parents_buffer =  torch.zeros(max_length).long().to('cuda:0')
    position_ids = torch.zeros(max_length).long().to('cuda:0')
    branch_prob = torch.zeros(w + 1).to('cuda:0')
    output_branch_prob = torch.zeros(w + 2).to('cuda:0')
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            draft_kv_len = 0
            target_kv_len = 0
            while input_ids.shape[1] < 256 and terminate == False:
                attn_mask.fill_(torch.finfo(dtype).min)
                spectree = GreedyTreeTest(prefix=input_ids.squeeze(0), device='cuda:0', temperature=T,
                                    top_p=top_p, 
                                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length,
                                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer, 
                                    parents_buffer = parents_buffer, 
                                    position_ids = position_ids, max_width=w)
                
                valid_tokens, draft_kv_len, target_kv_len,  b, terminate = spectree.verify(benchmark=True)
                initial_size = input_ids.shape[1]
                input_ids = valid_tokens.unsqueeze(0)
                
                
                if (input_ids[0] == 2)._is_any_true() or (input_ids[0] == 0)._is_any_true(): terminate = True
                if not terminate:
                    branch_prob[b] += 1
                    num_decoding_steps += (valid_tokens.shape[0] - initial_size)
                    num_large_model_steps += 1

                
            draft_model.clear_kv()
            target_model.clear_kv()
            if num_large_model_steps > 0:
                print(num_decoding_steps / num_large_model_steps)
    print("total decoding steps: {}".format(num_decoding_steps), "large model steps: {}".format(num_large_model_steps), "avg decoding step: {}".format(num_decoding_steps / num_large_model_steps))
    branch_prob = branch_prob / branch_prob.sum(dim=-1) 
    accumated_prob = branch_prob.cumsum(dim=-1)
    output_branch_prob[1:] = branch_prob
    print(output_branch_prob)
    torch.save(output_branch_prob, args.dst)
    return num_decoding_steps / num_large_model_steps


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
if args.dataset == 'openwebtext':
    tokenized_dataset_eval = load_from_disk("../dataset/openwebtext_eval").select(list(range(args.start, args.end)))
elif args.dataset == 'wiki':
    tokenized_dataset_eval = convert_wiki_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
elif args.dataset == 'cnn':
    tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
else:
    tokenized_dataset_eval = convert_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(args.start, args.end)))

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)


draft_model = GraphInferenceEngine(max_length=args.M, model_name_or_path = args.model, dtype = torch.float16, device="cuda:0")
target_model = GraphInferenceEngineTG(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, device="cuda:0", offloading=args.offloading)
graph_capture_list = list(range(1, 129))
draft_model.initialize_cuda_graph(graph_capture_list)

accelerator = Accelerator()
dataloader = accelerator.prepare(dataloader)
if args.ALG == "stochastic":
    simulation_stochastic(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P, w=args.W, max_length=args.M)

elif args.ALG == "greedy":
    simulation_greedy(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P, w=args.W, max_length=args.M)
