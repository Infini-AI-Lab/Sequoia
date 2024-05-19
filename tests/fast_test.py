import os
import sys
sys.path.append("..")
from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollatorForLanguageModeling, OPTForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np 
from datasets import load_from_disk, Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn.functional import softmax
import accelerate
from accelerate import Accelerator
import argparse
from data_converter import convert_dataset, convert_cnn_dataset, convert_wiki_dataset
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--target', type=str, help='target model')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=1.0, help='top_p')
parser.add_argument('--DP', type=float, default=1.1, help='draft_top_p')
parser.add_argument('--W', type=int, default=16, help='max width')
parser.add_argument('--dataset', type=str, default="../dataset/c4_small.json", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--dst', type=str, default="../acceptance-rate-vector.pt", help='destination for accepetance rate vector')
args = parser.parse_args()
print(args)
def get_residual(p: torch.Tensor, q:torch.Tensor):
    residual = p - q
    residual[residual < 0] = 0.0
    residual = residual / (residual.sum(dim=-1).unsqueeze(-1) + 1e-9)
    
    return residual

def evaluate(target_model : LlamaForCausalLM, draft_model: LlamaForCausalLM, dataloader: DataLoader, k:int, T=0.6, top_p=0.9, draft_top_p=0.99):
    num_eval_steps = len(dataloader)
    acceptance_rate = torch.zeros(k)
    num_samples = 0
    draft_model_prob = []
    token_accept_rate = []
    sampled_token_sets = []
    real_budget = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            target_logits : torch.Tensor = target_model(**batch).logits
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(target_logits, descending=True)
                cumulative_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                filter = cumulative_probs > top_p
                filter[..., 1:] = filter[..., :-1].clone()
                filter[..., 0] = 0
                indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                target_logits[indices_to_remove] = float('-inf')

            
            draft_logits : torch.Tensor = draft_model(**batch).logits
            target_prob = softmax(target_logits / T, dim=-1).squeeze(0)
            q = softmax(draft_logits / T, dim=-1).squeeze(0)
            
            for i in range(128, target_prob.shape[0]):
                token_acceptance_rate = torch.zeros(k)
                draft_tokens = []
                if batch['labels'][0][i] == -100 or batch['labels'][0][i] == 0: continue
                num_samples = num_samples + 1
                token_target_prob = target_prob[i]
                # token_draft_prob = q[i]
                #draft_model_prob.append(q[i].cpu())
                token_draft_logits = draft_logits[0][i]

                if draft_top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(token_draft_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                    filter = cumulative_probs > draft_top_p
                    filter[..., 1:] = filter[..., :-1].clone()
                    filter[..., 0] = 0
                    indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                    token_draft_logits[indices_to_remove] = float('-inf')

                token_draft_prob = softmax(token_draft_logits / T, dim=-1).squeeze(0)
                sampled_token = token_draft_prob.multinomial(num_samples=1, replacement=True)
                draft_tokens.append(sampled_token.item())
                real_budget = real_budget + 1
                token_acceptance_rate[0] = min(1.0, (token_target_prob[sampled_token]/ token_draft_prob[sampled_token]))

                token_target_prob = get_residual(token_target_prob, token_draft_prob)
                
                
                for j in range(k-1):
                    token_draft_logits[sampled_token] = - torch.inf
                    token_draft_prob = softmax(token_draft_logits / (T), dim=-1).squeeze(0)
                    if torch.isnan(token_draft_prob).long().sum() >= 1:
                        break
                    token_draft_prob = token_draft_prob / token_draft_prob.sum(-1)
                    sampled_token = token_draft_prob.multinomial(num_samples=1, replacement=True)
                    draft_tokens.append(sampled_token.item())
                    real_budget = real_budget + 1
                    branch_token_acceptance_rate = min(1, token_target_prob[sampled_token]/ token_draft_prob[sampled_token])
                    token_acceptance_rate[j+1] = (1 - token_acceptance_rate.sum()) * branch_token_acceptance_rate
                    
                    token_target_prob = get_residual(token_target_prob, token_draft_prob)
                acceptance_rate = acceptance_rate + token_acceptance_rate
                token_accept_rate.append(token_acceptance_rate.cpu())
                sampled_token_sets.append(draft_tokens)
                draft_model_prob.append(q[i][draft_tokens].cpu()) 
    return acceptance_rate / num_samples


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
dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator)
target_model = AutoModelForCausalLM.from_pretrained(args.target, 
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="auto"
                                                    )
draft_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda:0", torch_dtype=torch.bfloat16)
accelerator = Accelerator()
dataloader = accelerator.prepare(dataloader)

acceptance_rate_list = [0]
branch_acceptance_rate_list = [0]

acceptance_rate = evaluate(target_model, draft_model, dataloader, k=args.W, T=args.T, top_p=args.P, draft_top_p=args.DP)
x = torch.zeros(len(acceptance_rate) + 1)
x[1:] = acceptance_rate
torch.save(x, args.dst)
print(x)
