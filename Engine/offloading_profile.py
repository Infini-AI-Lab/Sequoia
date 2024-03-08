from offload_engine import OffloadEngine, _make_causal_mask
import argparse
import time
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-2-70b-hf",help='model')
parser.add_argument('--T', type=int, default=100, help='repeat times')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--P', type=int, default=128, help='prefix length')
parser.add_argument('--M', type=int, default=1536, help='max length')
parser.add_argument('--D', type=int, default=1, help='dec length')
args = parser.parse_args()
print(args)
PREFIX_LEN = args.P
MAX_LEN = args.M
DEC_LEN = args.D
MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = "cuda:0"
T = args.T
WARM_UP = 10

llm = OffloadEngine(max_length=MAX_LEN, model_name_or_path=args.model)
input_ids = torch.randint(low=3, high=30000, size=(1, PREFIX_LEN), device=DEVICE)
attention_mask = _make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
attention_mask = attention_mask[None, None, :, :]
position_ids = torch.arange(PREFIX_LEN, device=DEVICE).unsqueeze(0)
llm.inference(input_ids=input_ids, position_ids=position_ids, attn_mask=attention_mask[..., :PREFIX_LEN,:PREFIX_LEN], storage_ids=None)

input_ids = torch.randint(low=3, high=30000, size=(1, DEC_LEN), device=DEVICE)
storage_ids = torch.arange(DEC_LEN, device=DEVICE) + PREFIX_LEN
position_ids = storage_ids.clone().unsqueeze(0)
attention_mask = attention_mask[..., PREFIX_LEN: PREFIX_LEN + DEC_LEN,:PREFIX_LEN + DEC_LEN].clone()
for _ in range(WARM_UP):
    llm.inference(input_ids=input_ids, position_ids=position_ids, attn_mask=attention_mask, storage_ids=None)
    llm.set_kv_len(PREFIX_LEN)

torch.cuda.synchronize()
t1 = time.time()
for _ in range(T):
    llm.inference(input_ids=input_ids, position_ids=position_ids, attn_mask=attention_mask, storage_ids=None)
    llm.set_kv_len(PREFIX_LEN)
torch.cuda.synchronize()
t2 = time.time()

print("Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s".format(MAX_LEN, DEC_LEN, PREFIX_LEN, (t2 - t1)/ T))


