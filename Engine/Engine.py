import torch
from .Llama_KV import KV_Cache
from .Llama_model import LlamaForCausalLM_FI, LlamaForCausalLM_TG
from typing import List, Optional, Tuple, Union
import gc
import accelerate
class InferenceEngine:
    def __init__(self, 
        max_length:int,
        model_name_or_path :str,
        dtype = torch.float16,
        device = "cuda:0") -> None:
        
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

        self.model = LlamaForCausalLM_FI.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map=device)
        self.model.eval()
        self.model_config = self.model.config

        self.kv_cache = KV_Cache(config=self.model_config, max_length=max_length, device=device, dtype=dtype)
    
    @torch.inference_mode()
    def model_run(self, 
            input_ids: torch.LongTensor, 
            storage_ids :torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            debug :bool=False):
        if debug:
            _, input_length = input_ids.shape
            assert storage_ids.shape[0] == input_length
            assert attention_mask.shape[0] == input_length
            assert attention_mask.shape[1] == self.max_length
            assert position_ids.shape[1] == input_length
        
        logits = self.model(input_ids=input_ids, 
                    max_length=self.max_length, storage_ids=storage_ids,
                    attention_mask=attention_mask, position_ids=position_ids,
                    kv_cache=self.kv_cache, debug=debug)

        return logits
    
    def clear_kv(self):
        self.kv_cache.clear()
    
    def initialize_kv(self, k_cache :torch.Tensor, v_cache :torch.Tensor, kv_len :int):
        self.kv_cache.initialize_kv(k_cache, v_cache, kv_len)
    
    def gather_kv(self, indices: list[int]):
        self.kv_cache.gather_kv(indices)

    def get_kv_cache(self, in_place=False):
        if not in_place:
            return self.kv_cache.k_cache.clone(), self.kv_cache.v_cache.clone()
        else:
            return self.kv_cache.k_cache, self.kv_cache.v_cache

class InferenceEngineTG:
    def __init__(self, 
        max_length:int,
        model_name_or_path :str,
        dtype = torch.float16,
        device = "cuda:0",
        offloading = False) -> None:
        
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

        
        if offloading:
            self.model = LlamaForCausalLM_TG.from_pretrained(model_name_or_path, torch_dtype=dtype)
            self.model.eval()
            # device_map = accelerate.infer_auto_device_map(self.model, max_memory={0: 35 * (1 << 30), "cpu": 120 * (1 << 30)}, dtype=torch.float16)
            # self.model = accelerate.dispatch_model(self.model, main_device="cuda:0", device_map=device_map)

            self.model = accelerate.cpu_offload(self.model, execution_device=self.device)
        else:
            self.model = LlamaForCausalLM_TG.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map=device)
            self.model.eval()
        self.model_config = self.model.config

        self.kv_cache = KV_Cache(config=self.model_config, max_length=max_length, device=device, dtype=dtype)
    
    @torch.no_grad()
    def model_run(self, 
            input_ids: torch.LongTensor, 
            storage_ids :torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            debug :bool=False):
        if debug:
            _, input_length = input_ids.shape
            assert storage_ids.shape[0] == input_length
            assert attention_mask.shape[0] == input_length
            assert attention_mask.shape[1] == self.max_length
            assert position_ids.shape[1] == input_length
        
        logits = self.model(input_ids=input_ids, 
                    max_length=self.max_length, storage_ids=storage_ids,
                    attention_mask=attention_mask, position_ids=position_ids,
                    kv_cache=self.kv_cache, debug=debug)

        return logits
    
    def clear_kv(self):
        self.kv_cache.clear()
    
    def set_kv_len(self, kv_len :int):
        self.kv_cache.set_kv_len(kv_len)
    
    def initialize_kv(self, k_cache :torch.Tensor, v_cache :torch.Tensor, kv_len :int):
        self.kv_cache.initialize_kv(k_cache, v_cache, kv_len)
    
    def gather_kv(self, indices: list[int]):
        self.kv_cache.gather_kv(indices)

    def get_kv_cache(self, in_place=False):
        if not in_place:
            return self.kv_cache.k_cache.clone(), self.kv_cache.v_cache.clone()
        else:
            return self.kv_cache.k_cache, self.kv_cache.v_cache


def capture_graph(
    engine :InferenceEngine, decoding_seqlen :int =1, mempool=None, n_warmups :int=3
):
    device = engine.device
    dtype = engine.dtype
    static_input_ids = torch.full((1, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_position_ids = torch.full((1, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_storage_ids = torch.arange(decoding_seqlen, dtype=torch.long, device=device)
    static_attn_mask = torch.full((decoding_seqlen, engine.max_length), 0, dtype=dtype, device=device)
    static_attn_mask = static_attn_mask[None, None, :, :]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_logits = engine.model_run(
                    input_ids=static_input_ids, 
                    storage_ids=static_storage_ids, 
                    position_ids=static_position_ids, 
                    attention_mask=static_attn_mask
                    )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = engine.model_run(
                input_ids=static_input_ids, 
                storage_ids=static_storage_ids, 
                position_ids=static_position_ids, 
                attention_mask=static_attn_mask
                )
    def run(input_ids, storage_ids, position_ids, attn_mask):
        static_input_ids.copy_(input_ids)
        static_storage_ids.copy_(storage_ids)
        static_position_ids.copy_(position_ids)
        static_attn_mask.copy_(attn_mask)
        graph.replay()
        return static_logits.clone()
    
    return run

class GraphInferenceEngine:
    def __init__(self, 
        max_length:int,
        model_name_or_path :str,
        dtype = torch.float16,
        device = "cuda:0") -> None:

        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.engine = InferenceEngine(max_length=max_length, model_name_or_path=model_name_or_path, dtype=dtype, device=device)
        self.callables = {}
        self.mempool = None
    @torch.inference_mode()
    def initialize_cuda_graph(self, 
            decoding_seqlens :List[int],
            n_warmups=3):
        gc.collect()
        self.mempool = torch.cuda.graphs.graph_pool_handle()
        for decoding_seqlen in decoding_seqlens:
            if decoding_seqlen not in self.callables and decoding_seqlen !=0:
                self.callables[decoding_seqlen] = capture_graph(
                    engine=self.engine,
                    decoding_seqlen=decoding_seqlen,
                    mempool=self.mempool,
                    n_warmups=n_warmups
                )
        self.engine.clear_kv()
    @torch.inference_mode()
    def graph_inference(self,
            input_ids: torch.LongTensor, 
            storage_ids :torch.LongTensor,
            position_ids: Optional[torch.LongTensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            debug :bool=False):

            dec_length = input_ids.shape[1]
            if debug:
                assert input_ids.shape[0] == 1
                assert storage_ids.shape[0] == dec_length
                assert position_ids.shape[0] == 1
                assert position_ids.shape[1] == dec_length
                assert attn_mask.shape[2] == dec_length
                assert attn_mask.shape[3] == self.engine.max_length
                assert attn_mask.shape[0] == 1
                assert attn_mask.shape[1] == 1
                assert attn_mask.device == self.device
                assert storage_ids.device == self.device
                assert position_ids.device == self.device
                assert input_ids.device == self.device
            if dec_length in self.callables:
                logits = self.callables[dec_length](input_ids, storage_ids, position_ids, attn_mask)
            else:
                logits = self.inference(input_ids, storage_ids, position_ids, attn_mask)
            return logits
    
    def clear_kv(self):
        self.engine.clear_kv()
    
    def initialize_kv(self, k_cache :torch.Tensor, v_cache :torch.Tensor, kv_len :int):
        self.engine.initialize_kv(k_cache, v_cache, kv_len)
    
    def get_kv_cache(self, in_place=False):
        return self.engine.get_kv_cache(in_place=in_place)
    
    def gather_kv(self, indices: list[int]):
        self.engine.gather_kv(indices)
    
    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor, 
            storage_ids :torch.LongTensor,
            position_ids: Optional[torch.LongTensor] = None,
            attn_mask: Optional[torch.Tensor] = None):
        
        return self.engine.model_run(input_ids=input_ids, storage_ids=storage_ids,
                    attention_mask=attn_mask, position_ids=position_ids)


class GraphInferenceEngineTG:
    def __init__(self, 
        max_length:int,
        model_name_or_path :str,
        dtype = torch.float16,
        device = "cuda:0",
        offloading = False) -> None:

        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.engine = InferenceEngineTG(max_length=max_length, model_name_or_path=model_name_or_path, dtype=dtype, device=device, offloading=offloading)
    def clear_kv(self):
        self.engine.clear_kv()
    
    def initialize_kv(self, k_cache :torch.Tensor, v_cache :torch.Tensor, kv_len :int):
        self.engine.initialize_kv(k_cache, v_cache, kv_len)
    
    def get_kv_cache(self, in_place=False):
        return self.engine.get_kv_cache(in_place=in_place)
    
    def gather_kv(self, indices: list[int]):
        self.engine.gather_kv(indices)
    
    def set_kv_len(self, kv_len :int):
        self.engine.set_kv_len(kv_len)
    
    @torch.no_grad()
    def inference(self,
            input_ids: torch.LongTensor, 
            storage_ids :torch.LongTensor,
            position_ids: Optional[torch.LongTensor] = None,
            attn_mask: Optional[torch.Tensor] = None):
        
        return self.engine.model_run(input_ids=input_ids, storage_ids=storage_ids,
                    attention_mask=attn_mask, position_ids=position_ids)







