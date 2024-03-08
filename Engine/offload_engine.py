from transformers import LlamaForCausalLM, LlamaConfig
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from time import sleep
import math
import torch.nn.functional as F
from torch import nn
from typing import List, Optional, Tuple, Union

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
class Offload_KV_Cache:

    def __init__(self, 
        config :LlamaConfig,
        batch_size :int = 1,
        max_length :int = 256, 
        device :str = 'cuda:0',
        dtype = torch.float16) -> None:
        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0

    def initialize_kv(self,
            k_cache :torch.Tensor,
            v_cache :torch.Tensor,
            kv_len :int):
        
        self.k_cache[...,:kv_len,:] = k_cache[...,:kv_len,:]
        self.v_cache[...,:kv_len,:] = v_cache[...,:kv_len,:]

        self.kv_offset = kv_len
        
        
    
    def gather_kv(self, indices: list[int]):

        self.k_cache[..., :len(indices), :] = self.k_cache[..., indices, :]
        self.v_cache[..., :len(indices), :] = self.v_cache[..., indices, :]

        self.k_cache[..., len(indices):, :] = 0.0
        self.v_cache[..., len(indices):, :] = 0.0

        self.kv_offset = len(indices)
    
    def gather_kv_incremental(self, indices: list[int], offset:int):

        self.k_cache[..., offset:offset + len(indices), :] = self.k_cache[..., indices, :]
        self.v_cache[..., offset:offset + len(indices), :] = self.v_cache[..., indices, :]

        self.k_cache[..., offset + len(indices):, :] = 0.0
        self.v_cache[..., offset + len(indices):, :] = 0.0

        self.kv_offset = offset + len(indices)


    
    def update_kv_cache(self, 
            new_k_cache :torch.Tensor,
            new_v_cache :torch.Tensor,
            layer_idx :int,
            ):
        
        input_length = new_k_cache.shape[-2]
        
        
        self.k_cache[layer_idx][..., self.kv_offset: self.kv_offset + input_length, :] = new_k_cache
        self.v_cache[layer_idx][..., self.kv_offset: self.kv_offset + input_length, :] = new_v_cache
        

        if layer_idx == self.num_layers - 1:
            self.kv_offset += input_length
            return self.k_cache[layer_idx][...,: self.kv_offset, :], self.v_cache[layer_idx][...,: self.kv_offset, :]
        return self.k_cache[layer_idx][...,: self.kv_offset + input_length, :], self.v_cache[layer_idx][...,: self.kv_offset + input_length, :]

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset = 0
    
    def get_usable_length(self, layer_idx:int, input_length :int):
            if layer_idx == self.num_layers - 1:
                return self.kv_offset
            else:
                return self.kv_offset + input_length
    
    def set_kv_len(self, kv_len :int):
            self.kv_offset = kv_len

class LlamaLayer:
    def __init__(self, layer_idx) -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.cos_cache :torch.Tensor = None
        self.sin_cache :torch.Tensor = None

        self.layer_idx = layer_idx
    
    def init_parameters(self, hf_layer: LlamaDecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach().pin_memory()
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach().pin_memory()
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach().pin_memory()
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach().pin_memory()

        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach().pin_memory()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach().pin_memory()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach().pin_memory()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

        self.cos_cache :torch.Tensor= hf_layer.self_attn.rotary_emb.cos_cached
        self.sin_cache :torch.Tensor= hf_layer.self_attn.rotary_emb.sin_cached
    
    def init_gpu(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device)

        self.cos_cache = self.cos_cache.to(device)
        self.sin_cache = self.sin_cache.to(device)


class LlamaLayerBuffer:
    def __init__(self, device:str = 'cuda:0') -> None:
        self.device = device
    
    def init_space(self, layer: LlamaLayer):

        self.wq_buffer = torch.zeros_like(layer.wq).to(self.device)
        self.wk_buffer = torch.zeros_like(layer.wk).to(self.device)
        self.wv_buffer = torch.zeros_like(layer.wv).to(self.device)
        self.wo_buffer = torch.zeros_like(layer.wo).to(self.device)


        self.gate_proj_buffer = torch.zeros_like(layer.gate_proj).to(self.device)
        self.up_proj_buffer = torch.zeros_like(layer.up_proj).to(self.device)
        self.down_proj_buffer = torch.zeros_like(layer.down_proj).to(self.device)
    
    def sync_copy(self, layer: LlamaLayer):

        self.wq_buffer.copy_(layer.wq, non_blocking=True)
        self.wk_buffer.copy_(layer.wk, non_blocking=True)
        self.wv_buffer.copy_(layer.wv, non_blocking=True)
        self.wo_buffer.copy_(layer.wo, non_blocking=True)

        self.gate_proj_buffer.copy_(layer.gate_proj, non_blocking=True)
        self.up_proj_buffer.copy_(layer.up_proj, non_blocking=True)
        self.down_proj_buffer.copy_(layer.down_proj, non_blocking=True)

class Llama:
    def __init__(self, 
        model_name: str,
        max_length :int = 256, 
        device :str = 'cuda:0',
        dtype = torch.float16) -> None:
        
        self.device = device
        self.dtype = dtype
        self.config = LlamaConfig.from_pretrained(model_name)
        self.model_name = model_name
        self.max_length = max_length
        self.kv_cache = Offload_KV_Cache(self.config, max_length=max_length, device=device, dtype=dtype)
        
        self.init_parameters()
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
    def init_parameters(self):

        hf_model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)
        self.embed_tokens = hf_model.model.embed_tokens.weight.detach().to(self.device)
        self.lm_head = hf_model.lm_head.weight.detach().to(self.device)

        self.norm_weight = hf_model.model.norm.weight.detach().to(self.device)
        self.norm_variance_epsilon = hf_model.model.norm.variance_epsilon
        self.layers :list[LlamaLayer] = []
        for idx, hf_layer in enumerate(hf_model.model.layers):
            layer = LlamaLayer(idx)
            layer.init_parameters(hf_layer=hf_layer)
            layer.init_gpu(self.device)
            self.layers.append(layer)
            del hf_layer
        
        self.num_layers = len(self.layers)
        self.buffer = LlamaLayerBuffer(self.device)
        self.buffer.init_space(self.layers[0])

    def layer_compute(self, 
            buffer: LlamaLayerBuffer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor, 
            attention_mask: torch.FloatTensor):

        residual = hidden_states

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.layers[layer_idx].input_layernorm_variance_epsilon)
        hidden_states = self.layers[layer_idx].input_layernorm_weight * hidden_states.to(input_dtype)

        
        bsz, q_len, _ = hidden_states.size()

        query_states = F.linear(hidden_states, buffer.wq_buffer)
        key_states = F.linear(hidden_states, buffer.wk_buffer)
        value_states = F.linear(hidden_states, buffer.wv_buffer)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        
        kv_seq_len = key_states.shape[-2]
        kv_seq_len += self.kv_cache.kv_offset

        cos = self.layers[layer_idx].cos_cache[:kv_seq_len].to(value_states.dtype)
        sin = self.layers[layer_idx].sin_cache[:kv_seq_len].to(value_states.dtype)

        
        

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        
        key_states, value_states = self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        hidden_states = F.linear(attn_output, buffer.wo_buffer)
        
        hidden_states = residual + hidden_states

        
        residual = hidden_states
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.layers[layer_idx].post_attention_layernorm_variance_epsilon)
        
        hidden_states = self.layers[layer_idx].post_attention_layernorm_weight * hidden_states.to(input_dtype)
        
        up = F.linear(hidden_states, buffer.up_proj_buffer)
        gate = F.linear(hidden_states, buffer.gate_proj_buffer)
        gate = F.silu(gate)
        hidden_states = gate * up
        hidden_states = F.linear(hidden_states, buffer.down_proj_buffer)
        hidden_states = residual + hidden_states
        return hidden_states



    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.FloatTensor):
        
        hidden_states = F.embedding(input_ids, self.embed_tokens)
        for idx in range(self.num_layers):
            self.buffer.sync_copy(self.layers[idx])
            hidden_states = self.layer_compute(self.buffer, idx, hidden_states, position_ids, attention_mask)
        
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.norm_variance_epsilon)
        
        hidden_states = self.norm_weight * hidden_states.to(input_dtype)
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits

class OffloadEngine:
    def __init__(self,
        max_length:int,
        model_name_or_path :str,
        dtype = torch.float16,
        device = "cuda:0") -> None:

        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.engine = Llama(model_name=model_name_or_path, max_length=max_length, device=device, dtype=dtype)
    def clear_kv(self):
        self.engine.kv_cache.clear()
    
    def initialize_kv(self, k_cache :torch.Tensor, v_cache :torch.Tensor, kv_len :int):
        self.engine.kv_cache.initialize_kv(k_cache, v_cache, kv_len)
    
    def get_kv_cache(self, in_place=False):
        if not in_place:
            return self.engine.kv_cache.k_cache.clone(), self.engine.kv_cache.v_cache.clone()
        else:
            return self.engine.kv_cache.k_cache, self.engine.kv_cache.v_cache
    def gather_kv(self, indices: list[int]):
        self.engine.kv_cache.gather_kv(indices)
    
    def set_kv_len(self, kv_len :int):
        self.engine.kv_cache.set_kv_len(kv_len)
    
    def inference(self,
            input_ids: torch.LongTensor, 
            storage_ids :torch.LongTensor,
            position_ids: Optional[torch.LongTensor] = None,
            attn_mask: Optional[torch.Tensor] = None):
        
            return self.engine.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attn_mask)



if __name__ == "__main__":
    llm = OffloadEngine(max_length = 256, model_name_or_path = "meta-llama/Llama-2-70b-hf")
    #llm = Llama("meta-llama/Llama-2-70b-hf")
    
    input_ids = torch.LongTensor([
        [
    1, 21429, 29899,  6451, 22545,  1078,   505
        ]
    ]).cuda()
    position_ids = torch.LongTensor([
        [
    0, 1, 2, 3, 4, 5, 6
        ]
    ]).cuda()

    attention_mask = _make_causal_mask((1,8), torch.float16, device="cuda:0")

    logits = llm.inference(input_ids=input_ids, position_ids=position_ids, attn_mask=attention_mask[:-1, : -1][None, None, :, :], storage_ids=None)
    print(logits)
    new_input_ids = torch.LongTensor([
        [
            1407
        ]
    ]).cuda()

    new_position_ids = torch.LongTensor([
        [
            7
        ]
    ]).cuda()

    logits = llm.inference(input_ids=new_input_ids, position_ids=new_position_ids, attn_mask=attention_mask[-1:][None, None, :, :], storage_ids=None)
    print(logits)
