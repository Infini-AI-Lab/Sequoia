import torch
import dataclasses
from torch.nn.functional import softmax

def get_residual(p: torch.Tensor, q:torch.Tensor):
    residual = (p - q).relu_()
    residual = residual / (residual.sum(dim=-1).unsqueeze(-1))
    return residual

def sampling_without_replacement(
        sampling_logits: torch.Tensor, 
        rand: torch.Tensor,  
        num_samples: int,
        temperature :float):

        sampling_q = softmax(sampling_logits / temperature, dim=-1)
        position = (rand.log()/sampling_q).topk(k=num_samples).indices.flatten()
        return position

def sampling_with_replacement(
        sampling_logits: torch.Tensor,   
        num_samples: int,
        temperature :float):

        #sampling_q = softmax(sampling_logits / temperature, dim=-1)
        sampling_q = softmax(sampling_logits / temperature, dim=-1)    
        position = sampling_q.multinomial(num_samples=num_samples, replacement=False).flatten()
        return position
def sampling_argmax(
        sampling_logits: torch.Tensor, 
        num_samples: int):
        return sampling_logits.topk(k=num_samples).indices.flatten()

def expand_kv(kv_cache, k):
    kv_shape = kv_cache[0][0].shape
    new_kv_cache = ()
    for kv in kv_cache:
        new_kv_cache = new_kv_cache + ([kv[0].expand(k, kv_shape[1], kv_shape[2], kv_shape[3]), 
                kv[1].expand(k, kv_shape[1], kv_shape[2], kv_shape[3])],)
    return new_kv_cache

def cat_kv(old_kv, delta_kv, cut_len :int):
    new_kv_cache = ()
    for i in range(len(old_kv)):
          k = torch.cat([old_kv[i][0], delta_kv[i][0][..., -cut_len:, :]], dim=-2)
          v = torch.cat([old_kv[i][1], delta_kv[i][1][..., -cut_len:, :]], dim=-2)
          new_kv_cache += ([k,v],)
    return new_kv_cache
    
    
def make_tree_attention_mask(
        prefix_len :int,
        gen_len :int,
        ancestors :list[list[int]],
        device ="cpu",
        dtype = torch.float32
    ) -> torch.FloatTensor:
    tree_mask = torch.full((gen_len, gen_len + prefix_len), torch.finfo(dtype).min, dtype=dtype).to(device=device)
    for idx, ancestor in enumerate(ancestors):
        if len(ancestor) > 0:
            tree_mask[idx][ancestor] = 0.0
    return tree_mask[None, None, :, :]


def get_sampling_logits(logits :torch.Tensor, top_p:float, T: float, replicate = False):
    if replicate:
        logits = logits.clone()
    if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                filter = cumulative_probs > top_p
                filter[..., 1:] = filter[..., :-1].clone()
                filter[..., 0] = 0
                indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                logits[indices_to_remove] = float('-inf')
    return logits

def select_kv(kv_cache: tuple[list[torch.FloatTensor]], indices: list[int]):
        new_kv_cache = ()
        for k,v in kv_cache:
             k = k[..., indices, :]
             v = v[..., indices, :]
             new_kv_cache += ([k,v],)
        return new_kv_cache

@dataclasses.dataclass
class ChildrenAccept:
    accept_mark :int = None
    token :int = None
    position :int = None
    successor_order :int = -1
    residual :torch.FloatTensor = None

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    Copied from Huggingface
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask

def cuda_graph_for_residual(device="cuda:0", dtype=torch.float16, dim=32000, n_warmups=3, mempool=None):
    static_p = torch.full((dim,), 1, dtype=dtype, device=device)
    static_q = torch.full((dim,), 0, dtype=dtype, device=device)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_residual = get_residual(
                    static_p,
                    static_q
                    )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
         static_residual = get_residual(
                    static_p,
                    static_q
                    )
    def run(p, q):
        static_p.copy_(p)
        static_q.copy_(q)
        graph.replay()
        return static_residual.clone()
    
    return run

def cuda_graph_for_sampling_without_replacement(
                device="cuda:0", dtype=torch.float16, 
                dim=32000, max_length=384, 
                n_warmups=3, mempool=None,
                idx_len = 8, num_samples = 16,
                temperature = 0.6, tree_size = 64):
    
    static_sampling_logits = torch.full((idx_len, dim), 1, dtype=dtype, device=device)
    static_rand = torch.empty((idx_len, dim), dtype=dtype, device=device).uniform_()

    

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_position = sampling_without_replacement(
                 static_sampling_logits,
                 static_rand,
                 num_samples,
                 temperature
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_position = sampling_without_replacement(
                 static_sampling_logits,
                 static_rand,
                 num_samples,
                 temperature
            )
    def run(draft_logits, rand_vector):
        static_sampling_logits.copy_(draft_logits)
        static_rand.copy_(rand_vector)
        graph.replay()
        return static_position.clone()
    
    return run

def cuda_graph_for_sampling_argmax(
                device="cuda:0", dtype=torch.float16, 
                dim=32000, max_length=384, 
                n_warmups=3, mempool=None,
                idx_len = 8, num_samples = 16,
                temperature = 0.6, tree_size = 64):
    
    static_sampling_logits = torch.full((idx_len, dim), 1, dtype=dtype, device=device)
    

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_position = sampling_argmax(
                 static_sampling_logits,
                 num_samples
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_position = sampling_argmax(
                 static_sampling_logits,
                 num_samples
            )
    def run(draft_logits):
        static_sampling_logits.copy_(draft_logits)
        graph.replay()
        return static_position.clone()
    
    return run


def cuda_graph_for_sampling_with_replacement(
                device="cuda:0", dtype=torch.float16, 
                dim=32000, max_length=384, 
                n_warmups=3, mempool=None,
                idx_len = 8, num_samples = 16,
                temperature = 0.6, tree_size = 64):
    
    static_sampling_logits = torch.full((idx_len, dim), 1, dtype=dtype, device=device)
    

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_position = sampling_with_replacement(
                 static_sampling_logits,
                 num_samples,
                 temperature
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_position = sampling_with_replacement(
                 static_sampling_logits,
                 num_samples,
                 temperature
            )
    def run(draft_logits):
        static_sampling_logits.copy_(draft_logits)
        graph.replay()
        return static_position.clone()
    
    return run
    
        




