from .Llama_modules import LlamaDecoderLayer_FI, LlamaRMSNorm_FI, LlamaDecoderLayer_TG
from transformers import LlamaPreTrainedModel, LlamaConfig
import torch
import torch.nn as nn
import math
import warnings
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from torch.nn import CrossEntropyLoss
from .Llama_KV import KV_Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
class LlamaModel_FI(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer_FI(config=config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm_FI(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    
    def forward(
        self,
        input_ids: torch.LongTensor,
        max_length :int,
        storage_ids :torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: KV_Cache = None,
        debug :bool = False,
    ):
        
        
        
        inputs_embeds = self.embed_tokens(input_ids)
        
        # embed positions
        hidden_states = inputs_embeds
        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):            
            layer_outputs = decoder_layer(
                    hidden_states,
                    max_length=max_length,
                    storage_ids=storage_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                    debug=debug
                )
            hidden_states = layer_outputs
            
        hidden_states = self.norm(hidden_states)
       
        return hidden_states

class LlamaModel_TG(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer_TG(config=config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm_FI(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    
    def forward(
        self,
        input_ids: torch.LongTensor,
        max_length :int,
        storage_ids :torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: KV_Cache = None,
        debug :bool = False,
    ):
        
        
        
        inputs_embeds = self.embed_tokens(input_ids)
        
        # embed positions
        hidden_states = inputs_embeds
        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):            
            layer_outputs = decoder_layer(
                    hidden_states,
                    max_length=max_length,
                    storage_ids=storage_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                    debug=debug
                )
            hidden_states = layer_outputs
        hidden_states = self.norm(hidden_states)
       
        return hidden_states
class LlamaForCausalLM_FI(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel_FI(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    
    def forward(
        self,
        input_ids: torch.LongTensor,
        max_length :int,
        storage_ids :torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: KV_Cache = None,
        debug: bool = False
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            max_length=max_length,
            storage_ids=storage_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            debug=debug,
        )

        hidden_states = outputs
       
        logits = self.lm_head(hidden_states)
        

        return logits


class LlamaForCausalLM_TG(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel_TG(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    
    def forward(
        self,
        input_ids: torch.LongTensor,
        max_length :int,
        storage_ids :torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: KV_Cache = None,
        debug: bool = False
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            max_length=max_length,
            storage_ids=storage_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            debug=debug,
        )

        hidden_states = outputs
       
        logits = self.lm_head(hidden_states)
        return logits
