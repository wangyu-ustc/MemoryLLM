import time
import math
from .modeling_llama import LlamaForCausalLM, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from MemoryLLM.memoryllm.modules.base import BaseMemoryModel, MemoryLMOutputWithPastAndCrossAttentions
from typing import Optional, Tuple, Union, List
from torch.nn import CrossEntropyLoss
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache

logger = logging.get_logger(__name__)

class MemoryModule(nn.Module):

    def __init__(self, L, num_tokens, d, requires_grad=False):
        super(MemoryModule, self).__init__()
        self.memory = nn.Parameter(torch.randn([L, num_tokens, d]))
        self.memory.requires_grad = requires_grad
        print(f"Memory Pool Parameters: {len(self.memory.reshape(-1)) / 1_000_000_000:.4f} B")
    
class LlamaDropMemoryModel(LlamaForCausalLM, BaseMemoryModel):
    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)
        BaseMemoryModel.__init__(self, config)
        
        self.config = config
        self.L = config.num_hidden_layers
        self.d = config.hidden_size
        self.num_blocks = config.num_blocks
        self.num_tokens = config.num_tokens
        self.bos_token_id = config.bos_token_id
        self.virtual_num_blocks = config.virtual_num_blocks if hasattr(config, "virtual_num_blocks") else None

        self.add_bos_embedding = config.add_bos_embedding
        self.shrink_to_one_embedding = config.shrink_to_one_embedding
        self.drop_memory_per_layer = config.drop_memory_per_layer if hasattr(config, "drop_memory_per_layer") else False
        self.add_decoder_lora = config.add_decoder_lora
        self.tune_special_tokens = config.tune_special_tokens
        self.special_token_ids = config.special_token_ids
        self.maintain_memory_keys = config.maintain_memory_keys if hasattr(config, "maintain_memory_keys") else False

        self.skip_logits_except_the_last_hidden_state = False

        self.add_memory_embedding = False
        if hasattr(config, "add_memory_embedding") and config.add_memory_embedding:
            if config.spread_embeddings:
                self.memory_embeddings = nn.Parameter(torch.zeros([self.num_tokens, self.d]))
            else:
                self.memory_embeddings = nn.Parameter(torch.zeros([1, self.d]))
                
            self.add_memory_embedding = True

        self.memory = nn.Parameter(torch.randn([self.L, self.num_blocks * self.num_tokens, self.d]))
        print(f"Memory Pool Parameters: {len(self.memory.reshape(-1)) / 1_000_000_000:.4f} B")
        self.memory.requires_grad = False
        self.memory_keys = None

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.add_positional_embedding = False

        self.new_memory_positional_emb = nn.Parameter(torch.zeros([1, 1, self.d]))

        if config.add_bos_embedding:
            self.bos_embedding = nn.Parameter(torch.randn([self.L, 1, self.d]))

        self.add_selector = True if hasattr(config, "add_selector") and config.add_selector else False
        if self.tune_special_tokens:
            self.special_token_embeddings = nn.Parameter(torch.zeros([len(self.special_token_ids), self.d]))

        self._detach_memory = False

    def set_exclude_layers(self, layer_indices):
        self.exclude_layers = layer_indices

    def super_forward(self, *args, **kwargs):
        return LlamaForCausalLM.forward(self, *args, **kwargs)

    def detach_memory(self):
        self._detach_memory = True
    
    def attach_memory(self):
        self._detach_memory = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        delta_memory: Optional[List[List[torch.FloatTensor]]] = None,
        labels: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_delta_memory: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_injection: Optional[bool] = None,
        cat_to_maximum_memory: Optional[bool] = False,
        output_retriever_weights: Optional[bool] = False,
        return_full_retriever_weights: Optional[bool] = False,
        random_retriever_length: Optional[bool] = False,
        training: Optional[bool] = False,
        # return_memory_keys: Optional[bool] = False,
        encoder_query_indices: Optional[List[int]] = None,
        parallel_injection: Optional[bool] = False,
        scale_randomized_poe_ratio: Optional[bool] = None,
        scale_randomized_poe_ratio_for_memory: Optional[bool] = None,
        debug: Optional[bool] = False
    ) -> Union[Tuple, MemoryLMOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_injection is None:
            is_injection = output_delta_memory

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.model.gradient_checkpointing and self.model.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        
        if inputs_embeds is None:
            
            # assert self.model.embed_tokens.weight.requires_grad == False
            inputs_embeds = self.model.embed_tokens(input_ids)

            if self.tune_special_tokens:

                inputs_embeds = inputs_embeds.detach()
                new_inputs_embeds = []
                for idx, token_id in enumerate(input_ids[0]):
                    
                    if token_id in self.special_token_ids:
                        new_inputs_embeds.append(self.special_token_embeddings[self.special_token_ids.index(token_id.item())])
                    else:
                        new_inputs_embeds.append(inputs_embeds[0, idx])
                    
                inputs_embeds = torch.stack(new_inputs_embeds).unsqueeze(0)

            if self.add_memory_embedding and is_injection:
                if inputs_embeds.shape[1] < self.num_tokens:
                    inputs_embeds = torch.cat([
                        inputs_embeds,
                        self.memory_embeddings.unsqueeze(0).repeat(1, self.num_tokens - inputs_embeds.shape[1], 1)
                    ], dim=1)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        # if cache_position is None:
        # TODO: currently ignore cache_position
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        
        if self.initialized:
            if past_seen_tokens > 0:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
                if self._detach_memory:
                    cache_position += self.num_tokens * self.num_blocks
            else:
                if is_injection:
                    if self.add_memory_embedding:
                        cache_position = torch.arange(
                            0, inputs_embeds.shape[1] + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
                    else:
                        cache_position = torch.arange(
                            0, inputs_embeds.shape[1] + self.num_tokens + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
                elif delta_memory is not None and delta_memory.shape[2] == self.num_tokens and not cat_to_maximum_memory:
                    cache_position = torch.arange(
                        0, inputs_embeds.shape[1] + self.num_tokens + int(self.add_bos_embedding), device=inputs_embeds.device
                    )
                else:
                    if self._detach_memory:
                        cache_position = torch.arange(
                            self.num_tokens * self.num_blocks + int(self.add_bos_embedding), 
                            inputs_embeds.shape[1] + self.num_tokens * self.num_blocks + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
                        cache_position = torch.cat([
                            torch.tensor([0], device=inputs_embeds.device), cache_position
                        ])

                    else:
                        cache_position = torch.arange(
                            0, inputs_embeds.shape[1] + self.num_tokens * self.num_blocks + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
        
        else:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)
        # TODO: currently ignore position_ids
            
        position_ids = cache_position.unsqueeze(0)

        if not is_injection and self.initialized and (not (delta_memory is not None and not cat_to_maximum_memory)):

            if scale_randomized_poe_ratio_for_memory is not None:
                input_poes = position_ids[:, 1:-inputs_embeds.shape[1]]
                start_poe = input_poes[0, 0]
                # choose input_poes.shape[1] positions from int(2048 * scale_randomized_poe_ratio):
                all_indices = torch.randperm(int((self.num_blocks * self.num_tokens) * scale_randomized_poe_ratio_for_memory)) + start_poe.item()
                input_poes = all_indices[:input_poes.shape[1]].sort()[0]
                position_ids[:, 1:-inputs_embeds.shape[1]] = input_poes.to(position_ids.device)
                position_ids[:, -inputs_embeds.shape[1]:] += (scale_randomized_poe_ratio_for_memory - 1) * (self.num_blocks * self.num_tokens)

            if scale_randomized_poe_ratio is not None:
                input_poes = position_ids[:, -inputs_embeds.shape[1]:]
                start_poe = input_poes[0, 0]
                # choose input_poes.shape[1] positions from int(2048 * scale_randomized_poe_ratio):
                all_indices = torch.randperm(int(2048 * scale_randomized_poe_ratio)) + start_poe.item()
                input_poes = all_indices[:input_poes.shape[1]].sort()[0]        
                position_ids[:, -inputs_embeds.shape[1]:] = input_poes.to(position_ids.device)

        # TODO: check why sometimes there are zeros in attention_mask
        # causal_mask = self.model._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )
        causal_mask = None

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_delta_memory = [] if output_delta_memory else None
        all_retriever_weights = () if output_retriever_weights else None
        all_encoder_retriever_weights = () if (output_retriever_weights and encoder_query_indices is not None) else None

        if self.add_decoder_lora:

            if is_injection or (delta_memory is not None and delta_memory.shape[2] == self.num_tokens and not cat_to_maximum_memory):
                for name, module in self.named_modules():
                    if hasattr(module, "_active_adapter"):
                        module._active_adapter = ['default',]
            else:
                for _, module in self.named_modules():
                    if hasattr(module, "_active_adapter"):
                        module._active_adapter = ['decoder_adapter']            

        for idx, decoder_layer in enumerate(self.model.layers):
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if past_key_values is None or past_key_values.get_seq_length(layer_idx=idx) == 0:

                if is_injection or (not self._detach_memory):
                    
                    if self.add_memory_embedding and is_injection and self.initialized:
                        hidden_states = torch.cat([
                            self.bos_embedding[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1),
                            hidden_states,
                        ], dim=1)

                    else:
                        hidden_states = self.cat_memory_and_hiddens(idx,
                                                        hidden_states=hidden_states,
                                                        delta_memory=delta_memory,
                                                        is_injection=is_injection,
                                                        cat_to_maximum_memory=cat_to_maximum_memory)
                        
                else:
                    hidden_states = torch.cat([
                        self.bos_embedding[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1),
                        hidden_states
                    ], dim=1)
                
                prefix_token_length = hidden_states.shape[1] - inputs_embeds.shape[1] if self.initialized else 0

                if is_injection and prefix_token_length > 0:
                    prefix_token_length = min(prefix_token_length, hidden_states.shape[1] - self.num_tokens)

                if is_injection:
                    if self.new_memory_positional_emb.device != hidden_states.device:
                        hidden_states[:, -self.num_tokens:] += self.new_memory_positional_emb.to(hidden_states.device)
                    else:
                        hidden_states = torch.cat([
                            hidden_states[:, :-self.num_tokens],
                            hidden_states[:, -self.num_tokens:] + self.new_memory_positional_emb
                        ], dim=1)

            else:
                prefix_token_length = 0

            # if self.model.gradient_checkpointing and self.model.training:
            if self.model.gradient_checkpointing and training:

                layer_outputs = self.model._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    prefix_token_length,
                    output_retriever_weights,
                )

            else:

                assert not (is_injection and encoder_query_indices)
                
                try:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        prefix_token_length=prefix_token_length,
                        output_retriever_weights=output_retriever_weights,
                        return_full_retriever_weights=return_full_retriever_weights,
                        random_retriever_length=random_retriever_length,
                        encoder_query_indices=encoder_query_indices[idx] if encoder_query_indices is not None else None,
                        encoder_attention_mask=encoder_attention_mask,
                        training=training,
                    )
                except:
                    import ipdb; ipdb.set_trace()

            hidden_states = layer_outputs[0]

            if output_delta_memory:
                
                if parallel_injection:
                    all_delta_memory.append(hidden_states)
                else:
                    all_delta_memory.append(hidden_states[:, -self.num_tokens:])
                
            hidden_states = hidden_states[:, -inputs_embeds.shape[1]:]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if output_retriever_weights:
                if encoder_query_indices is not None:
                    retriever_weights = layer_outputs[-2]
                    encoder_retriever_weights = layer_outputs[-1]
                    all_encoder_retriever_weights += (encoder_retriever_weights,)
                else:
                    retriever_weights = layer_outputs[-1]
                if retriever_weights is not None:
                    all_retriever_weights += (retriever_weights,)

        hidden_states = self.model.norm(hidden_states)
            
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if output_delta_memory:

            if all_delta_memory[0].device != all_delta_memory[-1].device:
                assert not self.training
                device = all_delta_memory[0].device
                all_delta_memory = [x.to(device) for x in all_delta_memory]
                delta_memory = torch.stack(all_delta_memory, dim=0).transpose(0, 1)

            else:
                delta_memory = torch.stack(all_delta_memory, dim=0).transpose(0, 1)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if self.skip_logits_except_the_last_hidden_state:
                logits = self.lm_head(hidden_states[:, -1:])
            else:
                logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            return tuple(v for v in [loss, logits, hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return MemoryLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            delta_memory=delta_memory,
            retriever_weights=all_retriever_weights if (all_retriever_weights is not None and len(all_retriever_weights) > 0) else None,
            encoder_retriever_weights=all_encoder_retriever_weights if (all_encoder_retriever_weights is not None and len(all_encoder_retriever_weights) > 0) else None
            # memory_keys=all_memory_keys if (all_memory_keys is not None and len(all_memory_keys) > 0) else None
        )



class LlamaDropMemoryTreeModel(LlamaForCausalLM, BaseMemoryModel):
    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)
        BaseMemoryModel.__init__(self, config)
        
        self.config = config
        self.L = config.num_hidden_layers
        self.d = config.hidden_size
        self.num_blocks = config.num_blocks
        self.num_tokens = config.num_tokens
        self.bos_token_id = config.bos_token_id

        self.add_bos_embedding = config.add_bos_embedding
        self.shrink_to_one_embedding = config.shrink_to_one_embedding
        self.drop_memory_per_layer = config.drop_memory_per_layer if hasattr(config, "drop_memory_per_layer") else False
        self.add_decoder_lora = config.add_decoder_lora
        self.tune_special_tokens = config.tune_special_tokens
        self.special_token_ids = config.special_token_ids
        self.maintain_memory_keys = config.maintain_memory_keys if hasattr(config, "maintain_memory_keys") else False
        self.dropping_interval = config.dropping_interval if hasattr(config, "dropping_interval") else 16
        self.min_num_tokens = config.min_num_tokens if hasattr(config, "min_num_tokens") else 25
        self.fix_poe_for_encoder = config.fix_poe_for_encoder if hasattr(config, "fix_poe_for_encoder") else False

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.add_positional_embedding = False

        if config.add_bos_embedding:
            self.bos_embedding = nn.Parameter(torch.randn([self.L, 1, self.d]))

        self.add_selector = True if hasattr(config, "add_selector") and config.add_selector else False
        if self.tune_special_tokens:
            self.special_token_embeddings = nn.Parameter(torch.zeros([len(self.special_token_ids), self.d]))

        # long-term memory configurations
        if hasattr(config, "ltm_configs"):

            self.use_ltm = True

            self.ltm = [None] * self.L
            self.ltm_keys = [None] * self.L
            self.ltm_ages = [None] * self.L
            self.ltm_recall_frequencies = [None] * self.L

            if hasattr(config.ltm_configs, "use_cache") and config.ltm_configs.use_cache:
                self.ltm_use_cache = True
                self.ltm_cache = [None] * self.L
                self.ltm_keys_cache = [None] * self.L
                self.ltm_ages_cache = [None] * self.L
                self.ltm_recall_frequencies = [None] * self.L
            else:
                self.ltm_use_cache = False

            self.initial_rf_when_moving_stm_to_ltm = config.ltm_configs.get("initial_rf_when_moving_stm_to_ltm", 10)
            self.decay_frequency = config.ltm_configs.get("decay_frequency", 0.05)
            self.num_ltm_blocks = config.ltm_configs.get("num_ltm_blocks", 20)
            
            assert self.num_blocks > self.num_ltm_blocks

            ideal_list = [self.num_tokens * (1 - 1/(self.num_blocks))**i for i in range(1000)]
            final_list = [math.ceil(x / self.dropping_interval) * self.dropping_interval for x in ideal_list]
            final_list = [x for x in final_list if x >= self.min_num_tokens]
            while sum(final_list) > (self.num_blocks - self.num_ltm_blocks) * self.num_tokens:
                final_list = final_list[:-1]

            print(f"Short-Term Memory can last {len(final_list)} steps")
            print(f"Last chunk has {final_list[-1]} tokens")

            remaining_indicators = []
            importance_indicators = []

            for idx, frequency in enumerate(final_list[:-1]):
                if config.important_tokens == 'right':
                    remaining_indicators.extend(
                        [1] * final_list[idx+1] + [0] * (frequency - final_list[idx+1])
                    )
                    importance_indicators.extend(
                        [1] * final_list[-1] + [0] * (frequency - final_list[-1])
                    )
                else:
                    remaining_indicators.extend(
                        [0] * (frequency - final_list[idx+1]) + [1] * final_list[idx+1]
                    )
                    importance_indicators.extend(
                        [0] * final_list[-1] + [1] * (frequency - final_list[-1])
                    )

            importance_indicators.extend([1] * final_list[-1])
            remaining_indicators.extend([0] * final_list[-1])
            remaining_indicators = remaining_indicators[::-1]
            importance_indicators = importance_indicators[::-1]
            self.final_list = final_list[::-1]
            self.remaining_indicators = torch.tensor(remaining_indicators, dtype=torch.bool)
            self.importance_indicators = torch.tensor(importance_indicators, dtype=torch.bool)
            self.remaining_indices = torch.where(self.remaining_indicators)[0]
            self.unique_memory_numbers = torch.unique(torch.tensor(final_list))
            self.num_memory_tokens = sum(final_list)

            self.memory = nn.Parameter(torch.randn([self.L, self.num_memory_tokens, self.d]))
            print(f"Memory Pool Parameters: {len(self.memory.reshape(-1)) / 1_000_000_000:.4f} B")
            self.memory_keys = None

            self.num_ltm_tokens = self.num_blocks * self.num_tokens - self.num_memory_tokens

        else:

            self.use_ltm = False

            # get the memory dropping schedule
            ideal_list = [self.num_tokens * (1 - 1/self.num_blocks)**i for i in range(1000)]
            final_list = [math.ceil(x / self.dropping_interval) * self.dropping_interval for x in ideal_list]
            final_list = [x for x in final_list if x >= self.min_num_tokens]
            while sum(final_list) > self.num_blocks * self.num_tokens:
                final_list = final_list[:-1]
            remaining_indicators = []
            for idx, frequency in enumerate(final_list[:-1]):
                if config.important_tokens == 'right':
                    remaining_indicators.extend(
                        [1] * final_list[idx+1] + [0] * (frequency - final_list[idx+1])
                    )
                else:
                    remaining_indicators.extend(
                        [0] * (frequency - final_list[idx+1]) + [1] * final_list[idx+1]
                    )
            remaining_indicators.extend([0] * final_list[-1])
            remaining_indicators = remaining_indicators[::-1]
            self.final_list = final_list[::-1]
            self.remaining_indicators = torch.tensor(remaining_indicators, dtype=torch.bool)
            self.remaining_indices = torch.where(self.remaining_indicators)[0]
            self.unique_memory_numbers = torch.unique(torch.tensor(final_list))

            self.num_memory_tokens = sum(final_list)

            self.memory = nn.Parameter(torch.randn([self.L, self.num_memory_tokens, self.d]))
            print(f"Memory Pool Parameters: {len(self.memory.reshape(-1)) / 1_000_000_000:.4f} B")
            self.memory_keys = None

        # r_{t+1} = r_t + 32 * r_0 + 2560 * 1 - s * num => num = (2560 + 32 * r_0) / s
        # if r_0 = 100, s = 0.05, then we have `num = (2560 + 3200) / 0.05 = 115200`
        # Then every token would be maintained in LTM for r_0 / s = 2000 steps, leading to 2000 * 512 = 1_024_000 tokens

        self.important_tokens = config.important_tokens
        self.memory.requires_grad = False
        self._detach_memory = False

        self.add_memory_embedding = False

        if hasattr(config, "add_memory_embedding") and config.add_memory_embedding:
            
            if config.spread_embeddings:
                
                # self.memory_embeddings = nn.Parameter(torch.zeros([self.num_tokens, self.d]))
                # np.min(self.final_list), ..., 256 (+self.dropping_interval each time)
                
                memory_position_ids = []
                count = 0

                # self.unique_memory_numbers needs to be ascending
                self.unique_memory_numbers = np.sort(self.unique_memory_numbers).tolist()

                for num in self.unique_memory_numbers:

                    if num == np.min(self.final_list):
                        memory_position_ids.extend([count] * num)
                    else:
                        memory_position_ids.extend([count] * (num - last_num))

                    last_num = num
                    count += 1

                # self.memory_embeddings = nn.ModuleList([nn.Parameter(torch.zeros([1, self.d])) for _ in range(len(self.unique_memory_numbers))])
                self.memory_embeddings = nn.Parameter(torch.zeros([len(self.unique_memory_numbers), self.d]))

                if config.important_tokens == 'right':
                    memory_position_ids = memory_position_ids[::-1]

                self.memory_position_ids = torch.tensor(memory_position_ids)

                # self.memory_position_ids = [(num - np.min(self.final_list)) // self.dropping_interval for num in self.unique_memory_numbers]

            else:
                self.memory_embeddings = nn.Parameter(torch.zeros([1, self.d]))
            self.add_memory_embedding = True

        else:

            self.new_memory_positional_emb = nn.Parameter(torch.zeros([1, 1, self.d]))

    def set_exclude_layers(self, layer_indices):
        self.exclude_layers = layer_indices

    def super_forward(self, *args, **kwargs):
        return LlamaForCausalLM.forward(self, *args, **kwargs)

    def detach_memory(self):
        self._detach_memory = True
    
    def attach_memory(self):
        self._detach_memory = False

    def update_memory_with_delta_memory(self, delta_memory, cached_contexts_indicators=None, **kwargs):
        
        if len(delta_memory.shape) == 4:
            delta_memory = delta_memory.detach()[0]

        if self.initialized == 0:
            if delta_memory.shape[1] < (self.num_memory_tokens):
                if ((self.num_memory_tokens) % delta_memory.shape[1]) == 0:
                    delta_memory = torch.cat(
                        [delta_memory] * ((self.num_memory_tokens) // delta_memory.shape[1]), dim=1
                    )
                else:
                    delta_memory = torch.cat(
                        [delta_memory] * ((self.num_memory_tokens) // delta_memory.shape[1]) + 
                        [delta_memory[:, -((self.num_memory_tokens) % delta_memory.shape[1]):]], dim=1
                    )
            else:
                delta_memory = delta_memory[:, -self.num_memory_tokens:]

            # initialize STM
            self.memory.data = delta_memory

            if self.use_ltm:

                self.memory_ages = torch.zeros(self.memory.shape[1], dtype=torch.int)

                # initialize LTM
                self.fill_in_ltm(delta_memory)
                self.ltm_ages = [torch.tensor(x) for x in self.ltm_ages]

            self.initialized += 1

        else:

            # if self.memory.data.shape[1] == delta_memory.shape[1]:

            #     ages_to_add = len(self.final_list)
            #     self.memory.data = delta_memory
            #     remaining_indices = None
            #     if cached_contexts_indicators is not None:
            #         cached_contexts_indicators = torch.zeros_like(cached_contexts_indicators)

            # else:

            remaining_indices = torch.arange(self.memory.shape[1])

            ages_to_add = 0
            # TODO: this while loop can be pre-computed and optimized
            # TODO: there should be a mapping from `delta_memory.shape[1]` to `ages_to_add` and `remaining_indices`
            while len(remaining_indices) + delta_memory.shape[1] > self.num_memory_tokens:
                ages_to_add += 1
                remaining_indices = remaining_indices[self.remaining_indicators[:len(remaining_indices)]]

            assert len(remaining_indices) + delta_memory.shape[1] == self.num_memory_tokens

            if self.use_ltm:

                self.memory_ages += ages_to_add
                self.ltm_ages = [x + ages_to_add for x in self.ltm_ages]

                dropped_indices = np.setdiff1d(np.arange(self.num_memory_tokens), remaining_indices.cpu().numpy())
                dropped_indices = dropped_indices[torch.where(self.importance_indicators[dropped_indices])[0]]
                dropped_ages = self.memory_ages[dropped_indices]
                dropped_memory = self.memory.data[:, dropped_indices]

            self.memory.data = torch.cat([
                self.memory.data[:, remaining_indices],
                delta_memory
            ], dim=1)

            if cached_contexts_indicators is not None:
                if len(cached_contexts_indicators.shape) == 3:
                    # cached_contexts_indicators: [1, L, num_memory_tokens]
                    cached_contexts_indicators = torch.cat([
                        cached_contexts_indicators[:, :, remaining_indices],
                        torch.zeros([cached_contexts_indicators.shape[0], cached_contexts_indicators.shape[1], delta_memory.shape[1]]).to(cached_contexts_indicators.device)
                    ], dim=2)
                else:
                    # cached_contexts_indicators: [L, num_memory_tokens]
                    cached_contexts_indicators = torch.cat([
                        cached_contexts_indicators[:, remaining_indices],
                        torch.zeros([cached_contexts_indicators.shape[0], delta_memory.shape[1]]).to(cached_contexts_indicators.device)
                    ], dim=1)
            
            if self.use_ltm:

                new_ages = []
                for num_idx, num in enumerate(self.final_list[-ages_to_add:]):
                    new_ages.extend([ages_to_add - num_idx - 1] * num)

                self.memory_ages = torch.cat(
                    [self.memory_ages[remaining_indices],
                    torch.tensor(new_ages)]
                )

                # update LTM
                with torch.no_grad():
                    
                    for idx in range(self.L):

                        # use delta_memory to recall LTM
                        # delta_memory: [32, 256, 4096]
                        if len(self.ltm_keys[idx]) > (self.num_ltm_tokens * ages_to_add):
                            queries = self.model.layers[idx].self_attn.encoder_query_proj(self.model.layers[idx].input_layernorm(delta_memory[idx]))
                            predictions = (queries @ self.ltm_keys[idx].to(queries.device).transpose(-2, -1)).sigmoid().mean(dim=0)
                            indices = torch.topk(predictions, k=(self.num_ltm_tokens * ages_to_add))[1].cpu()
                            self.ltm_keys[idx][indices] += 1

                        self.ltm[idx] = torch.cat([
                            self.ltm[idx],
                            dropped_memory[idx].detach().cpu()
                        ])
                        self.ltm_ages[idx] = torch.cat([
                            self.ltm_ages[idx],
                            dropped_ages.detach().cpu()
                        ])
                        self.ltm_keys[idx] = torch.cat([
                            self.ltm_keys[idx],
                            self.model.layers[idx].self_attn.key_proj(
                                self.model.layers[idx].input_layernorm(
                                    dropped_memory[idx]
                                )
                            ).detach().cpu()
                        ])
                        self.ltm_recall_frequencies[idx] = torch.cat([
                            self.ltm_recall_frequencies[idx],
                            self.initial_rf_when_moving_stm_to_ltm * torch.ones(len(dropped_ages))
                        ])

                        self.ltm_recall_frequencies[idx] -= self.decay_frequency
                        indices = np.where(self.ltm_recall_frequencies[idx] > 0.01)[0] # sometimes it may be "2.0539126e-15", using 0.01 to filter out these cases
                        if len(indices) > self.num_ltm_blocks * self.num_tokens:
                            self.ltm[idx] = self.ltm[idx][indices]
                            self.ltm_keys[idx] = self.ltm_keys[idx][indices]
                            self.ltm_recall_frequencies[idx] = self.ltm_recall_frequencies[idx][indices]
                            self.ltm_ages[idx] = self.ltm_ages[idx][indices]

                        else:
                            self.ltm_recall_frequencies[idx] += self.decay_frequency

            if cached_contexts_indicators is not None:
                return cached_contexts_indicators

    def cat_memory_and_hiddens(self, idx, hidden_states, delta_memory=None, 
                               is_injection=False,
                               cat_to_maximum_memory=False,
                               random_retriever_length=False,
                               delta_memory_position_idx=None,
                               training=False):
        
        if not self.initialized:
            return hidden_states, None
    
        if delta_memory is None or len(delta_memory) == 0:
            if is_injection:
                cur_memory = self.memory[idx][ - self.num_tokens:].unsqueeze(0)
            else:
                cur_memory = self.memory[idx].unsqueeze(0)

        else:

            cur_memory = delta_memory[:, idx]
            
            if (not is_injection) and cat_to_maximum_memory:
                if cur_memory.shape[1] == self.memory.shape[1]:
                    pass
                else:

                    if delta_memory_position_idx is not None:
                        
                        start_memory_position_idx = self.final_list[:delta_memory_position_idx].sum() if delta_memory_position_idx > 0 else 0
                        old_memory = self.memory[idx].detach()
                        # old_memory[:, start_memory_position_idx:start_memory_position_idx+cur_memory.shape[1]] = cur_memory
                        # cur_memory = old_memory.unsqueeze(0)
                        cur_memory = torch.cat([
                            old_memory[:, :start_memory_position_idx].unsqueeze(0),
                            cur_memory,
                            old_memory[:, start_memory_position_idx+cur_memory.shape[1]:].unsqueeze(0)
                        ], dim=1)

                    else:

                        remaining_indices = torch.arange(self.memory.shape[1])

                        while len(remaining_indices) + cur_memory.shape[1] > self.num_memory_tokens:
                            # tmp_memory = tmp_memory[:, self.remaining_indicators[:tmp_memory.shape[1]]]
                            remaining_indices = remaining_indices[self.remaining_indicators[:len(remaining_indices)]]

                        cur_memory = torch.cat([
                            self.memory[idx].detach()[remaining_indices].unsqueeze(0),
                            cur_memory
                        ], dim=1)

        ltm_indices = None
        if self.use_ltm and (not is_injection) and not (delta_memory is not None and cat_to_maximum_memory == False):
            ltm, ltm_indices = self.get_ltm(idx, hidden_states, random_retriever_length=random_retriever_length)
            cur_memory = torch.cat([ltm.unsqueeze(0), cur_memory], dim=1)

        if self.add_bos_embedding:
            if self.bos_embedding[idx].device != cur_memory.device:
                cur_memory = torch.cat([self.bos_embedding[idx].unsqueeze(0).repeat(len(cur_memory), 1, 1).to(cur_memory.device), cur_memory], dim=1)
            else:
                cur_memory = torch.cat([self.bos_embedding[idx].unsqueeze(0).repeat(len(cur_memory), 1, 1), cur_memory], dim=1)

        return torch.cat([cur_memory, hidden_states], dim=1), ltm_indices

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        delta_memory: Optional[List[List[torch.FloatTensor]]] = None,
        labels: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_delta_memory: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_injection: Optional[bool] = None,
        cat_to_maximum_memory: Optional[bool] = False,
        output_retriever_weights: Optional[bool] = False,
        return_full_retriever_weights: Optional[bool] = False,
        random_retriever_length: Optional[bool] = False,
        training: Optional[bool] = False,
        # return_memory_keys: Optional[bool] = False,
        encoder_query_indices: Optional[List[int]] = None,
        memory_key_indicators: Optional[torch.BoolTensor] = None,
        delta_memory_position_idx: Optional[torch.LongTensor] = None,
        parallel_injection: Optional[bool] = False,
        debug: Optional[bool] = False
    ) -> Union[Tuple, MemoryLMOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_injection is None:
            is_injection = output_delta_memory

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.model.gradient_checkpointing and self.model.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        
        if inputs_embeds is None:
            
            # assert self.model.embed_tokens.weight.requires_grad == False
            inputs_embeds = self.model.embed_tokens(input_ids)

            if self.tune_special_tokens:

                inputs_embeds = inputs_embeds.detach()
                new_inputs_embeds = []
                for idx, token_id in enumerate(input_ids[0]):
                    
                    if token_id in self.special_token_ids:
                        new_inputs_embeds.append(self.special_token_embeddings[self.special_token_ids.index(token_id.item())])
                    else:
                        new_inputs_embeds.append(inputs_embeds[0, idx])
                    
                inputs_embeds = torch.stack(new_inputs_embeds).unsqueeze(0)

            if self.add_memory_embedding and is_injection:

                if self.memory_embeddings.shape[0] == 1:
                    inputs_embeds = torch.cat([
                        inputs_embeds,
                        self.memory_embeddings.unsqueeze(0).repeat(1, self.num_tokens, 1)
                    ], dim=1)
                else:
                    inputs_embeds = torch.cat([
                        inputs_embeds,
                        self.memory_embeddings[self.memory_position_ids].unsqueeze(0)
                    ], dim=1)
                
                # if inputs_embeds.shape[1] < self.num_tokens:
                #     inputs_embeds = torch.cat([
                #         inputs_embeds,
                #         self.memory_embeddings.unsqueeze(0).repeat(1, self.num_tokens - inputs_embeds.shape[1], 1)
                #     ], dim=1)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        # if cache_position is None:
        # TODO: currently ignore cache_position
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        
        if self.initialized:
            if past_seen_tokens > 0:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
                if self._detach_memory:
                    cache_position += self.num_memory_tokens
            else:
                if is_injection:
                    if self.add_memory_embedding:
                        cache_position = torch.arange(
                            0, inputs_embeds.shape[1] + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
                    else:
                        cache_position = torch.arange(
                            0, inputs_embeds.shape[1] + self.num_tokens + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
                elif delta_memory is not None and delta_memory.shape[2] < self.num_memory_tokens and not cat_to_maximum_memory:
                    
                    if self.fix_poe_for_encoder:

                        memory_pos = torch.arange(1, 1 + self.num_memory_tokens)

                        if self.important_tokens == "right":
                            memory_pos = memory_pos[-delta_memory.shape[2]:]
                        else:
                            memory_pos = memory_pos[:delta_memory.shape[2]]
                        
                        cache_position = torch.cat([
                            torch.arange(0, int(self.add_bos_embedding)),
                            memory_pos,
                            torch.arange(int(self.add_bos_embedding) + self.num_memory_tokens, int(self.add_bos_embedding) + self.num_memory_tokens + inputs_embeds.shape[1])
                        ]).to(inputs_embeds.device)

                    else:
                        cache_position = torch.arange(
                            0, inputs_embeds.shape[1] + delta_memory.shape[2] + int(self.add_bos_embedding), device=inputs_embeds.device
                        )

                else:
                    if self._detach_memory:
                        cache_position = torch.arange(
                            self.num_memory_tokens + int(self.add_bos_embedding), 
                            inputs_embeds.shape[1] + self.num_memory_tokens + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
                        cache_position = torch.cat([
                            torch.tensor([0], device=inputs_embeds.device), cache_position
                        ])

                    else:
                        if self.use_ltm:
                            cache_position = torch.arange(
                                0, inputs_embeds.shape[1] + self.num_memory_tokens + self.num_ltm_tokens + int(self.add_bos_embedding), device=inputs_embeds.device
                            )
                        else:
                            cache_position = torch.arange(
                                0, inputs_embeds.shape[1] + self.num_memory_tokens + int(self.add_bos_embedding), device=inputs_embeds.device
                            )
        
        else:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)
        # TODO: currently ignore position_ids
            
        position_ids = cache_position.unsqueeze(0)

        # TODO: check why sometimes there are zeros in attention_mask
        # causal_mask = self.model._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )
        causal_mask = None

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_delta_memory = [] if output_delta_memory else None
        all_retriever_weights = () if output_retriever_weights else None
        all_encoder_retriever_weights = () if (output_retriever_weights and encoder_query_indices is not None) else None
        all_ltm_indices = ()

        if self.add_decoder_lora:

            if is_injection or (delta_memory is not None and delta_memory.shape[2] == self.num_tokens and not cat_to_maximum_memory):
                for name, module in self.named_modules():
                    if hasattr(module, "_active_adapter"):
                        module._active_adapter = ['default',]
            else:
                for _, module in self.named_modules():
                    if hasattr(module, "_active_adapter"):
                        module._active_adapter = ['decoder_adapter']

        for idx, decoder_layer in enumerate(self.model.layers):
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if past_key_values is None or past_key_values.get_seq_length(layer_idx=idx) == 0:

                if is_injection or (not self._detach_memory):
                    
                    if self.add_memory_embedding and is_injection and self.initialized:
                        hidden_states = torch.cat([
                            self.bos_embedding[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1),
                            hidden_states,
                        ], dim=1)
                    else:
                        hidden_states, ltm_indices = self.cat_memory_and_hiddens(idx,
                                                    hidden_states=hidden_states,
                                                    delta_memory=delta_memory,
                                                    is_injection=is_injection,
                                                    cat_to_maximum_memory=cat_to_maximum_memory,
                                                    random_retriever_length=random_retriever_length,
                                                    delta_memory_position_idx=delta_memory_position_idx,
                                                    training=training)
                        
                        if ltm_indices is not None:
                            all_ltm_indices += (ltm_indices,)

                else:
                    hidden_states = torch.cat([
                        self.bos_embedding[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1),
                        hidden_states
                    ], dim=1)
                
                prefix_token_length = hidden_states.shape[1] - inputs_embeds.shape[1] if self.initialized else 0

                if is_injection and prefix_token_length > 0 and not self.add_memory_embedding:
                    prefix_token_length = min(prefix_token_length, hidden_states.shape[1] - self.num_tokens)

                if is_injection and not self.add_memory_embedding:
                    if self.new_memory_positional_emb.device != hidden_states.device:
                        hidden_states[:, -self.num_tokens:] += self.new_memory_positional_emb.to(hidden_states.device)
                    else:
                        hidden_states = torch.cat([
                            hidden_states[:, :-self.num_tokens],
                            hidden_states[:, -self.num_tokens:] + self.new_memory_positional_emb
                        ], dim=1)

            else:
                prefix_token_length = 0

            # if self.model.gradient_checkpointing and self.model.training:
            if self.model.gradient_checkpointing and training:

                layer_outputs = self.model._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    prefix_token_length,
                    output_retriever_weights,
                )

            else:

                assert not (is_injection and encoder_query_indices)
                
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    prefix_token_length=prefix_token_length,
                    output_retriever_weights=output_retriever_weights,
                    return_full_retriever_weights=return_full_retriever_weights,
                    random_retriever_length=random_retriever_length,
                    encoder_query_indices=encoder_query_indices[idx] if encoder_query_indices is not None else None,
                    encoder_attention_mask=encoder_attention_mask,
                    ltm_length=self.num_ltm_tokens if self.use_ltm else 0,
                    training=training,
                    memory_key_indicators=memory_key_indicators,
                )

            hidden_states = layer_outputs[0]
            if output_delta_memory:
                all_delta_memory.append(hidden_states[:, -self.num_tokens:])
            hidden_states = hidden_states[:, -inputs_embeds.shape[1]:]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if output_retriever_weights:
                if encoder_query_indices is not None:
                    retriever_weights = layer_outputs[-2]
                    encoder_retriever_weights = layer_outputs[-1]
                    all_encoder_retriever_weights += (encoder_retriever_weights,)
                else:
                    retriever_weights = layer_outputs[-1]
                if retriever_weights is not None:
                    all_retriever_weights += (retriever_weights,)

        hidden_states = self.model.norm(hidden_states)
            
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if output_delta_memory:

            if all_delta_memory[0].device != all_delta_memory[-1].device:
                assert not self.training
                device = all_delta_memory[0].device
                all_delta_memory = [x.to(device) for x in all_delta_memory]
                delta_memory = torch.stack(all_delta_memory, dim=0).transpose(0, 1)

            else:
                delta_memory = torch.stack(all_delta_memory, dim=0).transpose(0, 1)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            return tuple(v for v in [loss, logits, hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return MemoryLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            delta_memory=delta_memory,
            retriever_weights=all_retriever_weights if (all_retriever_weights is not None and len(all_retriever_weights) > 0) else None,
            encoder_retriever_weights=all_encoder_retriever_weights if (all_encoder_retriever_weights is not None and len(all_encoder_retriever_weights) > 0) else None,
            ltm_indices=all_ltm_indices if (len(all_ltm_indices) > 0 and all_ltm_indices[0] is not None) else None
            # memory_keys=all_memory_keys if (all_memory_keys is not None and len(all_memory_keys) > 0) else None
        )

        
class LlamaDropMemoryLTMModel(LlamaForCausalLM, BaseMemoryModel):
    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)
        BaseMemoryModel.__init__(self, config)
        
        self.config = config
        self.L = config.num_hidden_layers
        self.d = config.hidden_size
        self.num_tokens = config.num_tokens
        self.bos_token_id = config.bos_token_id

        self.add_bos_embedding = config.add_bos_embedding
        self.shrink_to_one_embedding = config.shrink_to_one_embedding
        self.drop_memory_per_layer = config.drop_memory_per_layer if hasattr(config, "drop_memory_per_layer") else False
        self.add_decoder_lora = config.add_decoder_lora
        self.tune_special_tokens = config.tune_special_tokens
        self.special_token_ids = config.special_token_ids
        self.maintain_memory_keys = config.maintain_memory_keys if hasattr(config, "maintain_memory_keys") else False
        self.virtual_num_blocks = config.virtual_num_blocks if hasattr(config, "virtual_num_blocks") else None

        # long-term memory configurations
        self.num_ltm_blocks = config.num_ltm_blocks if hasattr(config, "num_ltm_blocks") else 10
        self.num_blocks = config.num_blocks - self.num_ltm_blocks
        assert self.num_blocks > 0
        self.converge_ltm_number_tokens = config.converge_ltm_number_tokens if hasattr(config, "converge_ltm_number_tokens") else 178000
        self.decay_frequency = config.decay_frequency if hasattr(config, "decay_frequency") else 1

        self.update_ltm_mode = config.update_ltm_mode if hasattr(config, 'update_ltm_mode') else 'decouple'
        self.update_ltm_frequency = config.update_ltm_frequency if hasattr(config, "update_ltm_frequency") else config.num_blocks
        self.update_ltm_T0 = config.update_ltm_from if hasattr(config, "update_ltm_from") else 0
        self.update_ltm_num_tokens = config.update_ltm_num_tokens if hasattr(config, "update_ltm_num_tokens") else config.num_tokens
        self.update_step = 0

        '''
            num_ltm_blocks: number of long-term memory blocks, if it is 10, then 2560 tokens will be retrieved from long-term memory
            update_ltm_frequency: if it is set as 50, then we update long-term memory every 50 steps. 
            update_ltm_num_tokens: if it is set as 100, then 100 tokens will be recalled each time in the short-term memory.
            update_step: the current step of updating, when it is less than update_ltm_frequency, we don't update long-term memory. It will be reset after updating long-term memory.
        '''

        # setup short-term memory and long-term memory parameters
        self.ltm = [None] * self.L
        self.ltm_keys = [None] * self.L
        self.ltm_ages = [None] * self.L
        self.ltm_recall_frequencies = [None] * self.L
        self.initial_rf_when_moving_stm_to_ltm = config.initial_rf_when_moving_stm_to_ltm if hasattr(config, "initial_rf_when_moving_stm_to_ltm") else None
        self.put_cached_dropped_memory_on_cpu = config.put_cached_dropped_memory_on_cpu if hasattr(config, "put_cached_dropped_memory_on_cpu") else False

        # self.ltm_keys = [[None] * config.num_key_value_heads] * self.L
        # self.ltm_values = [[None] * config.num_key_value_heads] * self.L
        # self.ltm_recall_frequencies = [[None] * config.num_key_value_heads] * self.L

        self.memory = nn.Parameter(torch.randn([self.L, self.num_blocks * self.num_tokens, self.d]))
        print(f"Memory Pool Parameters: {len(self.memory.reshape(-1)) / 1_000_000_000:.4f} B")
        self.memory_keys = None
        self.memory_ages = [None] * self.L
        self.memory_recall_frequency = [np.zeros([self.memory.shape[1]]) for _ in range(self.L)]
        self.memory_position_indicators = [np.zeros(self.memory.shape[1]) for _ in range(self.L)]
        self.maintain_memory_keys = config.maintain_memory_keys if hasattr(config, "maintain_memory_keys") else False

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.memory.requires_grad = False
        self.add_positional_embedding = False

        self.new_memory_positional_emb = nn.Parameter(torch.zeros([1, 1, self.d]))

        if config.add_bos_embedding:
            self.bos_embedding = nn.Parameter(torch.randn([self.L, 1, self.d]))

        self.add_selector = True if hasattr(config, "add_selector") and config.add_selector else False
        assert self.add_selector
        if self.tune_special_tokens:
            self.special_token_embeddings = nn.Parameter(torch.zeros([len(self.special_token_ids), self.d]))

        self._detach_memory = False
        self.cached_dropped_memories, self.cached_dropped_memory_ages = None, None
        self.cached_dropped_keys = None

        self.skip_logits_except_the_last_hidden_state = False

    def set_exclude_layers(self, layer_indices):
        self.exclude_layers = layer_indices

    def super_forward(self, *args, **kwargs):
        return LlamaForCausalLM.forward(self, *args, **kwargs)

    def detach_memory(self):
        self._detach_memory = True
    
    def attach_memory(self):
        self._detach_memory = False

    def fill_in_ltm(self, delta_memory):

        if len(delta_memory.shape) == 4:
            delta_memory = delta_memory.detach()[0]

        with torch.no_grad():
            for idx in range(len(self.memory)):
                # current_memory = self.memory.data[idx]
                # self.ltm[idx] = current_memory[-self.num_tokens * self.num_ltm_blocks:].detach().cpu()
                self.ltm[idx] = delta_memory[idx].detach().cpu()
                self.ltm_recall_frequencies[idx] = np.zeros(self.ltm[idx].shape[0])
                self.ltm_keys[idx] = self.model.layers[idx].self_attn.key_proj(self.model.layers[idx].input_layernorm(delta_memory[idx].detach())).cpu()
                self.ltm_ages[idx] = np.zeros(self.ltm[idx].shape[0])

    def update_memory_with_delta_memory_and_keys(self,
                                                delta_memory, 
                                                delta_memory_ages,
                                                dropped_delta_memory,
                                                dropped_delta_memory_keys,
                                                dropped_delta_memory_ages):

        if len(dropped_delta_memory.shape) == 4:
            dropped_delta_memory = dropped_delta_memory[0]
        if len(dropped_delta_memory_keys.shape) == 4:
            dropped_delta_memory_keys = dropped_delta_memory_keys[0]

        max_delta_memory_age = max(delta_memory_ages.max().item(), dropped_delta_memory_ages.max().item())
        # call the update_memory_with_delta_memory function from the base class
        outputs = BaseMemoryModel.update_memory_with_delta_memory(self, 
                                    delta_memory.to(self.memory.device) if delta_memory.device != self.memory.device else delta_memory, 
                                    is_ltm=True, 
                                    cached_contexts_indicators=None, 
                                    retriever_weights=None,
                                    delta_memory_ages=delta_memory_ages,
                                    return_dropped_memories=True)

        ages_to_add = 1 + max_delta_memory_age

        # update ages
        for idx in range(self.L):
            self.ltm_ages[idx] += ages_to_add
            if self.cached_dropped_memory_ages is not None:
                self.cached_dropped_memory_ages[idx] += ages_to_add

        (dropped_memories, dropped_memory_ages) = outputs

        with torch.no_grad():
            dropped_memory_keys = [
                self.model.layers[idx].self_attn.key_proj(
                    self.model.layers[idx].input_layernorm(
                        dropped_memories[idx].cuda() if dropped_memories[idx].device == torch.device('cpu') else dropped_memories[idx]
                    )
                ).detach().cpu()
                for idx in range(self.L)
            ]
            
        dropped_memories = torch.stack(dropped_memories).detach().cpu()
        dropped_memory_keys = torch.stack(dropped_memory_keys).detach().cpu()

        dropped_memories = torch.cat([dropped_memories, dropped_delta_memory], dim=1)
        dropped_memory_keys = torch.cat([dropped_memory_keys, dropped_delta_memory_keys], dim=1)
        dropped_memory_ages = np.concatenate([np.stack(dropped_memory_ages) + ages_to_add, dropped_delta_memory_ages], axis=1)

        for idx in range(self.L):
            self.ltm_keys[idx] = torch.cat([self.ltm_keys[idx], dropped_memory_keys[idx]])
            self.ltm[idx] = torch.cat([self.ltm[idx], dropped_memories[idx]])
            self.ltm_ages[idx] = np.concatenate([self.ltm_ages[idx], dropped_memory_ages[idx]])

            if self.ltm[idx].shape[0] > 153600:
                self.ltm_keys[idx] = self.ltm_keys[idx][-153600:]
                self.ltm[idx] = self.ltm[idx][-153600:]
                self.ltm_ages[idx] = self.ltm_ages[idx][-153600:]

    def update_memory_with_delta_memory(self, 
                                        delta_memory, 
                                        cached_contexts_indicators=None, 
                                        retriever_weights=None, 
                                        delta_memory_ages=None,
                                        dropped_delta_memory=None,
                                        dropped_delta_memory_ages=None):
        
        initialized = self.initialized.item()

        if delta_memory_ages is not None and dropped_delta_memory_ages is not None:
            if dropped_delta_memory_ages.shape[1] > 0:
                max_delta_memory_age = max(delta_memory_ages.max().item(), dropped_delta_memory_ages.max().item())
            else:
                max_delta_memory_age = delta_memory_ages.max().item()
        else:
            max_delta_memory_age = None

        # call the update_memory_with_delta_memory function from the base class
        outputs = BaseMemoryModel.update_memory_with_delta_memory(self, 
                                    delta_memory, 
                                    cached_contexts_indicators, 
                                    is_ltm=True, 
                                    retriever_weights=retriever_weights,
                                    delta_memory_ages=delta_memory_ages,
                                    return_dropped_memories=(self.update_ltm_mode == 'immediate' and initialized))

        if initialized:
            
            if self.update_ltm_mode == 'decouple':
                for idx in range(self.L):
                    self.ltm_ages[idx] += 1
                cached_contexts_indicators = outputs
                self.update_step += 1
                if self.update_step >= self.update_ltm_frequency:
                    self.update_ltm()
                    self.update_step = 0
            else:
                ages_to_add = self.num_tokens if max_delta_memory_age is None else 1 + max_delta_memory_age

                for idx in range(self.L):
                    self.ltm_ages[idx] += ages_to_add
                    if self.cached_dropped_memory_ages is not None:
                        self.cached_dropped_memory_ages[idx] += ages_to_add

                # update long-term memory each time 
                if cached_contexts_indicators is not None:
                    cached_contexts_indicators, (dropped_memories, dropped_memory_ages) = outputs
                else:
                    (dropped_memories, dropped_memory_ages) = outputs

                # cat dropped_memories, dropped_delta_memory
                # cat dropped_memory_ages, drop_delta_memory_ages
                if dropped_delta_memory is not None:
                    dropped_memories = torch.cat([torch.stack(dropped_memories), dropped_delta_memory[0]], dim=1)
                    dropped_memory_ages = np.concatenate([np.stack(dropped_memory_ages) + ages_to_add, dropped_delta_memory_ages], axis=1)
                
                # self.update_ltm(dropped_memories, dropped_memory_ages)

                # Accumulate the dropped_memories and dropped_memory_ages and update for once

                if isinstance(dropped_memories, list):
                    dropped_memories = torch.stack(dropped_memories)

                if self.update_step == 0:

                    with torch.no_grad():
                        cached_dropped_keys = []
                        for idx in range(self.L):
                            cached_dropped_keys.append(
                                self.model.layers[idx].self_attn.key_proj(
                                    self.model.layers[idx].input_layernorm(
                                        dropped_memories[idx]
                                    )
                                ).detach().cpu()
                            )
                        self.cached_dropped_keys = torch.stack(cached_dropped_keys)

                    if self.put_cached_dropped_memory_on_cpu:
                        self.cached_dropped_memories = dropped_memories.detach().cpu()
                    else:
                        self.cached_dropped_memories = dropped_memories.detach()

                    self.cached_dropped_memory_ages = dropped_memory_ages

                else:
                    
                    with torch.no_grad():
                        cached_dropped_keys = []
                        for idx in range(self.L):
                            cached_dropped_keys.append(
                                self.model.layers[idx].self_attn.key_proj(
                                    self.model.layers[idx].input_layernorm(
                                        dropped_memories[idx]
                                    )
                                ).detach().cpu()
                            )
                        cached_dropped_keys = torch.stack(cached_dropped_keys)
                        self.cached_dropped_keys = torch.cat([
                            self.cached_dropped_keys,
                            cached_dropped_keys
                        ], dim=1)

                    # empty torch memory cache
                    torch.cuda.empty_cache()

                    if self.put_cached_dropped_memory_on_cpu:
                        self.cached_dropped_memories = torch.cat([
                            self.cached_dropped_memories, 
                            dropped_memories.detach().cpu()
                        ], dim=1)
                    else:
                        self.cached_dropped_memories = torch.cat([
                            self.cached_dropped_memories, 
                            dropped_memories.detach()
                        ], dim=1)
                    
                    self.cached_dropped_memory_ages = np.concatenate([
                        self.cached_dropped_memory_ages,
                        dropped_memory_ages
                    ], axis=1)

                self.update_step += ages_to_add

                if self.update_step >= self.update_ltm_frequency * self.num_tokens:
                    self.update_ltm(self.cached_dropped_memories, 
                                    self.cached_dropped_memory_ages,
                                    device=delta_memory.device if self.put_cached_dropped_memory_on_cpu else None,
                                    cached_dropped_keys=self.cached_dropped_keys)
                    self.update_step = 0
                    self.cached_dropped_memories, self.cached_dropped_memory_ages = None, None

        else:
            cached_contexts_indicators = outputs
            self.fill_in_ltm(delta_memory)

        return cached_contexts_indicators
        
    def cat_memory_and_hiddens(self, idx, hidden_states, delta_memory=None, 
                               is_injection=False,
                               cat_to_maximum_memory=False,
                               random_retriever_length=False):
        
        if not self.initialized:
            return hidden_states, None
    
        stm = self.get_stm(idx, hidden_states, delta_memory, is_injection, cat_to_maximum_memory)

        if stm.device != hidden_states.device:
            stm = stm.to(hidden_states.device)

        ltm_indices = None

        if (not is_injection) and (delta_memory is None or cat_to_maximum_memory):
            
            ltm, ltm_indices = self.get_ltm(idx, hidden_states, random_retriever_length=random_retriever_length)
            hidden_states = torch.cat([
                ltm.unsqueeze(0),
                stm,
                hidden_states
            ], dim=1)

        else:
            hidden_states = torch.cat([stm, hidden_states], dim=1)

        if self.add_bos_embedding:
            hidden_states = torch.cat([self.bos_embedding[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1), hidden_states], dim=1)
        
        return hidden_states, ltm_indices
    
    def use_decoder_lora(self):
        for _, module in self.named_modules():
            if hasattr(module, "_active_adapter"):
                module._active_adapter = ['decoder_adapter']
    
    def use_encoder_lora(self):
        for _, module in self.named_modules():
            if hasattr(module, "_active_adapter"):
                module._active_adapter = ['default']        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        delta_memory: Optional[List[List[torch.FloatTensor]]] = None,
        labels: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_delta_memory: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_injection: Optional[bool] = None,
        cat_to_maximum_memory: Optional[bool] = False,
        output_retriever_weights: Optional[bool] = False,
        return_full_retriever_weights: Optional[bool] = False,
        random_retriever_length: Optional[bool] = False,
        encoder_query_indices: Optional[List[int]] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, MemoryLMOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_injection is None:
            is_injection = output_delta_memory

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.model.gradient_checkpointing and self.model.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        
        if inputs_embeds is None:

            inputs_embeds = self.model.embed_tokens(input_ids)

            if self.tune_special_tokens:
                for idx, token_id in enumerate(input_ids[0]):
                    if token_id in self.special_token_ids:
                        inputs_embeds[:, idx] = self.special_token_embeddings[self.special_token_ids.index(token_id.item())]

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        # if cache_position is None:
        # TODO: currently ignore cache_position
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        
        if self.initialized:
            if past_seen_tokens > 0:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
                if self._detach_memory:
                    cache_position += self.num_tokens * (self.num_blocks + self.num_ltm_blocks)
            else:
                if is_injection:
                    cache_position = torch.arange(
                        0, inputs_embeds.shape[1] + self.num_tokens + int(self.add_bos_embedding), device=inputs_embeds.device
                    )
                elif delta_memory is not None and delta_memory.shape[2] == self.num_tokens and not cat_to_maximum_memory:
                    cache_position = torch.arange(
                        0, inputs_embeds.shape[1] + self.num_tokens + int(self.add_bos_embedding), device=inputs_embeds.device
                    )
                else:
                    if self._detach_memory:
                        cache_position = torch.arange(
                            self.num_tokens * (self.num_blocks + self.num_ltm_blocks) + int(self.add_bos_embedding), 
                            inputs_embeds.shape[1] + self.num_tokens * (self.num_blocks + self.num_ltm_blocks) + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
                        cache_position = torch.cat([
                            torch.tensor([0], device=inputs_embeds.device), cache_position
                        ])

                    else:
                        cache_position = torch.arange(
                            0, inputs_embeds.shape[1] + self.num_tokens * (self.num_blocks + self.num_ltm_blocks) + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
        
        else:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)
        # TODO: currently ignore position_ids
            
        position_ids = cache_position.unsqueeze(0)

        # TODO: check why sometimes there are zeros in attention_mask
        # causal_mask = self.model._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )
        causal_mask = None

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_delta_memory = [] if output_delta_memory else None
        all_retriever_weights = () if output_retriever_weights else None
        all_encoder_retriever_weights = () if (output_retriever_weights and encoder_query_indices is not None) else None
        all_ltm_indices = ()

        if self.add_decoder_lora:

            if is_injection or (delta_memory is not None and delta_memory.shape[2] == self.num_tokens and not cat_to_maximum_memory):
                self.use_encoder_lora()
            else:
                self.use_decoder_lora()

        ltm_indices = None

        for idx, decoder_layer in enumerate(self.model.layers):
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if past_key_values is None or past_key_values.get_seq_length(layer_idx=idx) == 0:

                if is_injection or (not self._detach_memory):

                    hidden_states, ltm_indices = self.cat_memory_and_hiddens(idx,
                                                    hidden_states=hidden_states,
                                                    delta_memory=delta_memory,
                                                    is_injection=is_injection,
                                                    cat_to_maximum_memory=cat_to_maximum_memory,
                                                    random_retriever_length=random_retriever_length)

                    all_ltm_indices += (ltm_indices,)

                else:
                    hidden_states = torch.cat([
                        self.bos_embedding[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1),
                        hidden_states
                    ], dim=1)
                
                prefix_token_length = hidden_states.shape[1] - inputs_embeds.shape[1] if self.initialized else 0

                if is_injection and prefix_token_length > 0:
                    prefix_token_length = min(prefix_token_length, hidden_states.shape[1] - self.num_tokens)

                if is_injection:
                    if self.new_memory_positional_emb.device != hidden_states.device:
                        hidden_states[:, -self.num_tokens:] += self.new_memory_positional_emb.to(hidden_states.device)
                    else:
                        hidden_states[:, -self.num_tokens:] += self.new_memory_positional_emb

            else:
                prefix_token_length = 0

            if self.model.gradient_checkpointing and self.model.training:

                layer_outputs = self.model._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    prefix_token_length,
                    output_retriever_weights,
                )
                
            else:
                
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    prefix_token_length=prefix_token_length,
                    output_retriever_weights=output_retriever_weights,
                    return_full_retriever_weights=return_full_retriever_weights,
                    random_retriever_length=random_retriever_length,
                    encoder_query_indices=encoder_query_indices[idx] if encoder_query_indices is not None else None,
                    ltm_length=self.num_ltm_blocks * self.num_tokens,
                    training=training,
                )

            hidden_states = layer_outputs[0]
            if output_delta_memory:
                all_delta_memory.append(hidden_states[:, -self.num_tokens:])

                # if self.initialized:
                #     # recall part of the memory according to the hidden states
                #     self.update_recall_frequency(idx, hidden_states)

            hidden_states = hidden_states[:, -input_ids.shape[1]:]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_retriever_weights:
                if encoder_query_indices is not None:
                    retriever_weights = layer_outputs[-2]
                    encoder_retriever_weights = layer_outputs[-1]
                    all_encoder_retriever_weights += (encoder_retriever_weights,)
                else:
                    retriever_weights = layer_outputs[-1]
                if retriever_weights is not None:
                    all_retriever_weights += (retriever_weights,)
            
        hidden_states = self.model.norm(hidden_states)
            
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if output_delta_memory:

            if all_delta_memory[0].device != all_delta_memory[-1].device:
                assert not self.training
                device = all_delta_memory[0].device
                all_delta_memory = [x.to(device) for x in all_delta_memory]
                delta_memory = torch.stack(all_delta_memory, dim=0).transpose(0, 1)

            else:
                delta_memory = torch.stack(all_delta_memory, dim=0).transpose(0, 1)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if self.skip_logits_except_the_last_hidden_state:
                logits = self.lm_head(hidden_states[:, -1:])
            else:
                logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            return tuple(v for v in [loss, logits, hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return MemoryLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            delta_memory=delta_memory,
            retriever_weights=all_retriever_weights if (all_retriever_weights is not None and len(all_retriever_weights) > 0) else None,
            encoder_retriever_weights=all_encoder_retriever_weights if (all_encoder_retriever_weights is not None and len(all_encoder_retriever_weights) > 0) else None,
            ltm_indices=all_ltm_indices if (len(all_ltm_indices) > 0 and all_ltm_indices[0] is not None) else None
        )


class LlamaDropMemorySLModel(LlamaForCausalLM, BaseMemoryModel):
    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)
        BaseMemoryModel.__init__(self, config)
        
        self.config = config
        self.L = config.num_hidden_layers
        self.d = config.hidden_size
        self.num_blocks = config.num_blocks
        self.num_tokens = config.num_tokens
        self.bos_token_id = config.bos_token_id

        self.add_bos_embedding = config.add_bos_embedding
        self.shrink_to_one_embedding = config.shrink_to_one_embedding
        self.drop_memory_per_layer = config.drop_memory_per_layer if hasattr(config, "drop_memory_per_layer") else False
        self.add_decoder_lora = config.add_decoder_lora
        
        self.memory = nn.Parameter(torch.randn([1, self.num_blocks * self.num_tokens, self.d]))
        print(f"Memory Pool Parameters: {len(self.memory.reshape(-1)) / 1_000_000_000:.4f} B")

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.memory.requires_grad = False
        self.add_positional_embedding = False

        if self.shrink_to_one_embedding:
            self.new_memory_positional_emb = nn.Parameter(torch.zeros([1, 1, self.d]))
        else:
            # only for llama2, randn, for others, should be zeros
            # for llama2, randn won't help training, but it will prevent the appearance of nan
            self.new_memory_positional_emb = nn.Parameter(torch.randn([1, self.num_tokens, self.d])) # this is lethal

        if config.add_bos_embedding:
            self.bos_embedding = nn.Parameter(torch.randn([self.L, 1, self.d]))
            self.bos_embedding.data[0, :] = self.model.embed_tokens.weight.data[self.bos_token_id]
            self.bos_embedding = nn.Parameter(self.bos_embedding)

        self._detach_memory = False

    def set_exclude_layers(self, layer_indices):
        self.exclude_layers = layer_indices

    def super_forward(self, *args, **kwargs):
        return LlamaForCausalLM.forward(self, *args, **kwargs)

    def detach_memory(self):
        self._detach_memory = True
    
    def attach_memory(self):
        self._detach_memory = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        delta_memory: Optional[List[List[torch.FloatTensor]]] = None,
        labels: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_delta_memory: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_injection: Optional[bool] = None,
        cat_to_maximum_memory: Optional[bool] = False,
    ) -> Union[Tuple, MemoryLMOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_injection is None:
            is_injection = output_delta_memory

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.model.gradient_checkpointing and self.model.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        # if cache_position is None:
        # TODO: currently ignore cache_position
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        
        if self.initialized:
            if past_seen_tokens > 0:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
                if self._detach_memory:
                    cache_position += self.num_tokens * self.num_blocks
            else:
                if is_injection:
                    cache_position = torch.arange(
                        0, inputs_embeds.shape[1] + self.num_tokens + int(self.add_bos_embedding), device=inputs_embeds.device
                    )
                elif delta_memory is not None and delta_memory.shape[2] == self.num_tokens and not cat_to_maximum_memory:
                    cache_position = torch.arange(
                        0, inputs_embeds.shape[1] + self.num_tokens + int(self.add_bos_embedding), device=inputs_embeds.device
                    )
                else:
                    if self._detach_memory:
                        cache_position = torch.arange(
                            self.num_tokens * self.num_blocks + int(self.add_bos_embedding), 
                            inputs_embeds.shape[1] + self.num_tokens * self.num_blocks + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
                        cache_position = torch.cat([
                            torch.tensor([0], device=inputs_embeds.device), cache_position
                        ])

                    else:
                        cache_position = torch.arange(
                            0, inputs_embeds.shape[1] + self.num_tokens * self.num_blocks + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
        
        else:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)
        # TODO: currently ignore position_ids
            
        position_ids = cache_position.unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones([input_ids.shape[0], input_ids.shape[1] + int(self.add_bos_embedding)], device=input_ids.device)
            
        causal_mask = self.model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_delta_memory = [] if output_delta_memory else None

        if self.add_decoder_lora:

            if is_injection or (delta_memory is not None and delta_memory.shape[2] == self.num_tokens and not cat_to_maximum_memory):
                for name, module in self.named_modules():
                    if hasattr(module, "_active_adapter"):
                        module._active_adapter = ['default',]
            else:
                for _, module in self.named_modules():
                    if hasattr(module, "_active_adapter"):
                        # module._active_adapter = ['default', 'decoder_adapter']
                        module._active_adapter = ['decoder_adapter']

        for idx, decoder_layer in enumerate(self.model.layers):
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if past_key_values is None or past_key_values.get_seq_length(layer_idx=idx) == 0:

                if is_injection or (not self._detach_memory):
                    hidden_states = self.cat_memory_and_hiddens(0,
                                                    hidden_states=hidden_states,
                                                    delta_memory=delta_memory,
                                                    is_injection=is_injection,
                                                    cat_to_maximum_memory=cat_to_maximum_memory)
                else:
                    hidden_states = torch.cat([
                        self.bos_embedding[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1),
                        hidden_states
                    ], dim=1)
                
                prefix_token_length = hidden_states.shape[1] - inputs_embeds.shape[1] if self.initialized else 0

                if is_injection and prefix_token_length > 0:
                    prefix_token_length = min(prefix_token_length, hidden_states.shape[1] - self.num_tokens)

                if is_injection:
                    if self.new_memory_positional_emb.device != hidden_states.device:
                        hidden_states[:, -self.num_tokens:] += self.new_memory_positional_emb.to(hidden_states.device)
                    else:
                        hidden_states[:, -self.num_tokens:] += self.new_memory_positional_emb

            else:
                prefix_token_length = 0

            if self.model.gradient_checkpointing and self.model.training:

                layer_outputs = self.model._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    prefix_token_length
                )
                
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    prefix_token_length=prefix_token_length,
                    # debug=debug
                )

            hidden_states = layer_outputs[0]
            if output_delta_memory and idx == self.L - 1:
                all_delta_memory.append(hidden_states[:, -self.num_tokens:])
            hidden_states = hidden_states[:, -input_ids.shape[1]:]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
        hidden_states = self.model.norm(hidden_states)
            
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if output_delta_memory:

            if all_delta_memory[0].device != all_delta_memory[-1].device:
                assert not self.training
                device = all_delta_memory[0].device
                all_delta_memory = [x.to(device) for x in all_delta_memory]
                delta_memory = torch.stack(all_delta_memory, dim=0).transpose(0, 1)

            else:
                delta_memory = torch.stack(all_delta_memory, dim=0).transpose(0, 1)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            return tuple(v for v in [loss, logits, hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return MemoryLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            delta_memory=delta_memory,
        )


class LlamaDropMemoryVaryLengthModel(LlamaForCausalLM, BaseMemoryModel):
    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)
        BaseMemoryModel.__init__(self, config)
        
        self.config = config
        self.L = config.num_hidden_layers
        self.d = config.hidden_size
        self.num_blocks = config.num_blocks
        self.num_tokens = config.num_tokens
        self.bos_token_id = config.bos_token_id

        self.compress_ratio = config.compress_ratio if hasattr(config, "compress_ratio") else (config.max_length / config.num_tokens)
        self.add_bos_embedding = config.add_bos_embedding
        self.shrink_to_one_embedding = config.shrink_to_one_embedding
        self.drop_memory_per_layer = config.drop_memory_per_layer if hasattr(config, "drop_memory_per_layer") else False
        self.add_decoder_lora = config.add_decoder_lora
        
        self.memory = nn.Parameter(torch.randn([self.L, self.num_blocks * self.num_tokens, self.d]))
        print(f"Memory Pool Parameters: {len(self.memory.reshape(-1)) / 1_000_000_000:.4f} B")

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.memory.requires_grad = False
        self.add_positional_embedding = False

        if self.shrink_to_one_embedding:
            self.new_memory_positional_emb = nn.Parameter(torch.zeros([1, 1, self.d]))
        else:
            # only for llama2, randn, for others, should be zeros
            # for llama2, randn won't help training, but it will prevent the appearance of nan
            self.new_memory_positional_emb = nn.Parameter(torch.randn([1, self.num_tokens, self.d])) # this is lethal

        if config.add_bos_embedding:
            self.bos_embedding = nn.Parameter(torch.randn([self.L, 1, self.d]))
            self.bos_embedding.data[0, :] = self.model.embed_tokens.weight.data[self.bos_token_id]
            self.bos_embedding = nn.Parameter(self.bos_embedding)

    def set_exclude_layers(self, layer_indices):
        self.exclude_layers = layer_indices

    def super_forward(self, *args, **kwargs):
        return LlamaForCausalLM.forward(self, *args, **kwargs)

    def cat_memory_and_hiddens(self, idx, hidden_states, delta_memory=None, 
                               is_injection=False,
                               cat_to_maximum_memory=False):
        
        if not self.initialized:
            return hidden_states
    
        if not is_injection:
            
            if delta_memory is None or len(delta_memory) == 0:

                cur_memory = self.memory[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1)
                
                # put on cuda
                if cur_memory.device != hidden_states.device:
                    cur_memory = cur_memory.to(hidden_states.device)

            else:

                if cat_to_maximum_memory:

                    cur_memory = delta_memory[:, idx]
                    old_memory = self.memory[idx].detach().unsqueeze(0).repeat(len(hidden_states), 1, 1) # detach might be unnecessary, but just to make sure

                    # put on cuda
                    if old_memory.device != hidden_states.device:
                        old_memory = old_memory.to(hidden_states.device)

                    # randomly sample (old_memory.shape[1] - cur_memory.shape[1]) from old_memory
                    sampled_indices = torch.randperm(old_memory.shape[1])[:old_memory.shape[1] - cur_memory.shape[1]]
                    # sort sampled_indices to make sure it is in ascending order
                    sampled_indices = sampled_indices.sort()[0]
                    old_memory = old_memory[:, sampled_indices, :]

                    cur_memory = torch.cat([
                        old_memory,
                        cur_memory,
                    ], dim=1)
                
                else:

                    cur_memory = delta_memory[:, idx]
                
            if self.add_bos_embedding:
                if self.bos_embedding[idx].device != cur_memory.device:
                    cur_memory = torch.cat([self.bos_embedding[idx].unsqueeze(0).repeat(len(cur_memory), 1, 1).to(cur_memory.device), cur_memory], dim=1)

                else:
                    cur_memory = torch.cat([self.bos_embedding[idx].unsqueeze(0).repeat(len(cur_memory), 1, 1), cur_memory], dim=1)
        else:

            if self.add_bos_embedding:
                cur_memory = self.bos_embedding[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1)

        hidden_states = torch.cat([cur_memory, hidden_states], dim=1)
        return hidden_states
    

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        delta_memory: Optional[List[List[torch.FloatTensor]]] = None,
        labels: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_delta_memory: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_injection: Optional[bool] = None,
        cat_to_maximum_memory: Optional[bool] = False,
    ) -> Union[Tuple, MemoryLMOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_injection is None:
            is_injection = output_delta_memory

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.model.gradient_checkpointing and self.model.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        # if cache_position is None:
        # TODO: currently ignore cache_position
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        
        if self.initialized:
            if past_seen_tokens > 0:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            else:
                if is_injection:
                    cache_position = torch.arange(
                        0, inputs_embeds.shape[1] + int(self.add_bos_embedding), device=inputs_embeds.device
                    )
                    memory_length = int(np.ceil(inputs_embeds.shape[1] / self.compress_ratio))

                elif delta_memory is not None:
                    if cat_to_maximum_memory:
                        cache_position = torch.arange(
                            0, inputs_embeds.shape[1] + self.num_tokens * self.num_blocks + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
                    else:
                        cache_position = torch.arange(
                            0, inputs_embeds.shape[1] + delta_memory.shape[2] + int(self.add_bos_embedding), device=inputs_embeds.device
                        )
                else:
                    cache_position = torch.arange(
                        0, inputs_embeds.shape[1] + self.num_tokens * self.num_blocks + int(self.add_bos_embedding), device=inputs_embeds.device
                    )
                    
        
        else:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)
        # TODO: currently ignore position_ids
            
        position_ids = cache_position.unsqueeze(0)

        try:
            causal_mask = self.model._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
        except:
            import ipdb; ipdb.set_trace()
        
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_delta_memory = [] if output_delta_memory else None

        if self.add_decoder_lora:

            if is_injection or (delta_memory is not None and delta_memory.shape[2] == self.num_tokens and not cat_to_maximum_memory):
                for name, module in self.named_modules():
                    if hasattr(module, "_active_adapter"):
                        module._active_adapter = ['default',]
            else:
                for _, module in self.named_modules():
                    if hasattr(module, "_active_adapter"):
                        # module._active_adapter = ['default', 'decoder_adapter']
                        module._active_adapter = ['decoder_adapter']

        for idx, decoder_layer in enumerate(self.model.layers):
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if past_key_values is None or past_key_values.get_seq_length(layer_idx=idx) == 0:

                hidden_states = self.cat_memory_and_hiddens(idx,
                                                hidden_states=hidden_states,
                                                delta_memory=delta_memory,
                                                is_injection=is_injection,
                                                cat_to_maximum_memory=cat_to_maximum_memory)
            
                prefix_token_length = hidden_states.shape[1] - inputs_embeds.shape[1] if self.initialized else 0

                # if is_injection and prefix_token_length > 0:
                #     prefix_token_length = min(prefix_token_length, hidden_states.shape[1] - self.num_tokens)

                if self.initialized and is_injection:
                    if self.new_memory_positional_emb.device != hidden_states.device:
                        hidden_states[:, -memory_length:] += self.new_memory_positional_emb.to(hidden_states.device)
                    else:
                        hidden_states[:, -memory_length:] += self.new_memory_positional_emb

            else:
                prefix_token_length = 0

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                prefix_token_length=prefix_token_length,
                # debug=debug
            )

            hidden_states = layer_outputs[0]

            if output_delta_memory:
                if self.initialized:
                    all_delta_memory.append(hidden_states[:, - memory_length:])
                else:
                    all_delta_memory.append(hidden_states[:, - self.num_tokens:])

            hidden_states = hidden_states[:, -input_ids.shape[1]:]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
        hidden_states = self.model.norm(hidden_states)
            
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if output_delta_memory:

            if all_delta_memory[0].device != all_delta_memory[-1].device:
                assert not self.training
                device = all_delta_memory[0].device
                all_delta_memory = [x.to(device) for x in all_delta_memory]
                delta_memory = torch.stack(all_delta_memory, dim=0).transpose(0, 1)

            else:

                delta_memory = torch.stack(all_delta_memory, dim=0).transpose(0, 1)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            return tuple(v for v in [loss, logits, hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return MemoryLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            delta_memory=delta_memory,
        )
    