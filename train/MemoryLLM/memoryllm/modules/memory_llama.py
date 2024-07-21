from .modeling_llama import LlamaForCausalLM, apply_rotary_pos_emb, rotate_half, LlamaDecoderLayer
import sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import copy
from MemoryLLM.memoryllm.modules.base import BaseMemoryModel, MemoryLMOutputWithPastAndCrossAttentions
from typing import Optional, Tuple, Union, List
from torch.nn import CrossEntropyLoss
from transformers.utils import logging

logger = logging.get_logger(__name__)

class LlamaDropMemoryModel(LlamaForCausalLM, BaseMemoryModel):
    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)
        BaseMemoryModel.__init__(self, config)
        
        self.config = config
        self.L = config.num_hidden_layers
        self.d = config.hidden_size
        self.num_blocks = config.num_blocks
        self.num_tokens = config.num_tokens

        self.add_bos_embedding = config.add_bos_embedding
        self.shrink_to_one_embedding = config.shrink_to_one_embedding
        self.drop_memory_per_layer = config.drop_memory_per_layer if hasattr(config, "drop_memory_per_layer") else False
        
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
            self.bos_embedding.data[0, :] = self.model.embed_tokens.weight.data[1]
            self.bos_embedding = nn.Parameter(self.bos_embedding)

    def set_exclude_layers(self, layer_indices):
        self.exclude_layers = layer_indices

    def super_forward(self, *args, **kwargs):
        return LlamaForCausalLM.forward(self, *args, **kwargs)

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
        is_injection: Optional[bool] = None,
        detach_indices: Optional[torch.LongTensor] = None,
        cat_memory_when_one_context: Optional[bool] = False,
    ) -> Union[Tuple, MemoryLMOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_injection is None:
            is_injection = output_delta_memory

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        model = self.model

        if inputs_embeds is None:
            inputs_embeds = model.embed_tokens(input_ids)

        if past_key_values is not None:

            if not self.initialized:
                past_key_values_length = 0
                attention_mask = attention_mask[:, self.memory.shape[1]:]

            past_key_values_length = past_key_values[-1][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            position_length = seq_length
            position_ids = torch.arange(
                past_key_values_length, position_length + past_key_values_length, dtype=torch.long, device=device
            )

            # embed positions
            if attention_mask is None:
                raise NotImplementedError
                # attention_mask = torch.ones(
                #     (batch_size, self.num_tokens + seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                # )
            
            position_ids = position_ids.unsqueeze(0).view(-1, position_length)

            if self.initialized and self.add_bos_embedding:
                attention_mask = torch.cat([torch.ones([attention_mask.shape[0], 1]).long().to(model.device), attention_mask], dim=1)

            prefix_token_length = 0

            attention_mask = model._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, 
            )

        else:

            past_key_values_length = 0
            # position_length = (seq_length + self.num_tokens + int(self.add_bos_embedding)) if self.initialized else seq_length
            # if delta_memory is None or self.share_position_ids:
            if delta_memory is None:
                if is_injection:
                    position_length = (seq_length + self.num_tokens + int(self.add_bos_embedding)) if self.initialized else seq_length
                else:
                    position_length = (seq_length + self.memory.shape[1] + int(self.add_bos_embedding)) if self.initialized else seq_length

            else:
                if delta_memory.shape[2] == self.num_tokens:
                    if cat_memory_when_one_context:
                        position_length = (seq_length + self.memory.shape[1] + int(self.add_bos_embedding)) if self.initialized else seq_length
                    else:
                        position_length = (seq_length + delta_memory.shape[2] + int(self.add_bos_embedding)) if self.initialized else seq_length

                else:
                    position_length = (seq_length + self.memory.shape[1] + int(self.add_bos_embedding)) if self.initialized else seq_length
            
            position_ids = torch.arange(
                past_key_values_length, position_length + past_key_values_length, dtype=torch.long, device=device
            )
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, position_length - int(self.add_bos_embedding)), dtype=torch.bool, device=inputs_embeds.device
                )
        
            position_ids = position_ids.unsqueeze(0).view(-1, position_length)

            if not self.initialized:
                if is_injection:
                    attention_mask = attention_mask[:, self.num_tokens:]
                else:
                    if delta_memory is not None:
                        attention_mask = attention_mask[:, delta_memory.shape[2]:]
                    else:
                        attention_mask = attention_mask[:, self.memory.shape[1]:]

            else:
                if self.add_bos_embedding:
                    attention_mask = torch.cat([torch.ones([batch_size, 1]).long().to(model.device), attention_mask], dim=1)

            if self.initialized:
                if delta_memory is None:
                    if is_injection:
                        # prefix_token_length = input_ids.shape[1]
                        prefix_token_length = min(input_ids.shape[1], self.num_tokens)
                        
                    else:
                        prefix_token_length = self.memory.shape[1]
                    prefix_token_length += 1 if self.add_bos_embedding else 0
                else:
                    if is_injection:
                        
                        # prefix_token_length = input_ids.shape[1] + delta_memory.shape[2] - self.num_tokens
                        prefix_token_length = input_ids.shape[1] + delta_memory.shape[2] - max(self.num_tokens, input_ids.shape[1])

                    else:
                        if delta_memory.shape[2] == self.num_tokens:
                            if cat_memory_when_one_context:
                                prefix_token_length = self.memory.shape[1]
                            else:
                                prefix_token_length = delta_memory.shape[2]
                        else:
                            prefix_token_length = self.memory.shape[1]
                    prefix_token_length += 1 if self.add_bos_embedding else 0

            else:
                prefix_token_length = 0

            attention_mask = model._prepare_decoder_attention_mask(
                attention_mask, (batch_size, attention_mask.shape[1]), inputs_embeds, past_key_values_length, 
                prefix_token_length=prefix_token_length
            )

        hidden_states = inputs_embeds

        if model.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        all_delta_memory = [] if output_delta_memory else None

        for idx, decoder_layer in enumerate(model.layers):
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if past_key_value is None:
                hidden_states = self.cat_memory_and_hiddens(idx,
                                                hidden_states=hidden_states,
                                                delta_memory=delta_memory,
                                                is_injection=is_injection,
                                                cat_memory_when_one_context=cat_memory_when_one_context)

            if model.gradient_checkpointing and self.training:

                raise NotImplementedError("gradient_checkpointing in our method not implemented")

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
                
                hidden_states = layer_outputs[0]

            else:
                
                if not self.initialized:

                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                
                else:

                    if output_delta_memory:
                        assert past_key_value is None
                        if self.new_memory_positional_emb.device != hidden_states.device:
                            hidden_states[:, -self.num_tokens:] += self.new_memory_positional_emb.to(hidden_states.device)
                        else:
                            hidden_states[:, -self.num_tokens:] += self.new_memory_positional_emb

                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        prefix_token_length=prefix_token_length,
                        detach_indices=detach_indices,
                    )

                hidden_states = layer_outputs[0]

                if output_delta_memory:
                    all_delta_memory.append(hidden_states[:, -self.num_tokens:])
                
                hidden_states = hidden_states[:, -input_ids.shape[1]:]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if model.norm.weight.device != hidden_states.device:
            assert not self.training
            norm = model.norm.to(hidden_states.device)
            hidden_states = norm(hidden_states)
        else:
            hidden_states = model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
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

        next_cache = next_decoder_cache if use_cache else None
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
