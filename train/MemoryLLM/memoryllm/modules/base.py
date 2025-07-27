import time
import torch
import numpy as np

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import logging
from abc import ABC

logger = logging.get_logger(__name__)

class MemoryLMOutputWithPastAndCrossAttentions(CausalLMOutputWithCrossAttentions):
    def __init__(
        self,
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        cross_attentions=None,
        delta_memory=None,
        last_hidden_state=None,
        retriever_weights=None,
        encoder_retriever_weights=None,
        ltm_indices=None,
    ):
        super().__init__(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            cross_attentions=cross_attentions,
        )
        self.delta_memory = delta_memory
        self.last_hidden_state = last_hidden_state
        self.retriever_weights = retriever_weights
        self.encoder_retriever_weights = encoder_retriever_weights
        self.ltm_indices = ltm_indices


class BaseMemoryModel(ABC):
    def __init__(self, config):
        self.config = config

    def fill_in_ltm(self, delta_memory):

        if len(delta_memory.shape) == 4:
            delta_memory = delta_memory.detach()[0]

        with torch.no_grad():
            for idx in range(len(self.memory)):
                # current_memory = self.memory.data[idx]
                # self.ltm[idx] = current_memory[-self.num_tokens * self.num_ltm_blocks:].detach().cpu()
                self.ltm[idx] = delta_memory[idx].detach().cpu()
                self.ltm_recall_frequencies[idx] = torch.ones(self.ltm[idx].shape[0]) * self.initial_rf_when_moving_stm_to_ltm
                self.ltm_keys[idx] = self.model.layers[idx].self_attn.key_proj(self.model.layers[idx].input_layernorm(delta_memory[idx].detach())).cpu()
                # self.ltm_ages[idx] = torch.zeros(self.ltm[idx].shape[0])
                self.ltm_ages[idx] = np.zeros(self.ltm[idx].shape[0], dtype=int)

    def inject_memory(self, context_ids, 
                            context_attention_mask=None,
                            delta_memory=None,
                            update_memory=False,
                            use_retriever=False):

        output = self(input_ids=context_ids,
                attention_mask=context_attention_mask,
                delta_memory=delta_memory,
                is_injection=True,
                output_delta_memory=True,
                return_dict=True)

        if update_memory:
            delta_memory = output.delta_memory
            if use_retriever:
                # get retriever_weights
                all_retriever_weights = []
                for idx in range(delta_memory.shape[1]):
                    delta_memory_queries = self.model.layers[idx].self_attn.encoder_query_proj(
                        self.model.layers[idx].input_layernorm(delta_memory[0, idx]))
                    if self.maintain_memory_keys:
                        memory_keys = self.memory_keys[idx]
                    else:
                        memory_keys = self.model.layers[idx].self_attn.key_proj(
                            self.model.layers[idx].input_layernorm(self.memory[idx]))
                    retriever_weights = (delta_memory_queries @ memory_keys.transpose(-2, -1)).sigmoid().mean(dim=0)
                    all_retriever_weights.append(retriever_weights)
                retriever_weights = torch.stack(all_retriever_weights)

            else:
                retriever_weights = None

            self.update_memory_with_delta_memory(delta_memory, retriever_weights=retriever_weights)
            return delta_memory

        else:
            return output.delta_memory

    def drop_memory(self, current_memory, drop_length=None, unsequeezed=True, return_remaining_indices=False, return_dropped_indices=False):

        if hasattr(self, "virtual_num_blocks") and self.virtual_num_blocks is not None:

            cur_memory_length = current_memory.shape[1] if unsequeezed else current_memory.shape[0]

            perm_indices = torch.randperm(cur_memory_length)

            if drop_length is None:
                drop_length = int(cur_memory_length * (1 / self.num_blocks))
            
            remaining_indices = perm_indices[:cur_memory_length - int(drop_length * self.num_blocks / self.virtual_num_blocks)]
            remaining_indices = remaining_indices.sort()[0]
            if cur_memory_length - drop_length == 0:
                remaining_indices = remaining_indices[:0]
            else:
                remaining_indices = remaining_indices[-(cur_memory_length - drop_length):]
            # dropped_indices = perm_indices[len(remaining_indices):]
            # dropped_indices = dropped_indices.sort()[0]
            # TODO: check if this is correct
            dropped_indices = torch.tensor(np.setdiff1d(np.arange(cur_memory_length), remaining_indices.cpu().numpy())).sort()[0]

            if unsequeezed:
                current_memory = current_memory[:, remaining_indices, :]
            else:
                current_memory = current_memory[remaining_indices, :]

        else:

            if unsequeezed:

                perm_indices = torch.randperm(current_memory.shape[1])

                if drop_length is None:
                    remaining_indices = perm_indices[:current_memory.shape[1] - int(current_memory.shape[1] * (1 / self.num_blocks))]
                else:
                    remaining_indices = perm_indices[:current_memory.shape[1] - drop_length]
                
                # sort remaining_indices to make sure it is in ascending order
                remaining_indices = remaining_indices.sort()[0]
                dropped_indices = perm_indices[len(remaining_indices):]
                dropped_indices = dropped_indices.sort()[0]

                current_memory = current_memory[:, remaining_indices, :]
                
            else:

                perm_indices = torch.randperm(current_memory.shape[0])

                if drop_length is None:
                    remaining_indices = perm_indices[:current_memory.shape[0] - int(current_memory.shape[0] * (1 / self.num_blocks))]
                else:
                    remaining_indices = perm_indices[:current_memory.shape[0] - drop_length]
                
                # sort remaining_indices to make sure it is in ascending order
                remaining_indices = remaining_indices.sort()[0]
                dropped_indices = perm_indices[len(remaining_indices):]
                dropped_indices = dropped_indices.sort()[0]

                current_memory = current_memory[remaining_indices, :]
        
        if return_remaining_indices and return_dropped_indices:
            return current_memory, remaining_indices, dropped_indices
        if return_remaining_indices:
            return current_memory, remaining_indices
        elif return_dropped_indices:
            return current_memory, dropped_indices
        else:
            return current_memory

    def update_memory_with_delta_memory(self, 
                                        delta_memory, 
                                        cached_contexts_indicators=None, 
                                        is_ltm=False,
                                        retriever_weights=None, 
                                        delta_memory_ages=None, 
                                        return_dropped_memories=False,
                                        memory=None):
        
        if len(delta_memory.shape) == 4:
            delta_memory = delta_memory.detach()[0]

        if self.initialized == 0:

            if delta_memory.shape[1] < (self.num_tokens * self.num_blocks):
                if ((self.num_tokens * self.num_blocks) % delta_memory.shape[1]) == 0:
                    delta_memory = torch.cat(
                        [delta_memory] * ((self.num_tokens * self.num_blocks) // delta_memory.shape[1]), dim=1
                    )
                else:
                    delta_memory = torch.cat(
                        [delta_memory] * ((self.num_tokens * self.num_blocks) // delta_memory.shape[1]) + 
                        [delta_memory[:, -((self.num_tokens * self.num_blocks) % delta_memory.shape[1]):]], dim=1
                    )

            else:
                delta_memory = delta_memory[:, -self.num_tokens * self.num_blocks:]

            self.memory.data = delta_memory
            with torch.no_grad():
                if self.maintain_memory_keys:
                    self.memory_keys = []
                    for idx in range(len(self.memory)):
                        self.memory_keys.append(self.model.layers[idx].self_attn.key_proj(self.model.layers[idx].input_layernorm(self.memory[idx])).data)
                    self.memory_keys = torch.stack(self.memory_keys)

            self.initialized += 1

            if is_ltm:
                self.memory_ages = np.zeros([self.memory.shape[0], self.memory.shape[1]], dtype=int)

        else:
            
            dropped_memory = [] if return_dropped_memories else None
            dropped_memory_ages = [] if return_dropped_memories else None

            for idx in range(len(self.memory)):

                if memory is None:
                    current_memory = self.memory.data[idx].detach()
                else:
                    current_memory = memory[idx].detach()

                if retriever_weights is not None:

                    retriever_labels = retriever_weights[idx] > 0.5
                    remaining_indices = torch.where(retriever_labels == 1)[0]

                    diff = delta_memory.shape[1] - (len(retriever_labels) - len(remaining_indices))
                    if diff > 0:
                        retriever_labels[remaining_indices[:diff]] = False
                        remaining_indices = torch.where(retriever_labels == 1)[0]
                    
                    indices_to_drop = torch.where(retriever_labels == 0)[0]
                    # randomly drop delta_memory.shape[1] indices in indices_to_drop
                    remaining_indices = torch.cat([
                        remaining_indices,
                        indices_to_drop[torch.randperm(len(indices_to_drop))[:len(indices_to_drop) - delta_memory.shape[1]]]
                    ]).cpu()
                    remaining_indices = remaining_indices.sort()[0]
                    self.memory.data[idx] = torch.cat([
                        current_memory[remaining_indices],
                        delta_memory[idx]
                    ])

                    if return_dropped_memories:
                        # TODO: fill this later
                        raise NotImplementedError


                else:

                    current_memory, remaining_indices, dropped_indices = self.drop_memory(current_memory, 
                                            delta_memory.shape[1], unsequeezed=False, 
                                            return_remaining_indices=True,
                                            return_dropped_indices=True)
                    if return_dropped_memories:
                        dropped_memory.append(
                            self.memory.data[idx][dropped_indices]
                        )
                    self.memory.data[idx] = torch.cat([current_memory, delta_memory[idx]], dim=0)

                if self.maintain_memory_keys:
                    self.memory_keys[idx] = torch.cat([
                        self.memory_keys[idx][remaining_indices],
                        self.model.layers[idx].self_attn.key_proj(self.model.layers[idx].input_layernorm(delta_memory[idx])).data
                    ])

                if is_ltm:

                    if len(remaining_indices) == 0:
                        
                        if delta_memory_ages is not None:

                            if return_dropped_memories:
                                dropped_memory_ages.append(self.memory_ages[idx] + 1 + max(delta_memory_ages[idx]))

                            self.memory_ages[idx] = delta_memory_ages[idx]
                        else:

                            raise NotImplementedError

                            if self.update_ltm_mode == 'immediate':
                                self.memory_ages[idx] = np.arange(delta_memory.shape[1])[::-1]
                            else:
                                self.memory_ages[idx] = np.zeros(delta_memory.shape[1])
                        
                        self.memory_recall_frequency[idx] = np.zeros(delta_memory.shape[1])
                        self.memory_position_indicators[idx] = np.ones(delta_memory.shape[1])

                    else:
                        
                        # np.array([1,2,3])[torch.tensor([2])] gives 3
                        # np.array([1,2,3])[np.array([2])] gives [3], we need the latter one
                        remaining_indices = np.array(remaining_indices)

                        self.memory_ages[idx] += (1 + max(delta_memory_ages[idx])) if delta_memory_ages is not None else self.num_tokens

                        if return_dropped_memories:
                            dropped_memory_ages.append(self.memory_ages[idx][dropped_indices])

                        if delta_memory_ages is not None:
                            self.memory_ages[idx] = np.concatenate([
                                self.memory_ages[idx][remaining_indices],
                                delta_memory_ages[idx]
                            ])
                            assert delta_memory_ages[idx].shape[0] == delta_memory[idx].shape[0]

                        else:
                            assert delta_memory.shape[1] == self.num_tokens
                            self.memory_ages[idx] = np.concatenate([
                                self.memory_ages[idx][remaining_indices],
                                # np.zeros([delta_memory.shape[1]])
                                np.arange(delta_memory.shape[1])[::-1]
                            ])

                        if self.update_ltm_mode == 'decouple':
                            self.memory_recall_frequency[idx] = np.concatenate([
                                self.memory_recall_frequency[idx][remaining_indices], 
                                np.zeros([delta_memory.shape[1]])
                            ])
                            remaining_position_indicators = self.memory_position_indicators[idx][remaining_indices]
                            if np.min(remaining_position_indicators) > 1:
                                remaining_position_indicators -= np.min(remaining_position_indicators) - 1
                            self.memory_position_indicators[idx] = np.concatenate([
                                remaining_position_indicators, np.ones(delta_memory.shape[1]) * (np.max(remaining_position_indicators) + 1)
                            ])

                if cached_contexts_indicators is not None:
                    if len(cached_contexts_indicators.shape) == 3:
                        # cached_contexts_indicators: [1, L, num_memory_tokens, d]
                        cached_contexts_indicators[:, idx] = torch.cat([
                            cached_contexts_indicators[:, idx][:, remaining_indices],
                            torch.zeros([cached_contexts_indicators.shape[0], delta_memory.shape[1]]).to(cached_contexts_indicators.device)
                        ], dim=1)
                    else:
                        # cached_contexts_indicators: [L, num_memory_tokens, d]
                        cached_contexts_indicators[idx] = torch.cat([
                            cached_contexts_indicators[idx][remaining_indices],
                            torch.zeros([delta_memory.shape[1]]).to(cached_contexts_indicators.device)
                        ])

        outputs = ()
        if cached_contexts_indicators is not None:
            outputs = (cached_contexts_indicators,)
        if return_dropped_memories:
            outputs += ((dropped_memory, dropped_memory_ages),)
        
        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def cat_memory_and_hiddens(self, idx, hidden_states, delta_memory=None, 
                               is_injection=False,
                               cat_to_maximum_memory=False):
        
        if not self.initialized:
            return hidden_states
    
        if delta_memory is None or len(delta_memory) == 0:

            if is_injection:
                cur_memory = self.memory[idx][ - self.num_tokens:].unsqueeze(0).repeat(len(hidden_states), 1, 1)
            else:
                cur_memory = self.memory[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1)
            
            # put on cuda
            if cur_memory.device != hidden_states.device:
                cur_memory = cur_memory.to(hidden_states.device)

        else:

            cur_memory = delta_memory[:, idx]
            
            if is_injection:

                assert cur_memory.shape[1] == self.num_tokens

            else:

                if self.virtual_num_blocks is not None:
                    
                    if cat_to_maximum_memory:

                        old_memory = self.memory[idx].detach().unsqueeze(0).repeat(len(hidden_states), 1, 1)

                        # now we have 12800 in old_memory, 
                        # we need to drop cur_memory.shape[1] tokens
                        # we can randomly drop cur_memory.shape[1] * (self.num_blocks / self.virtual_num_blocks) tokens, then drop the leftmost tokens to make sure we drop 256 tokens
                        sampled_indices = torch.randperm(old_memory.shape[1])[:old_memory.shape[1] - int(cur_memory.shape[1] * (self.num_blocks / self.virtual_num_blocks))]
                        sampled_indices = sampled_indices.sort()[0]
                        if (old_memory.shape[1] - cur_memory.shape[1]) == 0:
                            pass
                        else:
                            sampled_indices = sampled_indices[-(old_memory.shape[1] - cur_memory.shape[1]):]
                            old_memory = old_memory[:, sampled_indices, :]

                            cur_memory = torch.cat([
                                old_memory,
                                cur_memory,
                            ], dim=1)

                else:

                    if delta_memory.shape[2] > self.num_tokens:

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

                    if delta_memory.shape[2] == self.num_tokens and cat_to_maximum_memory:
                        # we need to cat the memory when there is only one context
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

        if self.add_bos_embedding:
            if self.bos_embedding[idx].device != cur_memory.device:
                cur_memory = torch.cat([self.bos_embedding[idx].unsqueeze(0).repeat(len(cur_memory), 1, 1).to(cur_memory.device), cur_memory], dim=1)
            else:
                cur_memory = torch.cat([self.bos_embedding[idx].unsqueeze(0).repeat(len(cur_memory), 1, 1), cur_memory], dim=1)

        return torch.cat([cur_memory, hidden_states], dim=1)

    # The followings are the functions for long-term memory
    def get_stm(self, 
                idx,
                hidden_states,
                delta_memory=None,
                is_injection=False,
                cat_to_maximum_memory=False):
        
        if delta_memory is None or len(delta_memory) == 0:
            if is_injection:
                cur_memory = self.memory[idx][ - self.num_tokens:].unsqueeze(0).repeat(len(hidden_states), 1, 1)
            else:
                cur_memory = self.memory[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1)
        else:
            cur_memory = delta_memory[:, idx]
            if (not is_injection) and cat_to_maximum_memory:

                old_memory = self.memory[idx].detach().unsqueeze(0).repeat(len(hidden_states), 1, 1) # detach might be unnecessary, but just to make sure

                # put on cuda
                if old_memory.device != hidden_states.device:
                    old_memory = old_memory.to(hidden_states.device)

                # randomly sample (old_memory.shape[1] - cur_memory.shape[1]) from old_memory
                if hasattr(self, "virtual_num_blocks") and self.virtual_num_blocks is not None:
                    sampled_indices = torch.randperm(old_memory.shape[1])[:old_memory.shape[1] - int(cur_memory.shape[1] * (self.num_blocks / self.virtual_num_blocks))]
                    sampled_indices = sampled_indices.sort()[0]
                    if (old_memory.shape[1] - cur_memory.shape[1]) == 0:
                        pass
                    else:
                        sampled_indices = sampled_indices[-(old_memory.shape[1] - cur_memory.shape[1]):]
                        old_memory = old_memory[:, sampled_indices, :]

                    cur_memory = torch.cat([
                        old_memory,
                        cur_memory,
                    ], dim=1)

                else:
                    sampled_indices = torch.randperm(old_memory.shape[1])[:old_memory.shape[1] - cur_memory.shape[1]]
                    # sort sampled_indices to make sure it is in ascending order
                    sampled_indices = sampled_indices.sort()[0]
                    old_memory = old_memory[:, sampled_indices, :]
                    cur_memory = torch.cat([
                        old_memory,
                        cur_memory,
                    ], dim=1)
                
        return cur_memory

    def get_ltm(self, idx, hidden_states, random_retriever_length=False):

        num_ltm_tokens = self.num_ltm_tokens if hasattr(self, "num_ltm_tokens") else self.num_ltm_blocks * self.num_tokens

        # get ltm_keys if ltm_keys are None
        if self.ltm_keys[idx] is None:

            with torch.no_grad():

                ltm = self.ltm[idx]
                tmp_ltm_keys = []
                batch_size = 64

                for batch_ltm in torch.split(ltm, batch_size):
                    batch_ltm = batch_ltm.to(hidden_states.device).to(hidden_states.dtype)
                    ltm_keys = self.model.layers[idx].self_attn.key_proj(self.model.layers[idx].input_layernorm(batch_ltm))
                    tmp_ltm_keys.append(ltm_keys)
                tmp_ltm_keys = torch.cat(tmp_ltm_keys, dim=0)

            tmp_ltm_keys = tmp_ltm_keys.detach().cpu()
            self.ltm_keys[idx] = tmp_ltm_keys

        with torch.no_grad():

            if len(self.ltm_keys[idx]) < num_ltm_tokens:

                indices = torch.tensor([])
                while len(indices) < num_ltm_tokens:
                    indices = torch.cat([
                        torch.arange(len(self.ltm_keys[idx])),
                        indices
                    ])
                indices = indices[-num_ltm_tokens:].to(torch.int)
            
            else:
                
                if random_retriever_length:
                    length = torch.randint(1, hidden_states.size(1), (1,)).item()
                    hidden_states = hidden_states[:, :length, :]

                queries = self.model.layers[idx].self_attn.query_proj(self.model.layers[idx].input_layernorm(hidden_states[0]))
                predictions = (queries @ self.ltm_keys[idx].to(queries.device).transpose(-2, -1)).sigmoid().mean(dim=0)

                if self.cached_dropped_memories is not None:
                    # cached_dropped_keys = self.model.layers[idx].self_attn.key_proj(self.model.layers[idx].input_layernorm(self.cached_dropped_memories[idx].to(queries.device)))
                    # cached_predictions = (queries @ cached_dropped_keys.transpose(-2, -1)).sigmoid().mean(dim=0)
                    cached_predictions = (queries @ self.cached_dropped_keys[idx].to(queries.device).transpose(-2, -1)).sigmoid().mean(dim=0)
                    predictions = torch.cat([predictions, cached_predictions], dim=0)

                indices = torch.topk(predictions, k=num_ltm_tokens).indices
                indices = torch.sort(indices)[0].cpu()

        # ages = self.ltm_ages[idx][indices.detach().cpu()]
        # indices = indices[np.argsort(ages)[::-1].copy()]
        # x = self.ltm[idx][indices.detach().cpu()].to(hidden_states.device)
        # return x, indices.detach().cpu()

        if self.cached_dropped_memories is None:
            ages = self.ltm_ages[idx][indices.detach().cpu()]
            indices = indices[np.argsort(ages)[::-1].copy()]
            x = self.ltm[idx][indices.detach().cpu()].to(hidden_states.device)
            return x.detach(), indices.detach().cpu()
        
        else:

            with torch.no_grad():

                ltm_indices = indices[torch.where(indices < self.ltm[idx].shape[0])[0]]
                dropped_indices = indices[len(ltm_indices):] - self.ltm[idx].shape[0]

                ltm_ages = self.ltm_ages[idx][np.array(ltm_indices.detach().cpu())]
                dropped_ages = self.cached_dropped_memory_ages[idx][np.array(dropped_indices.detach().cpu())]

                ltm_x = self.ltm[idx][ltm_indices].to(hidden_states.device)
                dropped_x = self.cached_dropped_memories[idx][dropped_indices].to(hidden_states.device)

                if len(dropped_ages) > 0 and len(ltm_ages) > 0:
                    all_ages = np.concatenate([ltm_ages, dropped_ages])
                else:
                    all_ages = ltm_ages if len(ltm_ages) > 0 else dropped_ages

                all_x = torch.cat([ltm_x, dropped_x], dim=0)

                all_x = all_x[np.argsort(all_ages)[::-1].copy()]

            return all_x.detach(), ltm_indices
    
    def update_recall_frequency(self, idx, hidden_states):

        with torch.no_grad():
            queries = self.model.layers[idx].self_attn.encoder_query_proj(self.model.layers[idx].input_layernorm(hidden_states[0]))
            keys = self.model.layers[idx].self_attn.key_proj(self.model.layers[idx].input_layernorm(self.memory[idx]))
            indices = torch.where((queries @ keys.transpose(-2, -1)).sigmoid().mean(dim=0) > 0.5)[0]
            self.memory_recall_frequency[idx][indices.detach().cpu().numpy()] += 1
    
    def update_ltm(self, dropped_memories=None, dropped_memory_ages=None, device=None, cached_dropped_keys=None):

        with torch.no_grad():

            # update ltm according to memory_recall_frequency
            for idx in range(len(self.memory)):

                if self.update_ltm_mode == 'decouple':

                    current_memory = self.memory.data[idx]
                    current_memory_recall_frequency = self.memory_recall_frequency[idx]
                    current_memory_position_indicators = self.memory_position_indicators[idx]

                    # find the positions with indicators being 0
                    # within the positions with indicators being 0, find the positions with recall frequency larger than 0
                    # put the positions with recall frequency larger than 0 into the long-term memory

                    start_step = np.max(current_memory_position_indicators) - self.update_ltm_frequency - self.update_ltm_T0 + 1
                    end_step = np.max(current_memory_position_indicators) - self.update_ltm_T0 + 1

                    positions_to_input_into_ltm = np.where((current_memory_position_indicators >= start_step) & (current_memory_position_indicators <= end_step) & (current_memory_recall_frequency > 1))[0]
                    memory_to_input_into_ltm = current_memory[positions_to_input_into_ltm]

                    self.ltm[idx] = torch.cat([
                        self.ltm[idx],
                        memory_to_input_into_ltm.detach().cpu()
                    ])

                    if self.initial_rf_when_moving_stm_to_ltm is None:
                        self.ltm_recall_frequencies[idx] = np.concatenate(
                            [self.ltm_recall_frequencies[idx],
                            current_memory_recall_frequency[positions_to_input_into_ltm]]
                        )
                    else:
                        self.ltm_recall_frequencies[idx] = np.concatenate(
                            [self.ltm_recall_frequencies[idx],
                            np.ones(positions_to_input_into_ltm.shape[0]) * self.initial_rf_when_moving_stm_to_ltm]
                        )

                    # TODO: change this to using "memory_keys" when self.maintain_memory_keys is True
                    self.ltm_keys[idx] = torch.cat([
                        self.ltm_keys[idx],
                        self.model.layers[idx].self_attn.key_proj(
                            self.model.layers[idx].input_layernorm(
                                memory_to_input_into_ltm
                            )
                        ).detach().cpu()
                    ])

                    self.ltm_ages[idx] = np.concatenate([
                        self.ltm_ages[idx],
                        self.memory_ages[idx][positions_to_input_into_ltm]
                    ])

                    # if len(self.ltm[idx]) > self.converge_ltm_number_tokens:
                    self.ltm_recall_frequencies[idx] -= self.decay_frequency * (self.update_step / (self.update_ltm_frequency * self.num_tokens))
                    indices = np.where(self.ltm_recall_frequencies[idx] > 0.01)[0] # sometimes it may be "2.0539126e-15", using 0.01 to filter out these cases
                    if len(indices) > self.num_ltm_blocks * self.num_tokens:
                        self.ltm[idx] = self.ltm[idx][indices]
                        self.ltm_keys[idx] = self.ltm_keys[idx][indices]
                        self.ltm_recall_frequencies[idx] = self.ltm_recall_frequencies[idx][indices]
                        self.ltm_ages[idx] = self.ltm_ages[idx][indices]

                    else:
                        self.ltm_recall_frequencies[idx] += self.decay_frequency

                else:
                    
                    current_memory = dropped_memories[idx]
                    self.ltm[idx] = torch.cat([
                        self.ltm[idx],
                        current_memory.detach().cpu()
                    ])
                    assert self.initial_rf_when_moving_stm_to_ltm is not None
                    self.ltm_recall_frequencies[idx] = np.concatenate(
                        [self.ltm_recall_frequencies[idx],
                        np.ones(current_memory.shape[0]) * self.initial_rf_when_moving_stm_to_ltm]
                    )

                    if cached_dropped_keys is not None:

                        self.ltm_keys[idx] = torch.cat([
                            self.ltm_keys[idx],
                            cached_dropped_keys[idx]
                        ])
                    
                    else:

                        self.ltm_keys[idx] = torch.cat([
                            self.ltm_keys[idx],
                            self.model.layers[idx].self_attn.key_proj(
                                self.model.layers[idx].input_layernorm(
                                    current_memory.to(device) if device is not None else current_memory
                                )
                            ).detach().cpu()
                        ])

                    self.ltm_ages[idx] = np.concatenate([
                        self.ltm_ages[idx].astype(int),
                        dropped_memory_ages[idx]
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


