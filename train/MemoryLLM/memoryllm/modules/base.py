import torch

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
        remaining_indices=None
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
        self.remaining_indices = remaining_indices
        self.last_hidden_state = last_hidden_state


class BaseMemoryModel(ABC):
    def __init__(self, config):
        self.config = config

    def inject_memory(self, context_ids, 
                            context_attention_masks,
                            delta_memory=None,
                            update_memory=False):

        output = self(input_ids=context_ids,
                attention_mask=context_attention_masks,
                delta_memory=delta_memory,
                output_delta_memory=True,
                return_dict=True)

        if update_memory:

            if output.delta_memory is None:
                # dirty fix for now
                self.update_memory_with_delta_memory(output.logits)
            else:
                self.update_memory_with_delta_memory(output.delta_memory)

            return output.delta_memory

        else:
            return output.delta_memory

    def drop_memory(self, current_memory, drop_length=None):

        if self.drop_memory_per_layer:

            all_indices = []

            for _ in range(current_memory.shape[0]):

                if drop_length is None:
                    left_indices = torch.randperm(current_memory.shape[1])[:current_memory.shape[1] - int(current_memory.shape[1] * (1 / self.num_blocks))]
                else:
                    left_indices = torch.randperm(current_memory.shape[1])[:current_memory.shape[1] - drop_length]
                
                left_indices = left_indices.sort()[0]

                all_indices.append(left_indices)

            all_indices = torch.stack(all_indices)

            current_memory = torch.gather(current_memory, 1, all_indices.unsqueeze(-1).repeat(1, 1, current_memory.shape[-1]).to(current_memory.device))

            return current_memory

        else:

            if drop_length is None:
                left_indices = torch.randperm(current_memory.shape[1])[:current_memory.shape[1] - int(current_memory.shape[1] * (1 / self.num_blocks))]
            else:
                left_indices = torch.randperm(current_memory.shape[1])[:current_memory.shape[1] - drop_length]
            
            # sort left_indices to make sure it is in ascending order
            left_indices = left_indices.sort()[0]

            current_memory = current_memory[:, left_indices, :]
            
            return current_memory

    def update_memory_with_delta_memory(self, delta_memory, remaining_indices=None):
        
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

            if self.put_memory_on_cpu:
                self.memory.data = delta_memory.detach().cpu()
            else:
                self.memory.data = delta_memory

        else:

            if delta_memory.shape[1] == self.num_tokens:

                current_memory = self.memory.data.detach() # detach might be unnecessary, but just to make sure

                if len(self.exclude_layers) > 0:
                    # Debug features
                    for l_i in range(current_memory.shape[0]):
                        if l_i not in self.exclude_layers:
                            current_memory_layer_i = current_memory[l_i].unsqueeze(0)
                            current_memory_layer_i = self.drop_memory(current_memory_layer_i)[0]
                            self.memory.data[l_i] = torch.cat([current_memory_layer_i, delta_memory[l_i]], dim=0)

                else:

                    # current_memory.shape: [L, num_blocks * num_tokens, d]
                    # we need to drop 1/num_blocks current_memories on dimension 1:
                    current_memory = self.drop_memory(current_memory)
                    if self.put_memory_on_cpu:
                        self.memory.data = torch.cat([current_memory, delta_memory.detach().cpu()], dim=1)
                    else:
                        if current_memory.device != delta_memory.device:
                            self.memory.data = torch.cat([current_memory, delta_memory.to(current_memory.device)], dim=1)
                        else:
                            try:
                                self.memory.data = torch.cat([current_memory, delta_memory], dim=1)
                            except:
                                print('''
                                RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":1123, please report a bug to PyTorch.
                                      ''')
                                self.memory.data = torch.cat([current_memory.cpu(), delta_memory.cpu()], dim=1)
                    
                    # assert self.memory.data.shape == (self.L, self.num_blocks * self.num_tokens, self.d)

            else:
                current_memory = self.memory.data.detach() # detach might be unnecessary, but just to make sure
                # current_memory.shape: [L, num_blocks * num_tokens, d]
                current_memory = self.drop_memory(current_memory, delta_memory.shape[1])
                if current_memory.device != delta_memory.device:
                    print("current_memory.device != delta_memory.device")
                    self.memory.data = torch.cat([current_memory, delta_memory], dim=1)
                else:
                    self.memory.data = torch.cat([current_memory, delta_memory.to(current_memory.device)], dim=1)

                # assert self.memory.data.shape == (self.L, self.num_blocks * self.num_tokens, self.d)
        
        if not self.initialized:
            self.initialized += 1


    def cat_memory_and_hiddens(self, idx, hidden_states, delta_memory=None, 
                               is_injection=False,
                               cat_memory_when_one_context=False):
        
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

                if delta_memory.shape[2] > self.num_tokens:

                    old_memory = self.memory.detach()[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1) # detach might be unnecessary, but just to make sure

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

                if delta_memory.shape[2] == self.num_tokens and cat_memory_when_one_context:
                    # we need to cat the memory when there is only one context
                    old_memory = self.memory.detach()[idx].unsqueeze(0).repeat(len(hidden_states), 1, 1) # detach might be unnecessary, but just to make sure

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


    def customized_generate(
        self, 
        inputs_ids,
        inputs_masks,
        tokenizer,
        max_new_tokens,
        delta_memory=None,
    ):

        assert len(inputs_ids) == 1, "We currently only support generation with batch_size=1"

        count = 0
        while (not inputs_ids[0][-1].eq(tokenizer.eos_token_id)) and count < max_new_tokens:

            model_outputs = self(
                input_ids=inputs_ids,
                attention_mask=inputs_masks,
                delta_memory=delta_memory,
                output_delta_memory=False,
                return_dict=True
            )

            count += 1

            new_id = model_outputs.logits[0][-1].argmax()

            inputs_ids = torch.cat(
                [inputs_ids,
                torch.tensor([new_id]).unsqueeze(0).to(inputs_ids.device)], dim=-1
            )
            inputs_masks = torch.cat(
                [inputs_masks,
                torch.tensor([1]).unsqueeze(0).to(inputs_masks.device)], dim=-1
            )

        return inputs_ids
