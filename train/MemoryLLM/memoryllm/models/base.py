import os
import torch
import numpy as np
import pytorch_lightning as pl
from MemoryLLM.memoryllm.util import calculate_exact_hit_accuracy
from collections import OrderedDict
from MemoryLLM.memoryllm.util import instantiate_from_config
import torch.nn.functional as F
import json
import copy
import random

from transformers.utils import logging

logger = logging.get_logger(__name__)

class BaseMemoryModelPL(pl.LightningModule):
    def __init__(self, 
                 monitor=None,
                 optimizer=None,
                 validation_dataset_names=None,
                 cat_memories=False,
                 cat_and_drop_memory=False,
                 new_context_number_ratio=None,
                 cut_unrelated_context=False,
                 adjust_weight_strategy=None,
                 adjust_memory_ratio_strategy=None,
                 detach_additional_memory=False,
                 shuffle_contexts=False,
                 shuffle_contexts_ratio=0.0,
                 num_contexts_schedule=None,
                 update_memory_during_training=False,
                 cat_memory_when_one_context=False,
                 backup_memory_when_validating=False,
                 related_position_when_validation='begin', 
                 occassionally_cat_memory_when_one_context_ratio=None,
                 detach_indices_when_one_context=False,
                 keep_gradient_for_the_last_step=False,
                 random_sample_length_ratio=1.0,
                 cache_data_for_longer_context=False,
                 remove_attention=False,
                 pass_ratio=0.5,
                 warmup_delta_memory=False):

        super(BaseMemoryModelPL, self).__init__()
    
        if monitor is not None:
            self.monitor = monitor

        self.optimizer = optimizer
        self.backup_memory_when_validating = backup_memory_when_validating
        self.new_context_number_ratio = new_context_number_ratio
        self.validation_dataset_names = validation_dataset_names
        self.cut_unrelated_context = cut_unrelated_context
        self.update_memory_during_training = update_memory_during_training
        self.num_contexts_schedule = num_contexts_schedule
        self.detach_additional_memory = detach_additional_memory
        self.shuffle_contexts = shuffle_contexts
        self.shuffle_contexts_ratio = shuffle_contexts_ratio
        self.cat_and_drop_memory = cat_and_drop_memory
        self.cat_memories = cat_memories
        self.cat_memory_when_one_context = cat_memory_when_one_context
        self.detach_indices_when_one_context = detach_indices_when_one_context
        self.keep_gradient_for_the_last_step = keep_gradient_for_the_last_step
        self.cache_data_for_longer_context = cache_data_for_longer_context
        self.occassionally_cat_memory_when_one_context_ratio = occassionally_cat_memory_when_one_context_ratio
        self.related_position_when_validation = related_position_when_validation
        self.random_sample_length_ratio = random_sample_length_ratio
        self.remove_attention = remove_attention
        self.pass_ratio = pass_ratio
        self.warmup_delta_memory = warmup_delta_memory

        if adjust_weight_strategy is not None:
            self.adjust_weight_strategy = instantiate_from_config(adjust_weight_strategy)
        else:
            self._negative_weight = 1
            self.adjust_weight_strategy = None
        
        if adjust_memory_ratio_strategy is not None:
            self.adjust_memory_ratio_strategy = instantiate_from_config(adjust_memory_ratio_strategy)
        else:
            self.adjust_memory_ratio_strategy = None
        

        if self.cache_data_for_longer_context:
            self.last_indicator = 1
            self.cached_sentence_ids = None
            self.cached_sentence_masks = None
            self.nuc_length = 0

        self.stable_weights = {}
        self.predictions = []
        self.targets = []

        self.validation_step_outputs = []

    def num_contexts(self, global_step):
        
        if self.num_contexts_schedule is None:
            return 1
        
        elif isinstance(self.num_contexts_schedule, list):

            num_of_contexts = 1

            while num_of_contexts <= len(self.num_contexts_schedule) and global_step > self.num_contexts_schedule[num_of_contexts - 1]:
                num_of_contexts += 1

            return num_of_contexts
    
        else:
            idx = 0
            while idx < len(self.num_contexts_schedule['checkpoints']) and \
                    global_step > self.num_contexts_schedule['checkpoints'][idx]:
                idx += 1
            return self.num_contexts_schedule['values'][idx]

    @property
    def negative_weight(self):
        if self.adjust_weight_strategy is None:
            return self._negative_weight
        else:
            return self.adjust_weight_strategy.get_weight()
    
    def training_step(self, batch, batch_idx):

        if len(batch) == 5:
            contexts_ids, contexts_attention_masks, sentence_ids, sentence_attention_masks, labels = batch
        else:
            if self.remove_attention:
                contexts_ids, sentence_ids, labels = batch
                contexts_attention_masks = None
                labels = [1]
            else:
                contexts_ids, contexts_attention_masks, sentence_ids, sentence_attention_masks, labels = batch

        additional_kwargs = {}
        if self.adjust_memory_ratio_strategy is not None:
            additional_kwargs['memory_ratio'] = self.adjust_memory_ratio_strategy.get_weight(self.trainer.global_step)

        skip_injection = False
        if self.cache_data_for_longer_context:
            
            if self.cat_and_drop_memory:
                if self.last_indicator == 1 or self.nuc_length == 0:
                    indicator = 0
                else:
                    # randomly choose between (0, 1)
                    indicator = random.randint(0, 1)
                
                if indicator == 1:
                    sentence_ids = self.cached_sentence_ids
                    sentence_attention_masks = self.cached_sentence_masks
                    skip_injection = True
            else:
                # if we do not use cat_and_drop_memory, it doesn't matter if we have just injected the context.
                if self.last_indicator == 1:
                    indicator = 0
                else:
                    # randomly choose between (0, 1)
                    if np.random.random() < self.pass_ratio:
                        indicator = 0
                    else:
                        indicator = 1

                if indicator == 1:
                    sentence_ids = self.cached_sentence_ids
                    sentence_attention_masks = self.cached_sentence_masks
                    skip_injection = True

        if not skip_injection:

            if self.num_contexts_schedule is not None:

                num_of_contexts = self.num_contexts(self.trainer.global_step)

                if isinstance(self.num_contexts_schedule, list):
                    checkpoints = self.num_contexts_schedule
                else:
                    checkpoints = self.num_contexts_schedule['checkpoints']

                if self.trainer.global_step in checkpoints:

                    if not self.trainer.global_step == checkpoints[0]:
                        
                        if isinstance(self.num_contexts_schedule, list):
                            self.stable_weights[len(self.stable_weights)] = self.negative_weight
                        else:
                            # for instance: 
                            # checkpoints: [5000, 10000, 15000, 20000, 50000]
                            # values: [1, 2, 3, 4, 5, 10]
                            # now assume we are at step 10000, then we would update:
                            # self.stable_weights[2] = self.negative_weight
                            self.stable_weights[self.num_contexts_schedule['values'][checkpoints.index(self.trainer.global_step)]] = self.negative_weight
                        
                        if self.adjust_weight_strategy is not None:
                            self.adjust_weight_strategy.reset()

                if np.random.random() < self.random_sample_length_ratio:
                    num = min(np.random.randint(1, num_of_contexts+1), len(contexts_ids))
                else:
                    num = min(num_of_contexts, len(contexts_ids))

                # determine the weight for loss
                # num = 1: means it's there is only one context, so we don't need weight
                # num != 1 and num != num_of_contexts: means it's in the middle of the contexts, so we use the stable weight
                # num == num_of_contexts: means it's the last context, so we use the negative weight
                if num == 1:
                    weight = 1
                    update_negative_weight = True if num_of_contexts > 1 else False
                elif num == num_of_contexts:
                    weight = self.negative_weight
                    update_negative_weight = False
                else:
                    # now assume num_of_contets = 3 and num = 2; that means we are at step 10000 - 15000
                    # then we would self.stable_weights[2], which makes sense
                    if num in self.stable_weights:
                        weight = self.stable_weights[num]
                        update_negative_weight = True
                    else:
                        weight = self.negative_weight
                        update_negative_weight = False

                num_of_contexts = num

                if labels[0] == 0:
                    # cut the end of the contexts    
                    contexts_ids = contexts_ids[:num_of_contexts]
                    if not self.remove_attention:
                        contexts_attention_masks = contexts_attention_masks[:num_of_contexts]
                elif labels[0] == 1:
                    # cut the beginning of the contexts
                    contexts_ids = contexts_ids[-num_of_contexts:]
                    if not self.remove_attention:
                        contexts_attention_masks = contexts_attention_masks[-num_of_contexts:]
                elif labels[0] == 2:
                    # it means the sentence is the second part of the contexts
                    if num_of_contexts == 1 and len(contexts_ids) > 1:
                        contexts_ids = contexts_ids[1:][:num_of_contexts]
                        if not self.remove_attention:
                            contexts_attention_masks = contexts_attention_masks[1:][:num_of_contexts]
                    else:
                        contexts_ids = contexts_ids[:num_of_contexts]
                        if not self.remove_attention:
                            contexts_attention_masks = contexts_attention_masks[:num_of_contexts]
                else:
                    raise ValueError("Invalid label")

            else:
                weight = 1
                update_negative_weight = False

                num_of_contexts = len(contexts_ids)

                if self.new_context_number_ratio is None:
                    # random cut number of negative contexts:
                    num = min(np.random.randint(1, num_of_contexts+1), len(contexts_ids))
                else:
                    if np.random.random() < self.new_context_number_ratio:
                        num = min(num_of_contexts, len(contexts_ids))
                    else:
                        num = min(np.random.randint(1, num_of_contexts), len(contexts_ids))

                num_of_contexts = num

                if labels[0] == 0:
                    # cut the end of the contexts    
                    contexts_ids = contexts_ids[:num_of_contexts]
                    if not self.remove_attention:
                        contexts_attention_masks = contexts_attention_masks[:num_of_contexts]
                else:
                    # cut the beginning of the contexts
                    contexts_ids = contexts_ids[-num_of_contexts:]
                    if not self.remove_attention:
                        contexts_attention_masks = contexts_attention_masks[-num_of_contexts:]
            
            if self.cat_and_drop_memory:

                if len(contexts_ids) > 1:

                    all_delta_memory = None
                    delta_memory = None

                    if self.shuffle_contexts and labels[0] == 0:
                        # only shuffle when there are unrelated contexts
                        context_indices = np.random.permutation(len(contexts_ids))
                    else:
                        context_indices = np.arange(len(contexts_ids))

                    if self.detach_additional_memory:
                        
                        detach_indices = torch.ones(self.model.num_blocks * self.model.num_tokens, dtype=torch.long)

                    for i in context_indices:

                        output = self.model(input_ids=contexts_ids[i],
                                            attention_mask=contexts_attention_masks[i] if not self.remove_attention else None,
                                            output_delta_memory=True,
                                            delta_memory=delta_memory,
                                            is_injection=True,)

                        if i < len(context_indices) - 1 or (not self.keep_gradient_for_the_last_step):
                            delta_memory = output.delta_memory.detach()
                        else:
                            delta_memory = output.delta_memory

                        if all_delta_memory is None:
                            all_delta_memory = delta_memory
                        else:
                            all_delta_memory = self.model.drop_memory(all_delta_memory[0]).unsqueeze(0)
                            all_delta_memory = torch.cat([
                                all_delta_memory,
                                delta_memory
                            ], dim=2)

                    delta_memory = all_delta_memory.detach()
                    torch.cuda.empty_cache()

                    if self.detach_additional_memory:
                        # randomly choose 2 * self.model.num_tokens from the last (all_delta_memory.shape[2]) positions to set to zero
                        # detach_indices[np.random.choice(self.model.num_blocks * self.model.num_tokens, 2 * self.model.num_tokens, replace=False)] = 0
                        if delta_memory.shape[2] > 2 * self.model.num_tokens:
                            detach_indices[(self.model.num_blocks * self.model.num_tokens - delta_memory.shape[2]) + np.random.choice(delta_memory.shape[2], 2 * self.model.num_tokens, replace=False)] = 0
                        else:
                            detach_indices[-delta_memory.shape[2]:] = 0
                            detach_indices[np.random.choice(self.model.num_blocks * self.model.num_tokens - delta_memory.shape[2], 2 * self.model.num_tokens - delta_memory.shape[2], replace=False)] = 0
                else:
                    
                    output = self.model(input_ids=contexts_ids[-1],
                                        attention_mask=contexts_attention_masks[-1] if not self.remove_attention else None,
                                        output_delta_memory=True,
                                        is_injection=True)

                    delta_memory = output.delta_memory

            else:

                if len(contexts_ids) > 1:
                    for i in range(len(contexts_ids)-1):
                        output = self.model(input_ids=contexts_ids[i],
                                            attention_mask=contexts_attention_masks[i] if not self.remove_attention else None,
                                            output_delta_memory=True,
                                            is_injection=True,)
                        self.model.update_memory_with_delta_memory(output.delta_memory.detach())
                
                output = self.model(input_ids=contexts_ids[-1],
                                    attention_mask=contexts_attention_masks[-1] if not self.remove_attention else None,
                                    output_delta_memory=True,
                                    is_injection=True, **additional_kwargs)
                
                delta_memory = output.delta_memory

        else:
            weight = 1.0
            update_negative_weight = False
            if not self.cat_and_drop_memory:
                output = self.model(input_ids=contexts_ids[-1],
                                    attention_mask=contexts_attention_masks[-1] if not self.remove_attention else None,
                                    output_delta_memory=True,
                                    is_injection=True, **additional_kwargs)
                
                delta_memory = output.delta_memory

                self.nuc_length += 1

        sentence_labels = sentence_ids.clone()
        if not self.remove_attention:
            sentence_labels[sentence_attention_masks[:, self.model.num_tokens:] == 0] = -100

        if not skip_injection:
            if self.cat_memories:
                if delta_memory.shape[2] > self.model.num_tokens:
                    sentence_attention_masks = torch.cat([
                                torch.ones(sentence_attention_masks.shape[0], delta_memory.shape[2] - self.model.num_tokens).to(sentence_attention_masks.device),
                                sentence_attention_masks,
                            ], dim=1)
                if self.detach_additional_memory and len(contexts_ids) > 1:
                    additional_kwargs['detach_indices'] = detach_indices
            
            cat_memory_when_one_context = False
            if self.cat_memory_when_one_context:
                cat_memory_when_one_context = True
            elif self.occassionally_cat_memory_when_one_context_ratio is not None:
                if np.random.random() < self.occassionally_cat_memory_when_one_context_ratio:
                    cat_memory_when_one_context = True
                    delta_memory = delta_memory.detach()

            if self.cat_and_drop_memory and delta_memory.shape[2] > self.model.num_tokens:
                
                sentence_attention_masks = torch.cat([
                            torch.ones(sentence_attention_masks.shape[0], self.model.num_tokens * self.model.num_blocks - self.model.num_tokens).to(sentence_attention_masks.device),
                            sentence_attention_masks,
                        ], dim=1)
                
                if self.detach_additional_memory and len(contexts_ids) > 1:
                    additional_kwargs['detach_indices'] = detach_indices
                
            else:
                if self.cat_and_drop_memory and cat_memory_when_one_context:
                    sentence_attention_masks = torch.cat([
                                torch.ones(sentence_attention_masks.shape[0], self.model.num_tokens * self.model.num_blocks - self.model.num_tokens).to(sentence_attention_masks.device),
                                sentence_attention_masks,
                            ], dim=1)
                    additional_kwargs['cat_memory_when_one_context'] = cat_memory_when_one_context

                    if self.detach_indices_when_one_context:
                        # we need to detach the memories:
                        detach_indices = torch.ones(self.model.num_blocks * self.model.num_tokens, dtype=torch.long)
                        detach_indices[:(self.model.num_blocks - 1)*self.model.num_tokens] = 0
                        additional_kwargs['detach_indices'] = detach_indices

        else:
            if self.cat_and_drop_memory:
                delta_memory = None
                sentence_attention_masks = torch.cat([
                            torch.ones(sentence_attention_masks.shape[0], self.model.num_tokens * self.model.num_blocks - self.model.num_tokens).to(sentence_attention_masks.device),
                            sentence_attention_masks,
                        ], dim=1)
        
        output = self.model(input_ids=sentence_ids,
                attention_mask=sentence_attention_masks if not self.remove_attention else None,
                labels=sentence_labels,
                delta_memory=delta_memory,
                output_delta_memory=False,
                is_injection=False,
                return_dict=True, **additional_kwargs)

        loss = output['loss']

        loss *= weight

        if not self.cat_and_drop_memory and self.regularization_scheduler is not None:
            regularization_loss, m0_norm, difference_norm = self.regularization_scheduler.get_regularization_loss(self.model.memory.detach(), delta_memory)
            self.log_dict({'loss/regularization': regularization_loss}, logger=True, on_step=True)
            self.log_dict({'loss/m0_norm': m0_norm}, logger=True, on_step=True)
            self.log_dict({'loss/difference_norm': difference_norm}, logger=True, on_step=True)
        else:
            regularization_loss = 0

        if update_negative_weight:
            if self.adjust_weight_strategy is not None:
                self.adjust_weight_strategy.update_weight(loss / weight)
                if self.adjust_weight_strategy.previous_loss is not None:
                    self.log_dict({'nw/running_loss': self.adjust_weight_strategy.previous_loss}, logger=True, on_step=True)
                    self.log_dict({'nw/cur_loss': loss}, logger=True, on_step=True)
    
        else:
            # log weight:
            self.log_dict({'nw/weight': float(weight)}, logger=True, on_step=True)

        if self.cache_data_for_longer_context:

            if indicator == 1:
                # self.log_dict({'loss/unrelated_c_{}'.format(self.nuc_length): loss / weight}, logger=True, on_step=True)
                self.log_dict({'loss/unrelated': (loss) / weight}, logger=True, on_step=True)
                self.log_dict({'loss/unrelated_nuc': self.nuc_length}, logger=True, on_step=True)

            else:
                if self.last_indicator == 1:
                    self.cached_sentence_ids = sentence_ids
                    self.cached_sentence_masks = sentence_attention_masks[:, - (self.model.num_tokens + sentence_ids.shape[1]):]
                    self.nuc_length = 0
                
                else:
                    self.nuc_length += len(contexts_ids)

                if self.num_contexts_schedule is not None:
                    self.log_dict({'loss/{}_c_{}'.format("related" if (labels[0] == 1 or num_of_contexts == 1) else "unrelated", num_of_contexts): loss / weight}, logger=True, on_step=True)
                else:
                    self.log_dict({'loss/{}_c_{}'.format("related" if (labels[0] == 1 or len(contexts_ids) == 1) else "unrelated", len(contexts_ids)): loss / weight}, logger=True, on_step=True)

            self.last_indicator = indicator

        else:
            # log the loss according to the number of contexts:
            if self.num_contexts_schedule is not None:
                self.log_dict({'loss/{}_c_{}'.format("related" if (labels[0] ==1 or num_of_contexts == 1) else "unrelated", num_of_contexts): loss / weight}, logger=True, on_step=True)
            else:
                if len(contexts_ids) == 1:
                    if cat_memory_when_one_context:
                        self.log_dict({'loss/related_c_with_memory_{}'.format(len(contexts_ids)): loss / weight}, logger=True, on_step=True)
                    else:
                        self.log_dict({'loss/related_c_{}'.format(len(contexts_ids)): loss / weight}, logger=True, on_step=True)

                else:
                    self.log_dict({'loss/{}_c_{}'.format("related" if labels[0] == 1 else "unrelated", len(contexts_ids)): loss / weight}, logger=True, on_step=True)

        if not self.warmup_delta_memory or (self.warmup_delta_memory and self.trainer.global_step > 5000):
            if self.update_memory_during_training and delta_memory is not None:
                with torch.no_grad():
                    self.model.update_memory_with_delta_memory(delta_memory)
            
        self.log_dict({"loss": loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss + regularization_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        cat_memories = self.cat_memories

        context_ids, context_attention_mask, \
        sentence_ids, sentence_attention_mask, \
        answer_ids, answer_attention_mask = batch[:6]

        unrelated_contexts_and_mask = batch[6:]

        qa_inputs = torch.cat([
            sentence_ids,
            answer_ids
        ], dim=1)
        qa_labels = qa_inputs.clone()
        qa_labels[:, :sentence_ids.shape[1]] = -100

        if not self.remove_attention:
            if self.cat_and_drop_memory:
                if sentence_attention_mask.shape[1] < self.model.memory.shape[1]:
                    new_sentence_attention_mask = torch.cat([
                        torch.ones(sentence_attention_mask.shape[0], self.model.memory.shape[1] - self.model.num_tokens).to(sentence_attention_mask.device),
                        sentence_attention_mask,
                    ], dim=1)
                else:
                    new_sentence_attention_mask = sentence_attention_mask
            else:
                new_sentence_attention_mask = sentence_attention_mask
        else:
            new_sentence_attention_mask = None

        if not self.remove_attention:
            forward_sentence_attention = torch.cat([
                new_sentence_attention_mask,
                torch.ones(answer_ids.shape).long().to(sentence_attention_mask.device)
            ], dim=1)
        else:
            forward_sentence_attention = None
        
        loss_without_context = self.model(input_ids=qa_inputs, attention_mask=forward_sentence_attention, labels=qa_labels).loss.item()

        if self.backup_memory is not None:
            self.model.memory.data = self.backup_memory.clone().to(self.model.memory.device)

        if hasattr(self.tokenizer, "memory_token_id"):
            context_ids, context_attention_mask = self.pad_memory_tokens(context_ids, context_attention_mask)

        delta_memory = self.model.inject_memory(
            context_ids,
            context_attention_mask if not self.remove_attention else None,
            update_memory=True
        )

        try:
            output = self.model.generate(
                inputs=sentence_ids, 
                attention_mask=new_sentence_attention_mask,
                max_new_tokens=10,
                pad_token_id=self.tokenizer.pad_token_id,
            )[:, len(sentence_ids[0]):][0].detach().cpu()
        except:
            output = torch.tensor([1, 1, 1])

        loss_with_context = self.model(input_ids=qa_inputs, attention_mask=forward_sentence_attention, labels=qa_labels).loss.item()

        if cat_memories:
            all_delta_memories = []
            all_delta_memories.append(delta_memory[0])

        middle_outputs = []
        middle_losses = []
        for idx in range(len(unrelated_contexts_and_mask) // 2):

            unrelated_context, unrelated_context_attention = unrelated_contexts_and_mask[idx*2],\
                unrelated_contexts_and_mask[idx*2 + 1],
            if hasattr(self.tokenizer, "memory_token_id"):
                unrelated_context, unrelated_context_attention = \
                    self.pad_memory_tokens(unrelated_context, unrelated_context_attention)

            # assert unrelated_contexts_and_mask[idx*2].shape[1] <= 384
            delta_memory = self.model.inject_memory(
                unrelated_context, unrelated_context_attention,
                update_memory=True,
            )

            if cat_memories:
                all_delta_memories.append(delta_memory[0])

            if not cat_memories:

                # if self.cat_and_drop_memory:
                #     if sentence_attention_mask.shape[1] < self.model.memory.shape[1]:
                #         new_sentence_attention_mask = torch.cat([
                #             torch.ones(sentence_attention_mask.shape[0], self.model.memory.shape[1] - self.model.num_tokens).to(sentence_attention_mask.device),
                #             sentence_attention_mask,
                #         ], dim=1)
                #     else:
                #         new_sentence_attention_mask = sentence_attention_mask
                # else:
                #     new_sentence_attention_mask = sentence_attention_mask
                try:
                    middle_out = self.model.generate(
                        inputs=sentence_ids, 
                        attention_mask=new_sentence_attention_mask,
                        max_new_tokens=10,
                        pad_token_id=self.tokenizer.eos_token_id
                    )[:, len(sentence_ids[0]):][0].detach().cpu()
                except:
                    middle_out = torch.tensor([1, 1, 1])

                middle_outputs.append(middle_out)
                loss = self.model(input_ids=qa_inputs, attention_mask=torch.cat([
                        new_sentence_attention_mask,
                        torch.ones(answer_ids.shape).long().to(sentence_attention_mask.device)
                    ], dim=1), labels=qa_labels).loss.item()
                middle_losses.append(loss)

        if cat_memories:

            for idx in range(len(unrelated_contexts_and_mask) // 2):

                self.model.update_memory_with_delta_memory(torch.cat(
                    all_delta_memories[:idx+2]
                , dim=1).unsqueeze(0))

                if self.model.memory.shape[1] > self.model.num_tokens:
                    new_sentence_attention_mask = torch.cat([
                                torch.ones(sentence_attention_mask.shape[0], self.model.memory.shape[1] - self.model.num_tokens).to(sentence_attention_mask.device),
                                sentence_attention_mask,
                            ], dim=1)
                else:
                    new_sentence_attention_mask = sentence_attention_mask
                try:
                    middle_out = self.model.generate(
                        inputs=sentence_ids, 
                        attention_mask=new_sentence_attention_mask,
                        max_new_tokens=10,
                        pad_token_id=self.tokenizer.eos_token_id
                    )[:, len(sentence_ids[0]):][0].detach().cpu()
                    middle_outputs.append(middle_out)
                except:
                    middle_outputs.append(torch.tensor([1,1,1]))

            self.model.update_memory_with_delta_memory(all_delta_memories[0].unsqueeze(0))

        if dataloader_idx > len(self.validation_step_outputs) - 1:
            self.validation_step_outputs.append([])

        self.validation_step_outputs[dataloader_idx].append({
            'dataloader_idx': dataloader_idx,
            'prediction': output,
            'target': answer_ids[0].cpu(),
            'dataloader_idx': dataloader_idx,
            "middle_outputs": middle_outputs,
            "loss_without_context": loss_without_context,
            'loss_with_context': loss_with_context,
            'middle_losses': middle_losses
        })

        if cat_memories:
            self.model.memory.data = self.model.memory.data[:, -self.model.num_tokens:]

        # return {'prediction': output, 
        #         'target': answer_ids[0].cpu(), 
        #         'dataloader_idx': dataloader_idx, 
        #         "middle_outputs": middle_outputs,
        #         "loss_without_context": loss_without_context,
        #         'loss_with_context': loss_with_context,
        #         'middle_losses': middle_losses}

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []
        if self.backup_memory_when_validating:
            self.backup_memory = self.model.memory.data.clone().detach()
        else:
            self.backup_memory = None
    
    def on_validation_epoch_end(self):

        # import ipdb; ipdb.set_trace()
        outputs = self.validation_step_outputs

        # if len(self.validation_dataset_names) == 1:
        #     outputs = [outputs]
        # generated_outputs = {}
        accs = []

        for output in outputs:

            preds = self.tokenizer.batch_decode([x['prediction'].tolist() for x in output])
            # targets = self.tokenizer.batch_decode([x['target'].tolist()[:-1] for x in output])
            # TODO: check why "[:-1]" is not needed here
            targets = self.tokenizer.batch_decode([x['target'].tolist() for x in output])
            dataloader_idx = output[0].get('dataloader_idx', 0)
            accuracy = calculate_exact_hit_accuracy(preds, targets)

            # generated_outputs[f'targets_{self.validation_dataset_names[dataloader_idx]}'] = targets
            # generated_outputs[f'predictions_{self.validation_dataset_names[dataloader_idx]}'] = preds

            # calculate loss
            loss_without_context = np.mean([x['loss_without_context'] for x in output])
            loss_with_context = np.mean([x['loss_with_context'] for x in output])

            # log
            if self.validation_dataset_names is not None:
                self.log(f'val/{self.validation_dataset_names[dataloader_idx]}', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log(f'val/{self.validation_dataset_names[dataloader_idx]}_bold_loss', loss_without_context, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log(f'val/{self.validation_dataset_names[dataloader_idx]}_w_context', loss_with_context, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

                # generated_outputs[f'val/{self.validation_dataset_names[dataloader_idx]}'] = accuracy

            else:
                self.log(f'val/{dataloader_idx}', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log(f'val/{dataloader_idx}_bold_loss', loss_without_context, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log(f'val/{dataloader_idx}_w_context', loss_with_context, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            accs.append(accuracy)

            # log middle outputs
            for step in range(len(output[0]['middle_outputs'])):
                preds_cur_step = []
                for idx in range(len(preds)):
                    preds_cur_step.append(output[idx]['middle_outputs'][step].tolist())
                
                preds_cur_step = self.tokenizer.batch_decode(preds_cur_step)

                accuracy = calculate_exact_hit_accuracy(preds_cur_step, targets)

                if self.validation_dataset_names is not None:
                    self.log(f'val/m_{self.validation_dataset_names[dataloader_idx]}_{step+1}', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                    self.log(f'val/{self.validation_dataset_names[dataloader_idx]}_{step+1}_loss', np.mean([x['middle_losses'][step] for x in output]), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                    # generated_outputs[f'val/m_{self.validation_dataset_names[dataloader_idx]}_{step+1}'] = accuracy
                else:
                    self.log(f'val/m_{idx}_{step+1}', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                    self.log(f'val/{idx}_{step+1}_loss', np.mean([x['middle_losses'][step] for x in output]), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

                # generated_outputs[f'predictions_{self.validation_dataset_names[dataloader_idx]}_step_{step}'] = preds_cur_step
        
        # calculate average accuracy
        avg_acc = np.mean(accs)

        self.log('val/avg_acc', avg_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # if os.path.exists("results"):
        #     count = 0
        #     generated_outputs['avg_acc'] = avg_acc
        #     filename = "./results/val_results_{}.json"
        #     while os.path.exists(filename.format(count)):
        #         count += 1
        #     with open(filename.format(count), "w") as f:
        #         json.dump(generated_outputs, f)
        #     f.close()

        # return {
        #     'generated_outputs': generated_outputs
        # }

    def configure_optimizers(self):

        if self.optimizer is None:
            # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            optimizer = torch.optim.AdamW([param for param in self.model.parameters() if param.requires_grad], lr=self.learning_rate)
            
        else:
            # optimizer: deepspeed.ops.adam.DeepSpeedCPUAdam
            if "deepspeed" in self.optimizer:
                import deepspeed
                import ninja
            if "onebit" in self.optimizer:
                from deepspeed.runtime.fp16 import onebit
            if "FusedAdam" in self.optimizer:
                from deepspeed.ops.adam import FusedAdam
            optimizer = eval(self.optimizer)
            optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
            # optimizer = optimizer([param for param in self.model.parameters() if param.requires_grad], lr=self.learning_rate)

        return optimizer
        
    def init_from_ckpt(self, ckpt_path):
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "checkpoint/mp_rank_00_model_states.pt")
        sd = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        else:
            def rename_keys(state_dict):
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    new_key = key.replace("_forward_module.", "")
                    new_state_dict[new_key] = value
                return new_state_dict
            sd = rename_keys(sd["module"])

        if 'model.base_model.model.new_memory_positional_emb' in sd.keys():
            if sd['model.base_model.model.new_memory_positional_emb'].shape != self.model.base_model.model.new_memory_positional_emb.shape:
                try:
                    sd['model.base_model.model.new_memory_positional_emb'] = sd['model.base_model.model.new_memory_positional_emb'].view(self.model.base_model.model.new_memory_positional_emb.shape)
                except:
                    del sd['model.base_model.model.new_memory_positional_emb']

        if 'model.base_model.model.memory' in sd.keys():
            if sd['model.base_model.model.memory'].shape != self.model.base_model.model.memory.shape:
                # delete the key in sd
                del sd['model.base_model.model.memory']

        if 'model.base_model.model.model.embed_tokens.weight' in sd.keys():
            if sd['model.base_model.model.model.embed_tokens.weight'].shape != self.model.base_model.model.model.embed_tokens.weight.shape:
                del sd['model.base_model.model.model.embed_tokens.weight']
        if 'model.base_model.model.lm_head.weight' in sd.keys():
            if sd['model.base_model.model.lm_head.weight'].shape != self.model.base_model.model.lm_head.weight.shape:
                del sd['model.base_model.model.lm_head.weight']

        # filtered_state_dict = {k: v for k, v in sd.items() if k != 'model.memory'}
        # missing, unexpected = self.load_state_dict(filtered_state_dict, strict=False)
        missing, unexpected = self.load_state_dict(sd, strict=False)

        if len(missing) > 0 and (hasattr(self.model, "split_encoder_decoder") and self.model.split_encoder_decoder) and 'decoder' in missing[1]:

            # Assume it is LoRA model
            # Create a copy of the items to iterate over, so we don't modify the dictionary while iterating
            items = list(sd.items())

            # Loop through the copied list of items
            for key, value in items:
                if "model.base_model.model.model" in key:
                    # Replace the key as needed and update the original dictionary
                    new_key = key.replace("model.base_model.model.model", "model.base_model.model.decoder")
                    sd[new_key] = value

        missing, unexpected = self.load_state_dict(sd, strict=False)

        # missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        print("Missing:", missing)
        print("Unexpected:", unexpected)



