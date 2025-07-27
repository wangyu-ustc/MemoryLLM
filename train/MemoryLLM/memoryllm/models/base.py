import os
import sys
import time
import torch
import warnings
import numpy as np
import pytorch_lightning as pl
from MemoryLLM.memoryllm.util import calculate_exact_hit_accuracy, calculate_qa_f1_score
from MemoryLLM.memoryllm.modules.scheduler import LinearDecayScheduler
from MemoryLLM.memoryllm.util import instantiate_from_config
import torch.nn.functional as F
from sklearn import metrics
from transformers.utils import logging
from MemoryLLM.memoryllm.util import get_num_tokens

logger = logging.get_logger(__name__)

class BaseMemoryModelPL(pl.LightningModule):
    def __init__(self, 
                 monitor=None,
                 optimizer=None,
                 lr_scheduler=None,
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
                 cat_to_maximum_memory=False,
                 one_context_ratio=None,
                 one_context_ratio_scheduler=None,
                 backup_memory_when_validating=False,
                 related_position_when_validation='begin', 
                 occassionally_cat_to_maximum_memory_ratio=None,
                 detach_indices_when_one_context=False,
                 keep_gradient_for_the_last_step=False,
                 random_sample_length_ratio=1.0,
                 cache_data_for_longer_context=False,
                 cache_data_for_longer_context_from=0,
                 remove_attention=False,
                 pass_ratio=0.5,
                 empty_prob=1.0,
                 warmup_delta_memory=False,
                 instruct=None,
                 mask_instruction_tokens=False,
                 detach_memory_ratio=0.0,
                 is_ift=False,
                 is_pretrain_ift=False,
                 put_cache_on_cpu=False,
                 embedding_learning_rate=None,
                 half_last=False,
                 num_of_additional_tokens_to_mask=0,
                 full_context_and_sentence_training_ratio=0.0,
                 complete_with_random_memory_insertion_ratio=0.0,
                 random_insertion_max_length=4,
                 retriever_penalty_weight=0.01,
                 add_penalty_on_retriever_weights=False,
                 pretraining_selector_steps=0,
                 split_retrieval_loss_per_layer=False,
                 selector_learning_rate=None,
                 bce_loss_weight=1.0,
                 negative_loss_weight=1.0,
                 selector_loss_type='bce',
                 apply_retriever_gradients_layer_ratio=0.0,
                 log_selector_acc_interval=50,
                 eager_update_memory=False,
                 selector_loss_per_token=False,
                 random_retriever_length=False,
                 add_encoder_retriever=False,
                 use_retriever_when_dropping_from=None,
                 use_retriever_during_validation=True,
                 train_retriever_on_important_tokens=False,
                 train_with_random_memory_insertion_ratio=0.0,
                 keep_gradient_in_random_insertion=True,
                 efficient_get_delta_memory=False,
                 random_length_for_encoder=True,
                 parallel_injection=False,
                 parallel_chunk_size=None,
                 scale_randomized_poe_ratio=None,
                 scale_randomized_poe_ratio_for_memory=None,
                 randomized_poe_ratio=None,
                 randomized_poe_for_memory_ratio=None):

        super(BaseMemoryModelPL, self).__init__()
    
        if monitor is not None:
            self.monitor = monitor

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
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
        self.cat_to_maximum_memory = cat_to_maximum_memory
        self.detach_indices_when_one_context = detach_indices_when_one_context
        self.keep_gradient_for_the_last_step = keep_gradient_for_the_last_step
        self.cache_data_for_longer_context = cache_data_for_longer_context
        self.occassionally_cat_to_maximum_memory_ratio = occassionally_cat_to_maximum_memory_ratio
        self.related_position_when_validation = related_position_when_validation
        self.random_sample_length_ratio = random_sample_length_ratio
        self.remove_attention = remove_attention
        self.pass_ratio = pass_ratio
        self.warmup_delta_memory = warmup_delta_memory
        self.instruct = instruct
        self.mask_instruction_tokens = mask_instruction_tokens
        self.cache_data_for_longer_context_from = cache_data_for_longer_context_from
        self.empty_prob = empty_prob
        self.detach_memory_ratio = detach_memory_ratio
        self.is_ift = is_ift
        self.is_pretrain_ift = is_pretrain_ift
        self.put_cache_on_cpu = put_cache_on_cpu
        self.embedding_learning_rate = embedding_learning_rate
        self.selector_learning_rate = selector_learning_rate
        self._one_context_ratio = one_context_ratio
        self.one_context_ratio_scheduler = LinearDecayScheduler(
            **one_context_ratio_scheduler) if one_context_ratio_scheduler is not None else None
        self.half_last = half_last
        self.num_of_additional_tokens_to_mask = num_of_additional_tokens_to_mask
        self.full_context_and_sentence_training_ratio = full_context_and_sentence_training_ratio
        self.eager_update_memory = eager_update_memory
        self.parallel_injection = parallel_injection
        self.parallel_chunk_size = parallel_chunk_size
        self.train_retriever_on_important_tokens = train_retriever_on_important_tokens
        self.train_with_random_memory_insertion_ratio = train_with_random_memory_insertion_ratio
        self.complete_with_random_memory_insertion_ratio = complete_with_random_memory_insertion_ratio
        self.random_insertion_max_length = random_insertion_max_length
        self.efficient_get_delta_memory = efficient_get_delta_memory
        self.keep_gradient_in_random_insertion = keep_gradient_in_random_insertion
        self.random_length_for_encoder = random_length_for_encoder

        # configs for scaling up the model
        self.scale_randomized_poe_ratio = scale_randomized_poe_ratio
        self.scale_randomized_poe_ratio_for_memory = scale_randomized_poe_ratio_for_memory
        self.randomized_poe_ratio = randomized_poe_ratio
        self.randomized_poe_for_memory_ratio =randomized_poe_for_memory_ratio

        self.val_injection_steps_per_eval = 40

        # configurations about retriever weights
        self.retriever_penalty_weight = retriever_penalty_weight
        self.add_penalty_on_retriever_weights = add_penalty_on_retriever_weights
        self.split_retrieval_loss_per_layer = split_retrieval_loss_per_layer
        self.pretraining_selector_steps = pretraining_selector_steps
        self.apply_retriever_gradients_layer_ratio = apply_retriever_gradients_layer_ratio
        self.log_selector_acc_interval = log_selector_acc_interval
        self.selector_loss_per_token = selector_loss_per_token
        self.random_retriever_length = random_retriever_length
        self.add_encoder_retriever = add_encoder_retriever
        self.use_retriever_when_dropping_from = use_retriever_when_dropping_from
        self.use_retriever_during_validation = use_retriever_during_validation

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
            self.cached_sentence_labels = None
            self.cached_exp_label = None
            self.cached_contexts_indicators = None
            self.cached_contexts_length = None
            self.nuc_length = 0

        self.stable_weights = {}
        self.predictions = []
        self.targets = []

        self.validation_step_outputs = []

        self.selector_loss_type = selector_loss_type
        self.bce_loss_weight = bce_loss_weight
        self.negative_loss_weight = negative_loss_weight

    @property
    def one_context_ratio(self):
        if self.one_context_ratio_scheduler is not None:
            return self.one_context_ratio_scheduler.get_ratio(self.trainer.global_step)
        elif self._one_context_ratio is not None:
            return self._one_context_ratio
        else:
            return 0

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
    
    # def ift_training_step(self, batch, batch_idx):

    # commented for now as I constantly take ift_training_step as training_step
    # def ift_training_step(self, batch, batch_idx):
        
    #     contexts_ids, sentence_ids, sentence_labels, label = batch

    #     delta_memory = None

    #     skip_injection = False
    #     if self.cache_data_for_longer_context:
    #         if self.last_indicator == 1 or self.nuc_length == 0:
    #             indicator = 0
    #         else:
    #             # randomly choose between (0, 1)
    #             # indicator = random.randint(0, 1)
    #             if np.random.random() < self.pass_ratio:
    #                 indicator = 0
    #             else:
    #                 indicator = 1
            
    #         if indicator == 1:
    #             if self.put_cache_on_cpu:
    #                 sentence_ids = self.cached_sentence_ids.to(sentence_ids.device)
    #                 sentence_labels = self.cached_sentence_labels.to(sentence_labels.device)
    #                 label = self.cached_label
    #             else:
    #                 sentence_ids, sentence_labels = self.sentence_ids, self.sentence_labels
    #             skip_injection = True
        
    #     if not skip_injection:

    #         if self.num_contexts_schedule is not None:

    #             num_of_contexts = self.num_contexts(self.trainer.global_step)
    #             contexts_ids = contexts_ids[-num_of_contexts:]

    #         if len(contexts_ids) > 1:

    #             all_delta_memory = None
    #             delta_memory = None

    #             context_indices = np.arange(len(contexts_ids))

    #             for i in context_indices:

    #                 output = self.model(input_ids=contexts_ids[i],
    #                                     output_delta_memory=True,
    #                                     delta_memory=delta_memory,
    #                                     is_injection=True,)

    #                 delta_memory = output.delta_memory.detach()
    #                 if all_delta_memory is None:
    #                     all_delta_memory = delta_memory
    #                 else:
    #                     if self.model.drop_memory_per_layer:
    #                         new_all_delta_memory = []
    #                         for idx in range(all_delta_memory.shape[1]):
    #                             new_all_delta_memory.append(torch.cat([
    #                                 self.model.drop_memory(all_delta_memory[0, idx], 
    #                                     drop_length=int(all_delta_memory[0, idx].shape[0] * (delta_memory[0, idx].shape[0] / (self.model.num_blocks * self.model.num_tokens))), 
    #                                     unsequeezed=False),
    #                                 delta_memory[0, idx]
    #                             ], dim=0))
    #                         all_delta_memory = torch.stack(new_all_delta_memory).unsqueeze(0) 

    #                     else:
    #                         all_delta_memory = self.model.drop_memory(all_delta_memory[0]).unsqueeze(0)
    #                         all_delta_memory = torch.cat([
    #                             all_delta_memory,
    #                             delta_memory
    #                         ], dim=2)

    #             delta_memory = all_delta_memory.detach()
    #             torch.cuda.empty_cache()

    #         elif len(contexts_ids) == 1:
                
    #             output = self.model(input_ids=contexts_ids[-1],
    #                                 output_delta_memory=True,
    #                                 is_injection=True)

    #             delta_memory = output.delta_memory
            
    #         else:
    #             pass

    #     try:
            
    #         output = self.model(input_ids=sentence_ids,
    #             labels=sentence_labels,
    #             delta_memory=delta_memory,
    #             output_delta_memory=False,
    #             is_injection=False,
    #             return_dict=True,
    #             cat_to_maximum_memory=True)
            
    #     # print error
    #     except Exception as e:
    #         print(e)
    #         print(sentence_ids.shape)
    #         print(sentence_labels.shape)
    #         print(delta_memory.shape)
    #         print(delta_memory)
    #         print(contexts_ids)
    #         print("Sentence: ", self.tokenizer.decode(sentence_ids[0], skip_special_tokens=True))
    #         for context in contexts_ids:
    #             print("Context: ", self.tokenizer.decode(context[0], skip_special_tokens=True))
    #         print("sentence_labels:", sentence_labels)
        
    #     loss = output['loss']

    #     if self.cache_data_for_longer_context:

    #         cached_contexts_indicators_initialized = False

    #         if label[0] == 1:
    #             tag = 'complete'
    #         elif label[0] == 4:
    #             tag = 'repeat'
    #         elif label[0] == 2:
    #             tag = 'ift'

    #         if indicator == 1:

    #             self.log_dict({f'loss/unrelated_{tag}': loss}, logger=True, on_step=True)
    #             self.log_dict({f'loss/unrelated_{tag}_nuc': self.nuc_length}, logger=True, on_step=True)

    #         else:

    #             if self.last_indicator == 1:
    #                 if self.cached_sentence_ids is None:
    #                     self.cached_sentence_ids = sentence_ids.cpu()
    #                     self.cached_sentence_labels = sentence_labels.cpu()
    #                     self.cached_label = label.cpu()
    #                     self.nuc_length = 0
    #                     if self.add_selector: 
    #                         self.cached_contexts_indicators = torch.zeros([self.model.memory.shape[0], self.model.memory.shape[1]])
    #                         self.cached_contexts_indicators[:, -delta_memory.shape[2]:] = True
    #                         cached_contexts_indicators_initialized = True
                    
    #                 else:
    #                     if np.random.random() < self.empty_prob:
    #                         self.cached_sentence_ids = sentence_ids.cpu()
    #                         self.cached_sentence_labels = sentence_labels.cpu()
    #                         self.cached_label = label.cpu()
    #                         self.nuc_length = 0
    #                         if self.add_selector:
    #                             self.cached_contexts_indicators = torch.zeros([self.model.memory.shape[0], self.model.memory.shape[1]])
    #                             self.cached_contexts_indicators[:, -delta_memory.shape[2]:] = True
    #                             cached_contexts_indicators_initialized = True

    #             else:
    #                 self.nuc_length += len(contexts_ids)
                
    #             if len(contexts_ids) < 5:
    #                 self.log_dict({f"loss/{tag}" + "_c_{}".format(len(contexts_ids)): loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #             elif len(contexts_ids) < 10:
    #                 self.log_dict({f'loss/{tag}_c_5_to_10': loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #             elif len(contexts_ids) < 15:
    #                 self.log_dict({f'loss/{tag}_c_10_to_15': loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #             elif len(contexts_ids) < 20:
    #                 self.log_dict({f'loss/{tag}_c_15_to_20': loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #             else:
    #                 self.log_dict({f'loss/{tag}_c_20_plus': loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    #         self.last_indicator = indicator
        
    #     else:

    #         if label[0] == 1:
    #             tag = 'complete'
    #         elif label[0] == 4:
    #             tag = 'repeat'
    #         elif label[0] == 2:
    #             tag = 'ift'
            
    #         if len(contexts_ids) < 5:
    #             self.log_dict({f"loss/{tag}" + "_c_{}".format(len(contexts_ids)): loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #         elif len(contexts_ids) < 10:
    #             self.log_dict({f"loss/{tag}_c_5_to_10": loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #         elif len(contexts_ids) < 15:
    #             self.log_dict({f"loss/{tag}_c_10_to_15": loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #         elif len(contexts_ids) < 20:
    #             self.log_dict({f"loss/{tag}_c_15_to_20": loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #         else:
    #             self.log_dict({f"loss/{tag}_c_20_plus": loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    #     if self.update_memory_during_training:
    #         if delta_memory is not None:
    #             with torch.no_grad():
    #                 if self.cache_data_for_longer_context:
    #                     if cached_contexts_indicators_initialized:
    #                         self.model.update_memory_with_delta_memory(delta_memory)
    #                     else:
    #                         self.cached_contexts_indicators = self.model.update_memory_with_delta_memory(delta_memory, self.cached_contexts_indicators)
    #                 else:
    #                     self.model.update_memory_with_delta_memory(delta_memory)

    #     self.log_dict({"loss": loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    #     # log some long-term memory statistics

    #     if self.trainer.global_step % 50 == 0:
    #         # log per layer
    #         for idx in range(self.model.L):
    #             # log len(self.model.ltm[idx])
    #             self.log_dict({'ltm_per_layer/size_{}'.format(idx): len(self.model.ltm[idx])}, logger=True, on_step=True)
    #             self.log_dict({'ltm_per_layer/total_rf_{}'.format(idx): np.sum(self.model.ltm_recall_frequencies[idx])}, logger=True, on_step=True)
    #     else:
    #         np.mean([len(self.model.ltm[idx]) for idx in range(self.model.L)])
    #         self.log_dict({'ltm/size': np.mean([len(self.model.ltm[idx]) for idx in range(self.model.L)])}, logger=True, on_step=True)
    #         self.log_dict({'ltm/total_rf': np.mean([np.sum(self.model.ltm_recall_frequencies[idx]) for idx in range(self.model.L)])}, logger=True, on_step=True)

    #     return loss

    def penalty_on_retriever_weights(self, retriever_weights):
        # retriever_weights: [bsz, klen]
        # make it far away from 0.5
        # return - self.retriever_penalty_weight * F.mse_loss(retriever_weights, torch.ones_like(retriever_weights) * 0.5)

        # penalty w(1-w)
        penalty_1 = (retriever_weights * (1 - retriever_weights)).mean()
        # |w|
        penalty_2 = retriever_weights.abs().mean()

        return penalty_1 + penalty_2


    def get_selector_loss(self, retriever_weights, retriever_weights_labels, label):

        if self.train_retriever_on_important_tokens:
            retriever_weights_labels = retriever_weights_labels[:, self.model.importance_indicators]
            retriever_weights = retriever_weights[:, torch.where(self.model.importance_indicators)[0]]

        if self.selector_loss_type == 'bce':
            positive_indices = torch.where(retriever_weights_labels > 0.5)
            negative_indices = torch.where(retriever_weights_labels < 0.5)
            if self.selector_loss_per_token:
                retriever_weights = retriever_weights.transpose(1, 2)
            if len(positive_indices[0]) == 0:
                positive_loss = 0
            else:
                positive_loss = ( - torch.log(retriever_weights[positive_indices] + 1e-5)).mean() * self.bce_loss_weight
            
            if len(negative_indices[0]) == 0:
                negative_loss = 0
            else:
                negative_loss = ( - torch.log(1 - retriever_weights[negative_indices] + 1e-5)).mean() * self.bce_loss_weight

        elif self.selector_loss_type == 'mse':
            if self.selector_loss_per_token:
                raise NotImplementedError("selector_loss_per_token is not implemented for mse loss")
            positive_loss = F.mse_loss(retriever_weights[torch.where(retriever_weights_labels > 0.5)], retriever_weights_labels[torch.where(retriever_weights_labels > 0.5)])
            negative_loss = F.mse_loss(retriever_weights[torch.where(retriever_weights_labels < 0.5)], retriever_weights_labels[torch.where(retriever_weights_labels < 0.5)])
        else:
            raise ValueError("Invalid selector loss type")
        
        if self.selector_loss_type in ['bce', 'mse']:
            loss = (positive_loss + negative_loss * self.negative_loss_weight) / (1 + self.negative_loss_weight)
            self.log_dict({f"selector/{label}negative_loss": negative_loss}, logger=True, on_step=True)
            self.log_dict({f"selector/{label}positive_loss": positive_loss}, logger=True, on_step=True)
        
        self.log_dict({f"selector/{label}loss": loss}, logger=True, on_step=True)

        return loss

    def get_positive_and_negative_loss(self, skip_injection, delta_memory, retriever_weights, is_encoder_loss=False, retriever_weights_labels=None):

        retriever_weights = torch.stack(list(retriever_weights)).squeeze(1)

        label = 'encoder_' if is_encoder_loss else ''
        if skip_injection: # this means indicator==1
            # retriever_weights: [bsz, klen]
            retriever_weights_labels = self.cached_contexts_indicators.to(retriever_weights.dtype).to(retriever_weights.device)
            if self.selector_layers is not None:
                retriever_weights_labels = retriever_weights_labels[self.selector_layers]
            label += 'uc_'

        else:
            if retriever_weights_labels is None:
                assert delta_memory is not None
                retriever_weights_labels = torch.zeros([self.model.memory.shape[0], self.model.memory.shape[1]], dtype=retriever_weights.dtype, device=retriever_weights.device)
                retriever_weights_labels[:, -delta_memory.shape[2]:] += 1
            label += ''

        loss = self.get_selector_loss(retriever_weights, retriever_weights_labels, label)

         # calculate f1 score
        if self.selector_loss_per_token:
            # repeat retriever_weights_labels to match the shape of retriever_weights
            # retriever_weights_labels = torch.stack([retriever_weights_labels] * retriever_weights.shape[1], dim=1)
            retriever_weights = retriever_weights.mean(dim=1)

        if self.trainer.global_step % self.log_selector_acc_interval == 0:

            # save the acc per layer:
            for idx in range(retriever_weights.shape[0]):

                if retriever_weights_labels[idx].sum() == 0:
                    f1 = 0
                else:
                    f1 = metrics.f1_score(retriever_weights_labels[idx].detach().to(torch.float16).reshape(-1).cpu().numpy(), (retriever_weights[idx].detach().reshape(-1).to(torch.float16).cpu().numpy() > 0.5))
                
                if self.selector_layers is None:
                    self.log_dict({f"selector_per_layer/{label}f1_{idx}": f1}, logger=True, on_step=True)
                else:
                    self.log_dict({f"selector_per_layer/{label}f1_{self.selector_layers[idx]}": f1}, logger=True, on_step=True)
        
        if retriever_weights_labels.sum() > 0:
            f1_all = metrics.f1_score(retriever_weights_labels.to(torch.float16).reshape(-1).detach().cpu().numpy(), (retriever_weights.to(torch.float16).detach() > 0.5).reshape(-1).cpu().numpy())
        else:
            f1_all = 0
        self.log_dict({f"selector/{label}f1_all": f1_all}, logger=True, on_step=True)
        return loss 
    
    def schedule_contexts_and_sentences(self, 
                                        contexts_ids, 
                                        sentence_ids, 
                                        labels,
                                        cat_to_maximum_memory,
                                        sentence_start_idx=None):

        with torch.no_grad():

            sentence_labels = None

            if self.num_contexts_schedule is not None:

                if not cat_to_maximum_memory:
                    num_of_contexts = 1
                else:
                    num_of_contexts = self.num_contexts(self.trainer.global_step)

                if np.random.random() < self.random_sample_length_ratio:
                    num_of_contexts = min(np.random.randint(1, num_of_contexts+1), len(contexts_ids))
                else:
                    num_of_contexts = min(num_of_contexts, len(contexts_ids))

                if labels[0] == 0:
                    # cut the end of the contexts    
                    contexts_ids = contexts_ids[:num_of_contexts]
                elif labels[0] == 1:
                    # cut the beginning of the contexts
                    contexts_ids = contexts_ids[-num_of_contexts:]
                elif labels[0] == 2:
                    # it means the sentence is the second part of the contexts
                    if num_of_contexts == 1 and len(contexts_ids) > 1:
                        contexts_ids = contexts_ids[1:][:num_of_contexts]
                    else:
                        contexts_ids = contexts_ids[:num_of_contexts]
                elif labels[0] == 3:
                    contexts_ids = contexts_ids[:num_of_contexts]

                elif labels[0] == 4:

                    # target is context during completion task training
                    contexts_ids = contexts_ids[:num_of_contexts]

                    sentence_ids = torch.cat(contexts_ids, dim=1)[:, :self.max_seq_length]

                    if self.instruct is not None:
                        sentence = self.instruct.strip() + " " + self.tokenizer.decode(sentence_ids[0], skip_special_tokens=True)
                        sentence_ids = self.tokenizer(sentence, 
                                                    add_special_tokens=False, 
                                                    return_tensors='pt',
                                                    truncation=True,
                                                    max_length=self.max_seq_length)['input_ids'].to(sentence_ids.device)
                        sentence_labels = sentence_ids.clone()
                        if self.mask_instruction_tokens:
                            sentence_labels[:, :len(self.tokenizer(self.instruct.strip(),
                                                                    add_special_tokens=False).input_ids) + self.num_of_additional_tokens_to_mask] = -100
                    else:
                        sentence_labels = sentence_ids.clone()

                else:
                    raise ValueError("Invalid label")
            
            if sentence_labels is None:
                sentence_labels = sentence_ids.clone()

        contexts_ids = [x.detach() for x in contexts_ids]
        sentence_ids = sentence_ids.detach()
        sentence_labels = sentence_labels.detach()

        return contexts_ids, sentence_ids, sentence_labels

    def log_loss(self, labels, cat_to_maximum_memory, num_of_contexts, loss, random_position=False):

        random_position_str = '_random' if random_position else ''

        if labels[0] == 3 or labels[0] == 4:
            if cat_to_maximum_memory:
                if num_of_contexts < 5:
                    self.log_dict({'repeat/c_with_memory_{}_{}'.format(random_position_str, num_of_contexts): loss}, logger=True, on_step=True)
                elif num_of_contexts < 10:
                    self.log_dict({'repeat/c_with_memory_5_to_10_{}'.format(random_position_str): loss}, logger=True, on_step=True)
                elif num_of_contexts < 15:
                    self.log_dict({'repeat/c_with_memory_10_to_15_{}'.format(random_position_str): loss}, logger=True, on_step=True)
                elif num_of_contexts < 20:
                    self.log_dict({'repeat/c_with_memory_15_to_20_{}'.format(random_position_str): loss}, logger=True, on_step=True)
                else:
                    self.log_dict({'repeat/c_with_memory_20_plus_{}'.format(random_position_str): loss}, logger=True, on_step=True)

            else:
                self.log_dict({'repeat/c_{}'.format(num_of_contexts): loss}, logger=True, on_step=True)
        
        else:
            # if num_of_contexts == 1:
            
            if cat_to_maximum_memory:
                if num_of_contexts < 5:
                    self.log_dict({'loss/related_c_with_memory_{}_{}'.format(random_position_str, num_of_contexts): loss}, logger=True, on_step=True)
                elif num_of_contexts < 10:
                    self.log_dict({'loss/related_c_with_memory_5_to_10_{}'.format(random_position_str): loss}, logger=True, on_step=True)
                elif num_of_contexts < 15:
                    self.log_dict({'loss/related_c_with_memory_10_to_15_{}'.format(random_position_str): loss}, logger=True, on_step=True)
                elif num_of_contexts < 20:
                    self.log_dict({'loss/related_c_with_memory_15_to_20_{}'.format(random_position_str): loss}, logger=True, on_step=True)
                else:
                    self.log_dict({'loss/related_c_with_memory_20_plus_{}'.format(random_position_str): loss}, logger=True, on_step=True)
                    
            else:
                self.log_dict({'loss/related_c_{}'.format(num_of_contexts): loss}, logger=True, on_step=True)

    def debug_recording_time(self, contexts_ids):

        if len(contexts_ids) > 20:
            # time 1:
            # start_time = time.time()
            # all_delta_memory = self.parallel_process_contexts(contexts_ids)
            # end_time = time.time()
            # print("Time for parallel_process_contexts: ", end_time - start_time)

            start_time = time.time()
            with torch.no_grad():
                delta_memory = None
                for context_idx, context_ids in enumerate(contexts_ids):
                    delta_memory = self.model(input_ids=context_ids,
                                output_delta_memory=True,
                                is_injection=True,
                                delta_memory=delta_memory,
                                training=True).delta_memory
            end_time = time.time()
            print("Time for sequential_process_contexts without gradient: ", end_time - start_time)

            start_time = time.time()
            delta_memory = None
            for context_idx, context_ids in enumerate(contexts_ids):
                delta_memory = self.model(input_ids=context_ids,
                            output_delta_memory=True,
                            is_injection=True,
                            delta_memory=delta_memory,
                            training=True).delta_memory.detach()
            end_time = time.time()
            print("Time for sequential_process_contexts with gradient: ", end_time - start_time)

            # start_time = time.time()
            # with torch.no_grad():
            #     for context_idx, context_ids in enumerate(contexts_ids):

            #         if self.parallel_injection:
            #             delta_memory = all_delta_memory[context_idx]
            #         else:
            #             output = self.model(input_ids=context_ids,
            #                                 output_delta_memory=True,
            #                                 delta_memory=delta_memory,
            #                                 is_injection=True,
            #                                 training=True)
            #             delta_memory = output.delta_memory.detach()

            #         with torch.no_grad():
            #             # get retriever weights
            #             all_retriever_weights = []
            #             for idx in range(self.model.memory.shape[0]):
            #                 delta_memory_queries = self.model.model.model.layers[idx].self_attn.encoder_query_proj(
            #                     self.model.model.model.layers[idx].input_layernorm(delta_memory[0, idx]))
            #                 if self.model.maintain_memory_keys:
            #                     memory_keys = self.model.memory_keys[idx]
            #                 else:
            #                     memory_keys = self.model.model.model.layers[idx].self_attn.key_proj(
            #                         self.model.model.model.layers[idx].input_layernorm(self.model.memory[idx]))
            #                 retriever_weights = (delta_memory_queries @ memory_keys.transpose(-2, -1)).sigmoid().mean(dim=0)
            #                 all_retriever_weights.append(retriever_weights)
            #             retriever_weights = torch.stack(all_retriever_weights)
            #         retriever_weights = retriever_weights.detach()

            #         if context_idx == 0:
            #             self.cached_contexts_indicators = self.model.update_memory_with_delta_memory(delta_memory, self.cached_contexts_indicators,
            #                                                                                         retriever_weights=retriever_weights)
            #             retriever_weights_labels = torch.ones([self.model.memory.shape[0], self.model.memory.shape[1]], dtype=delta_memory.dtype, device=delta_memory.device)
            #             retriever_weights_labels[:, - self.model.num_tokens:] = 0
                        
            #         else:
            #             contexts_indicators = self.model.update_memory_with_delta_memory(delta_memory, torch.stack([self.cached_contexts_indicators, retriever_weights_labels.to(self.cached_contexts_indicators.device)]),
            #                                                                                         retriever_weights=retriever_weights)
            #             self.cached_contexts_indicators = contexts_indicators[0]
            #             retriever_weights_labels = contexts_indicators[1]

            #     retriever_weights_labels = 1 - retriever_weights_labels
            #     delta_memory = None
            # end_time = time.time()
            # print("Time for eager_update_memory:", end_time - start_time)

            start_time = time.time()
            self.inject_contexts_ids_into_memory(contexts_ids, False)
            end_time = time.time()
            print("time for injecting contexts_ids:", end_time - start_time)

            import ipdb; ipdb.set_trace()

    def parallel_process_contexts(self, contexts_ids):

        with torch.no_grad():
            contexts_embeds = []
            contexts_lengths = []
            for context_ids in contexts_ids:
                context_embeds = self.model.model.model.embed_tokens(context_ids)
                if context_embeds.shape[1] < self.model.num_tokens:
                    context_embeds = torch.cat([
                        context_embeds,
                        self.model.memory_embeddings.unsqueeze(0).repeat(1, self.model.num_tokens - context_embeds.shape[1], 1)
                    ], dim=1)
                contexts_embeds.append(context_embeds)
                contexts_lengths.append(context_embeds.shape[1])
            all_embeds = torch.cat(contexts_embeds, dim=1)
            attention_mask = torch.ones(all_embeds.shape[1]+1, all_embeds.shape[1]+1, dtype=torch.bool).tril(diagonal=0)
            cur_length = 1
            for idx in range(1, len(contexts_lengths)):
                cur_length += contexts_lengths[idx-1]
                attention_mask[cur_length:cur_length+contexts_lengths[idx],1:cur_length] = False

            delta_memory = self.model(
                inputs_embeds=all_embeds,
                encoder_attention_mask=attention_mask,
                parallel_injection=True,
                output_delta_memory=True,
                is_injection=True,
                training=True).delta_memory
            
            all_delta_memory = []
            cur_length = 0
            for idx in range(len(contexts_lengths)):
                # all_delta_memory.append(delta_memory[:, :, cur_length:cur_length+contexts_lengths[idx]][:, :, -256:])
                all_delta_memory.append(delta_memory[:, :, cur_length+contexts_lengths[idx]-256:cur_length+contexts_lengths[idx]])
                cur_length += contexts_lengths[idx]
            
        return [x.detach() for x in all_delta_memory]

    def inject_contexts_ids_into_memory(self, contexts_ids, use_retriever_when_dropping=False):

        if self.parallel_injection:
            all_delta_memory = self.parallel_process_contexts(contexts_ids)

        if use_retriever_when_dropping:

            # we have to use eager update memory in this case
            assert self.add_selector

            delta_memory = None

            for context_idx, context_ids in enumerate(contexts_ids):
                if self.parallel_injection:
                    delta_memory = all_delta_memory[context_idx]
                else:
                    output = self.model(input_ids=context_ids,
                                        output_delta_memory=True,
                                        delta_memory=delta_memory,
                                        is_injection=True,
                                        training=True)
                    delta_memory = output.delta_memory.detach()

                with torch.no_grad():
                    # get retriever weights
                    all_retriever_weights = []
                    for idx in range(self.model.memory.shape[0]):
                        delta_memory_queries = self.model.model.model.layers[idx].self_attn.encoder_query_proj(
                            self.model.model.model.layers[idx].input_layernorm(delta_memory[0, idx]))
                        if self.model.maintain_memory_keys:
                            memory_keys = self.model.memory_keys[idx]
                        else:
                            memory_keys = self.model.model.model.layers[idx].self_attn.key_proj(
                                self.model.model.model.layers[idx].input_layernorm(self.model.memory[idx]))
                        retriever_weights = (delta_memory_queries @ memory_keys.transpose(-2, -1)).sigmoid().mean(dim=0)
                        all_retriever_weights.append(retriever_weights)
                    retriever_weights = torch.stack(all_retriever_weights)
                retriever_weights = retriever_weights.detach()

                if context_idx == 0:
                    self.cached_contexts_indicators = self.model.update_memory_with_delta_memory(delta_memory, self.cached_contexts_indicators,
                                                                                                retriever_weights=retriever_weights)
                    retriever_weights_labels = torch.ones([self.model.memory.shape[0], self.model.memory.shape[1]], dtype=delta_memory.dtype, device=delta_memory.device)
                    retriever_weights_labels[:, - self.model.num_tokens:] = 0
                    
                else:
                    contexts_indicators = self.model.update_memory_with_delta_memory(delta_memory, torch.stack([self.cached_contexts_indicators, retriever_weights_labels.to(self.cached_contexts_indicators.device)]),
                                                                                                retriever_weights=retriever_weights)
                    self.cached_contexts_indicators = contexts_indicators[0]
                    retriever_weights_labels = contexts_indicators[1]

            retriever_weights_labels = 1 - retriever_weights_labels
            return retriever_weights_labels

        elif self.eager_update_memory:

            for context_idx, context_ids in enumerate(contexts_ids):
                    
                if self.parallel_injection:
                    delta_memory = all_delta_memory[context_idx]
                else:
                    output = self.model(input_ids=context_ids,
                                        output_delta_memory=True,
                                        is_injection=True,
                                        delta_memory=delta_memory,
                                        training=True)
                    delta_memory = output.delta_memory.detach()

                if context_idx == 0:
                    self.cached_contexts_indicators = self.model.update_memory_with_delta_memory(delta_memory, self.cached_contexts_indicators)
                    retriever_weights_labels = torch.ones([self.model.memory.shape[0], self.model.memory.shape[1]], dtype=delta_memory.dtype, device=delta_memory.device)
                    retriever_weights_labels[:, - self.model.num_tokens:] = 0
                else:
                    if self.cached_contexts_indicators is None:
                        retriever_weights_labels = self.model.update_memory_with_delta_memory(delta_memory, retriever_weights_labels)
                    else:
                        contexts_indicators = self.model.update_memory_with_delta_memory(delta_memory, torch.stack([self.cached_contexts_indicators, retriever_weights_labels.to(self.cached_contexts_indicators.device)]))
                        self.cached_contexts_indicators = contexts_indicators[0]
                        retriever_weights_labels = contexts_indicators[1]

                retriever_weights_labels = 1 - retriever_weights_labels

        else:

            # here we need to have:
            # (1) all_delta_memory
            # (2) all_dropped_memory
            # (3) all_delta_memory_ages

            with torch.no_grad():

                if hasattr(self.model, "remaining_indicators"):

                    all_delta_memory = None
                    delta_memory = None

                    retriever_weights_labels = torch.zeros([self.model.memory.shape[0], self.model.memory.shape[1]])

                    # TODO: when there are more than maybe 100 contexts, we need to maintain the dropped_memory and dropped_memory_ages variable

                    for context_idx, context_ids in enumerate(contexts_ids):

                        output = self.model(input_ids=context_ids,
                                        output_delta_memory=True,
                                        delta_memory=delta_memory,
                                        is_injection=True,
                                        training=True)
                        delta_memory = output.delta_memory

                        if all_delta_memory is None:
                            all_delta_memory = delta_memory
                        else:
                            all_delta_memory = torch.cat([
                                all_delta_memory[:, :, self.model.remaining_indicators[-all_delta_memory.shape[2]:]],
                                delta_memory
                            ], dim=2)
                        
                        retriever_weights_labels = torch.cat([retriever_weights_labels[:, self.model.remaining_indicators],
                                                            torch.ones([self.model.memory.shape[0], self.model.num_tokens])], dim=1)
                    
                    delta_memory = all_delta_memory

                    self.cached_contexts_indicators = self.model.update_memory_with_delta_memory(
                        delta_memory,
                        self.cached_contexts_indicators
                    )
                
                else:

                    if self.efficient_get_delta_memory:

                        if hasattr(self.model, "virtual_num_blocks") and self.model.virtual_num_blocks is not None:
                            counts = get_num_tokens(self.model.num_tokens * self.model.virtual_num_blocks, self.model.num_tokens, len(contexts_ids))
                            total_num_tokens = sum(counts[1:])
                            if total_num_tokens > self.model.memory.shape[1]:
                                idx = 1
                                while sum(counts[-idx:]) < self.model.memory.shape[1]:
                                    idx += 1
                                counts[-idx] -= sum(counts[-idx:]) - self.model.memory.shape[1]
                                idx += 1
                                while idx < len(counts):
                                    counts[-idx] = 0
                                    idx += 1

                        else:
                            counts = get_num_tokens(self.model.memory.shape[1], self.model.num_tokens, len(contexts_ids))

                        delta_memory = None
                        all_delta_memory, all_delta_memory_ages, dropped_delta_memory, dropped_delta_memory_ages = [], [], [], []

                        with torch.no_grad():

                            for ctx_idx, ctx_ids in enumerate(contexts_ids):

                                delta_memory = self.model(
                                    ctx_ids,
                                    is_injection=True,
                                    output_delta_memory=True,
                                    delta_memory=delta_memory
                                ).delta_memory

                                if counts[ctx_idx+1] == 0:
                                    dropped_delta_memory.append(delta_memory)
                                    dropped_delta_memory_ages.append(torch.stack(
                                        [torch.arange(self.model.num_tokens).flip(dims=[0]) + self.model.num_tokens * (len(contexts_ids) - ctx_idx - 1) for _ in range(self.model.L)]
                                    ))

                                cur_delta_memory = []
                                cur_dropped_delta_memory = []
                                cur_memory_ages = []
                                cur_dropped_memory_ages = []

                                for layer_idx in range(delta_memory.shape[1]):

                                    ages = torch.arange(self.model.num_tokens).flip(dims=[0]) + self.model.num_tokens * (len(contexts_ids) - ctx_idx - 1)
                                    indices = torch.randperm(delta_memory.shape[2])
                                    indices_to_keep = indices[:counts[ctx_idx+1]]
                                    indices_to_drop = indices[counts[ctx_idx+1]:]
                                    indices_to_keep, indices_to_drop = torch.sort(indices_to_keep)[0], torch.sort(indices_to_drop)[0]
                                    cur_delta_memory.append(delta_memory[:, layer_idx, indices_to_keep])
                                    cur_dropped_delta_memory.append(delta_memory[:, layer_idx, indices_to_drop])
                                    cur_memory_ages.append(ages[indices_to_keep])
                                    cur_dropped_memory_ages.append(ages[indices_to_drop])

                                all_delta_memory.append(torch.stack(cur_delta_memory, dim=1))
                                all_delta_memory_ages.append(torch.stack(cur_memory_ages))
                                dropped_delta_memory.append(torch.stack(cur_dropped_delta_memory, dim=1))
                                dropped_delta_memory_ages.append(torch.stack(cur_dropped_memory_ages))
                            
                            all_delta_memory = torch.cat(all_delta_memory, dim=2)
                            dropped_delta_memory = torch.cat(dropped_delta_memory, dim=2)
                            all_delta_memory_ages = np.array(torch.cat(all_delta_memory_ages, dim=1))
                            dropped_delta_memory_ages = np.array(torch.cat(dropped_delta_memory_ages, dim=1))

                    else:

                        delta_memory = None
                        # initialize dropped_delta_memory and dropped_delta_memory_ages
                        dropped_delta_memory, dropped_delta_memory_ages = None, None

                        for context_idx, context_ids in enumerate(contexts_ids):

                            output = self.model(input_ids=context_ids,
                                            output_delta_memory=True,
                                            delta_memory=delta_memory,
                                            is_injection=True,
                                            training=True)
                            delta_memory = output.delta_memory

                            if context_idx == 0:
                                # initialize all_delta_memory and all_delta_memory_ages
                                all_delta_memory = delta_memory
                                # all_delta_memory_ages = np.zeros([delta_memory.shape[1], delta_memory.shape[2]])
                                all_delta_memory_ages = np.stack([np.arange(delta_memory.shape[2])[::-1] for _ in range(delta_memory.shape[1])])

                            else:

                                new_memory, new_ages, dropped_memory, dropped_ages = [], [], [], []

                                for idx in range(self.model.L):
                                    # drop memory
                                    indices = torch.randperm(all_delta_memory.shape[2])

                                    if hasattr(self.model, "virtual_num_blocks") and self.model.virtual_num_blocks is not None:
                                        indices_to_keep = indices[:all_delta_memory.shape[2] - int(all_delta_memory.shape[2] * (1 / self.model.virtual_num_blocks))]
                                        indices_to_drop = indices[len(indices_to_keep):]
                                        indices_to_keep, indices_to_drop = torch.sort(indices_to_keep)[0], torch.sort(indices_to_drop)[0]

                                    else:
                                        indices_to_keep = indices[:all_delta_memory.shape[2] - int(all_delta_memory.shape[2] * (1 / self.model.num_blocks))]
                                        indices_to_drop = indices[len(indices_to_keep):]
                                        indices_to_keep, indices_to_drop = torch.sort(indices_to_keep)[0], torch.sort(indices_to_drop)[0]

                                    new_memory.append(all_delta_memory[:, idx, indices_to_keep])
                                    dropped_memory.append(all_delta_memory[:, idx, indices_to_drop])
                                    new_ages.append(all_delta_memory_ages[idx][np.array(indices_to_keep)])
                                    dropped_ages.append(all_delta_memory_ages[idx][np.array(indices_to_drop)])
                                
                                all_delta_memory = torch.cat([torch.stack(new_memory, dim=1), delta_memory], dim=2)
                                all_delta_memory_ages = np.concatenate([np.stack(new_ages) + self.model.num_tokens, np.stack([np.arange(delta_memory.shape[2])[::-1] for _ in range(delta_memory.shape[1])])], axis=1)
                                
                                if dropped_delta_memory is None:
                                    dropped_delta_memory = torch.stack(dropped_memory, dim=1)
                                    dropped_delta_memory_ages = np.stack(dropped_ages) + self.model.num_tokens

                                else:
                                    dropped_delta_memory = torch.cat([dropped_delta_memory, torch.stack(dropped_memory, dim=1)], dim=2)
                                    dropped_delta_memory_ages = np.concatenate([dropped_delta_memory_ages, np.stack(dropped_ages)], axis=1) + self.model.num_tokens

                        if all_delta_memory.shape[2] > self.model.memory.shape[1]:

                            dropped_delta_memory = torch.cat([
                                dropped_delta_memory,
                                all_delta_memory[:, :, :(all_delta_memory.shape[2] - self.model.memory.shape[1])],
                            ], dim=2)
                            dropped_ages = np.concatenate([
                                dropped_delta_memory_ages,
                                all_delta_memory_ages[:, :(all_delta_memory.shape[2] - self.model.memory.shape[1])],
                            ], axis=1)
                            all_delta_memory = all_delta_memory[:, :, (all_delta_memory.shape[2] - self.model.memory.shape[1]):]
                            all_delta_memory_ages = all_delta_memory_ages[:, (all_delta_memory.shape[2] - self.model.memory.shape[1]):]

                    retriever_weights_labels = torch.ones([self.model.memory.shape[0], self.model.memory.shape[1]], dtype=delta_memory.dtype, device=delta_memory.device)

                    if self.cached_contexts_indicators is None:
                        retriever_weights_labels = self.model.update_memory_with_delta_memory(
                            all_delta_memory,
                            retriever_weights_labels, 
                            delta_memory_ages=all_delta_memory_ages, 
                            dropped_delta_memory=dropped_delta_memory,
                            dropped_delta_memory_ages=dropped_delta_memory_ages)
                    else:
                        contexts_indicators = self.model.update_memory_with_delta_memory(
                            all_delta_memory, 
                            torch.stack([self.cached_contexts_indicators, retriever_weights_labels.to(self.cached_contexts_indicators.device)]),
                            delta_memory_ages=all_delta_memory_ages, 
                            dropped_delta_memory=dropped_delta_memory,
                            dropped_delta_memory_ages=dropped_delta_memory_ages
                        )
                        self.cached_contexts_indicators = contexts_indicators[0]
                        retriever_weights_labels = contexts_indicators[1]
                    
                    retriever_weights_labels = 1 - retriever_weights_labels

            return retriever_weights_labels

    def convert_contexts_ids_into_delta_memory(self, contexts_ids):

        if len(contexts_ids) > 1:

            if self.parallel_injection:
                        
                if self.keep_gradient_for_the_last_step:
                    raise NotImplementedError("keep_gradient_for_the_last_step is not implemented for parallel_injection")
                
                # every 25 contexts for one time
                parallel_chunk_size = self.parallel_chunk_size if self.parallel_chunk_size is not None else self.model.num_blocks // 2
                all_delta_memory = []
                for contexts_ids_chunk in [contexts_ids[i:i+parallel_chunk_size] for i in range(0, len(contexts_ids), parallel_chunk_size)]:
                    all_delta_memory.extend(self.parallel_process_contexts(contexts_ids_chunk))

                with torch.no_grad():
                    final_delta_memory = None
                    for delta_memory in all_delta_memory:
                        if final_delta_memory is None:
                            final_delta_memory = delta_memory
                        else:
                            if hasattr(self.model, "remaining_indicators"):
                                final_delta_memory = torch.cat([
                                    final_delta_memory[:, :, self.model.remaining_indicators[-final_delta_memory.shape[2]:]],
                                    delta_memory
                                ], dim=2)
                            elif self.model.drop_memory_per_layer:
                                new_all_delta_memory = []
                                for idx in range(final_delta_memory.shape[1]):
                                    new_all_delta_memory.append(torch.cat([
                                        self.model.drop_memory(final_delta_memory[0, idx], 
                                            drop_length=int(final_delta_memory[0, idx].shape[0] * (delta_memory[0, idx].shape[0] / (self.model.num_blocks * self.model.num_tokens))), 
                                            unsequeezed=False),
                                        delta_memory[0, idx]
                                    ], dim=0))
                                final_delta_memory = torch.stack(new_all_delta_memory).unsqueeze(0) 
                            else:
                                raise NotImplementedError

                delta_emory = final_delta_memory.detach()
                torch.cuda.empty_cache()

            else:

                if self.efficient_get_delta_memory:
                    
                    if hasattr(self.model, "virtual_num_blocks") and self.model.virtual_num_blocks is not None:
                        counts = get_num_tokens(self.model.num_tokens * self.model.virtual_num_blocks, self.model.num_tokens, len(contexts_ids))
                        total_num_tokens = sum(counts[1:])

                        if total_num_tokens > self.model.memory.shape[1]:
                            idx = 1
                            while sum(counts[-idx:]) < self.model.memory.shape[1]:
                                idx += 1
                            counts[-idx] -= sum(counts[-idx:]) - self.model.memory.shape[1]
                            idx += 1
                            while idx < len(counts):
                                counts[-idx] = 0
                                idx += 1

                    else:
                        counts = get_num_tokens(self.model.memory.shape[1], self.model.num_tokens, len(contexts_ids))

                    delta_memory = None
                    all_delta_memory = None

                    with torch.no_grad():
                        for ctx_idx, ctx_ids in enumerate(contexts_ids):

                            if counts[ctx_idx + 1] == 0:
                                continue

                            delta_memory = self.model(
                                ctx_ids,
                                is_injection=True,
                                output_delta_memory=True,
                                delta_memory=delta_memory,
                            ).delta_memory

                            cur_delta_memory = []
                            for layer_idx in range(delta_memory.shape[1]):
                                random_indices = np.sort(np.random.choice(np.arange(self.model.num_tokens), size=counts[ctx_idx + 1], replace=False))
                                cur_delta_memory.append(
                                    delta_memory[:, layer_idx, torch.tensor(random_indices)]
                                )
                            if all_delta_memory is None:
                                all_delta_memory = torch.stack(cur_delta_memory, dim=1)
                            else:
                                all_delta_memory = torch.cat([all_delta_memory, torch.stack(cur_delta_memory, dim=1)], dim=2)

                    delta_memory = all_delta_memory.detach()  
                    return delta_memory 

                if hasattr(self.model, "virtual_num_blocks") and self.model.virtual_num_blocks is not None:
                    raise NotImplementedError("virtual_num_blocks is not implemented for vanilla injection")

                all_delta_memory = None
                delta_memory = None
                # maintain the age of these delta_memories

                context_indices = np.arange(len(contexts_ids))

                for i in context_indices:

                    output = self.model(input_ids=contexts_ids[i],
                                        output_delta_memory=True,
                                        delta_memory=delta_memory,
                                        is_injection=True,
                                        training=True)

                    if i == len(context_indices) - 1 and self.keep_gradient_for_the_last_step:
                        delta_memory = output.delta_memory
                    else:
                        delta_memory = output.delta_memory.detach()

                    if all_delta_memory is None:
                        all_delta_memory = delta_memory
                        if hasattr(self.model, 'ltm'):
                            delta_memory_ages = np.zeros([delta_memory.shape[1], delta_memory.shape[2]])

                    else:
                        
                        if hasattr(self.model, "remaining_indicators"):
                            all_delta_memory = torch.cat([
                                all_delta_memory[:, :, self.model.remaining_indicators[-all_delta_memory.shape[2]:]],
                                delta_memory
                            ], dim=2)
                            # TODO: add delta_memory_ages update

                        elif self.model.drop_memory_per_layer:
                            new_all_delta_memory = []
                            if hasattr(self.model, 'ltm'):
                                new_delta_memory_ages = []

                            for idx in range(all_delta_memory.shape[1]):
                                remaining_all_delta_memory, remaining_indices = self.model.drop_memory(all_delta_memory[0, idx], 
                                        drop_length=int(all_delta_memory[0, idx].shape[0] * (delta_memory[0, idx].shape[0] / (self.model.num_blocks * self.model.num_tokens))), 
                                        unsequeezed=False,
                                        return_remaining_indices=True)
                                new_all_delta_memory.append(torch.cat([
                                    remaining_all_delta_memory,
                                    delta_memory[0, idx]
                                ], dim=0))
                                if hasattr(self.model, 'ltm'):
                                    new_delta_memory_ages.append(
                                        np.concatenate([
                                            delta_memory_ages[idx][remaining_indices] + 1,
                                            np.zeros(delta_memory.shape[2])
                                        ])
                                    )
                            
                            all_delta_memory = torch.stack(new_all_delta_memory).unsqueeze(0) 
                            if hasattr(self.model, 'ltm'):
                                delta_memory_ages = np.stack(new_delta_memory_ages)

                        else:
                            all_delta_memory = self.model.drop_memory(all_delta_memory[0]).unsqueeze(0)
                            all_delta_memory = torch.cat([
                                all_delta_memory,
                                delta_memory
                            ], dim=2)

                delta_memory = all_delta_memory.detach()
                torch.cuda.empty_cache()

        else:

            output = self.model(input_ids=contexts_ids[-1],
                                output_delta_memory=True,
                                is_injection=True,
                                training=True)

            delta_memory = output.delta_memory
        
        return delta_memory

    def training_step_complete(self, contexts_ids, sentence_ids, labels):

        # 0. identify the index of the chunk to keep gradient:
        index = np.random.choice(np.arange(min(len(contexts_ids), self.num_contexts(self.trainer.global_step))))

        sentence_labels = sentence_ids.clone()

        # 2.1. identify the position we are going to insert the delta_memory at:
        # TODO: there might be errors if the length of the contexts_ids is less than the length of the final_list
        start_idx = np.random.choice(np.arange(len(self.model.final_list) - len(contexts_ids) + 1))

        # 2.2. convert all contexts_ids into delta_memory
        delta_memory = None

        all_delta_memory = []

        for context_idx, context_ids in enumerate(contexts_ids):
            
            if context_idx == index and self.keep_gradient_in_random_insertion:
                output = self.model(input_ids=context_ids,
                                    output_delta_memory=True,
                                    delta_memory=delta_memory,
                                    is_injection=True,
                                    training=True)
                delta_memory = output.delta_memory 
            
            else:
                if delta_memory is not None:
                    delta_memory = delta_memory.detach()

                with torch.no_grad():
                    output = self.model(input_ids=context_ids,
                                        output_delta_memory=True,
                                        delta_memory=delta_memory,
                                        is_injection=True,
                                        training=True)
                delta_memory = output.delta_memory.detach()
            
            if self.important_tokens == 'right':       
                all_delta_memory.append(delta_memory[:, :, -self.model.final_list[start_idx + context_idx]:])
            else:
                all_delta_memory.append(delta_memory[:, :, :self.model.final_list[start_idx + context_idx]])
        
        delta_memory = torch.cat(all_delta_memory, dim=2)

        output = self.model(input_ids=sentence_ids,
            labels=sentence_labels,
            delta_memory=delta_memory,
            output_delta_memory=False,
            delta_memory_position_idx=start_idx,
            is_injection=False,
            return_dict=True, 
            training=True,)

        loss = output.loss

        self.log_loss(labels, True, len(contexts_ids), loss, random_position=True)

        return loss


    def training_step_repeat(self, contexts_ids, sentence_ids, labels):

        # # randomly select four sequential chunks from contexts_ids
        # if len(contexts_ids) > 4:
        #     random_start_idx = np.random.choice(np.arange(len(contexts_ids) - 4 + 1))
        #     contexts_ids = contexts_ids[random_start_idx:random_start_idx+4]

        if len(contexts_ids) > len(self.model.final_list):
            random_start_idx = np.random.choice(np.arange(len(contexts_ids) - len(self.model.final_list) + 1))
            contexts_ids = contexts_ids[random_start_idx:random_start_idx+len(self.model.final_list)]

        # 0. identify the index of the chunk to keep gradient:
        index = np.random.choice(np.arange(min(len(contexts_ids), self.num_contexts(self.trainer.global_step))))

        # 1. shard the first 50 steps of the contexts
        contexts_ids, sentence_ids, sentence_labels = self.schedule_contexts_and_sentences(contexts_ids, sentence_ids, labels, 
                                                                                           cat_to_maximum_memory=True,
                                                                                           sentence_start_idx=index)

        # 2. convert all contexts_ids into delta_memory, where only one chunk of them has gradient

        # 2.1. identify the position we are going to insert the delta_memory at:
        # TODO: there might be errors if the length of the contexts_ids is less than the length of the final_list
        start_idx = np.random.choice(np.arange(len(self.model.final_list) - len(contexts_ids) + 1))

        # 2.2. convert all contexts_ids into delta_memory
        delta_memory = None

        all_delta_memory = []

        for context_idx, context_ids in enumerate(contexts_ids):

            if context_idx == index and self.keep_gradient_in_random_insertion:
                output = self.model(input_ids=context_ids,
                                    output_delta_memory=True,
                                    delta_memory=delta_memory,
                                    is_injection=True,
                                    training=True)
                delta_memory = output.delta_memory   
            
            else:
                if delta_memory is not None:
                    delta_memory = delta_memory.detach()

                with torch.no_grad():
                    output = self.model(input_ids=context_ids,
                                        output_delta_memory=True,
                                        delta_memory=delta_memory,
                                        is_injection=True,
                                        training=True)
                delta_memory = output.delta_memory.detach()
            
            if self.important_tokens == 'right':       
                all_delta_memory.append(delta_memory[:, :, -self.model.final_list[start_idx + context_idx]:])
            else:
                all_delta_memory.append(delta_memory[:, :, :self.model.final_list[start_idx + context_idx]])
        
        delta_memory = torch.cat(all_delta_memory, dim=2)

        output = self.model(input_ids=sentence_ids,
            labels=sentence_labels,
            delta_memory=delta_memory,
            output_delta_memory=False,
            delta_memory_position_idx=start_idx,
            is_injection=False,
            return_dict=True, 
            training=True,)

        loss = output.loss

        self.log_loss(labels, True, len(contexts_ids), loss, random_position=True)

        return loss

    def training_step(self, batch, batch_idx):

        if len(batch) == 3:
            contexts_ids, sentence_ids, labels = batch
        else:
            contexts_ids, sentence_ids, sentence_labels, labels = batch

        if len(contexts_ids) == 0:
            output = self.model(input_ids=sentence_ids,
                labels=sentence_labels,
                output_delta_memory=False,
                is_injection=False,
                return_dict=True, 
                training=True)
            loss = output.loss
            self.log_dict({"loss_ift_no_context": loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss

        sentence_labels = None
        skip_injection = False

        if np.random.random() < self.full_context_and_sentence_training_ratio or labels[0] == 5:
            if labels[0] == 1:
                sentence_ids = torch.cat(contexts_ids + [sentence_ids], dim=1)[:, :self.max_seq_length_when_detaching_memory]
            output = self.model(input_ids=sentence_ids,
                labels=sentence_ids.clone(),
                output_delta_memory=False,
                is_injection=False,
                return_dict=True,
                training=True)
            loss = output['loss']
            self.log_dict({'length_full': sentence_ids.shape[1]}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict({"loss_full": loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss

        if labels[0] == 4 and np.random.random() < self.train_with_random_memory_insertion_ratio and len(contexts_ids) < self.random_insertion_max_length:
            return self.training_step_repeat(contexts_ids, sentence_ids, labels)
        elif labels[0] == 1 and np.random.random() < self.complete_with_random_memory_insertion_ratio and len(contexts_ids) < self.random_insertion_max_length:
            return self.training_step_complete(contexts_ids, sentence_ids, labels)

        if self.cache_data_for_longer_context and self.cache_data_for_longer_context_from <= self.trainer.global_step:
            
            if self.last_indicator == 1 or self.nuc_length == 0:
                indicator = 0
            else:
                # randomly choose between (0, 1)
                # indicator = random.randint(0, 1)
                if np.random.random() < self.pass_ratio:
                    indicator = 0
                else:
                    indicator = 1
            
            if indicator == 1:
                if self.put_cache_on_cpu:
                    sentence_ids = self.cached_sentence_ids.to(sentence_ids.device)
                    sentence_labels = self.cached_sentence_labels.to(sentence_ids.device)
                else:
                    sentence_ids = self.cached_sentence_ids
                    sentence_labels = self.cached_sentence_labels
                skip_injection = True

        cat_to_maximum_memory = self.cat_to_maximum_memory or (self.occassionally_cat_to_maximum_memory_ratio is not None and np.random.random() < self.occassionally_cat_to_maximum_memory_ratio)
        use_retriever_when_dropping = (self.use_retriever_when_dropping_from is not None and self.trainer.global_step > self.use_retriever_when_dropping_from)

        retriever_weights_labels = None

        if not skip_injection:

            contexts_ids, sentence_ids, sentence_labels = self.schedule_contexts_and_sentences(contexts_ids, sentence_ids, labels, cat_to_maximum_memory)

            delta_memory = None

            if hasattr(self.model, 'ltm') and cat_to_maximum_memory:
                retriever_weights_labels = self.inject_contexts_ids_into_memory(contexts_ids, use_retriever_when_dropping)

            else:
                if (self.eager_update_memory or use_retriever_when_dropping) and cat_to_maximum_memory:
                    raise NotImplementedError
                else:
                    delta_memory = self.convert_contexts_ids_into_delta_memory(contexts_ids)

        additional_kwargs = {}

        if not skip_injection:

            tmp_delta_memory = delta_memory
            # if not (self.eager_update_memory or use_retriever_when_dropping):
            if len(contexts_ids) == 1 and cat_to_maximum_memory and delta_memory is not None:
                if not self.keep_gradient_for_the_last_step:
                    delta_memory = delta_memory.detach()
            elif len(contexts_ids) == 1 and not cat_to_maximum_memory:
                if hasattr(self.model, "unique_memory_numbers") and self.random_length_for_encoder:
                    length = np.random.choice(self.model.unique_memory_numbers)
                    if self.important_tokens == 'right':
                        tmp_delta_memory = delta_memory[:, :, -length:]
                    else:
                        tmp_delta_memory = delta_memory[:, :, :length]

            additional_kwargs['cat_to_maximum_memory'] = cat_to_maximum_memory

        else:
            delta_memory = None
            tmp_delta_memory = None
        
        if self.add_selector and cat_to_maximum_memory:

            additional_kwargs['output_retriever_weights'] = True

            if self.selector_loss_per_token:
                additional_kwargs['return_full_retriever_weights'] = True
            
            if self.random_retriever_length:
                additional_kwargs['random_retriever_length'] = True
            
            if self.add_encoder_retriever:
                if not skip_injection:
                    encoder_query_indices = torch.zeros([self.model.memory.shape[0], self.model.memory.shape[1]], dtype=sentence_ids.dtype, device=sentence_ids.device)
                    encoder_query_indices[:, -self.model.num_tokens:] += 1
                    additional_kwargs['encoder_query_indices'] = encoder_query_indices

        if self.train_retriever_on_important_tokens:
            additional_kwargs['memory_key_indicators'] = self.model.importance_indicators

        if self.scale_randomized_poe_ratio is not None:
            if self.randomized_poe_ratio is not None:
                if np.random.random() < self.randomized_poe_ratio:
                    additional_kwargs['scale_randomized_poe_ratio'] = self.scale_randomized_poe_ratio
                else:
                    additional_kwargs['scale_randomized_poe_ratio'] = None
            else:
                additional_kwargs['scale_randomized_poe_ratio'] = self.scale_randomized_poe_ratio
        
        if self.scale_randomized_poe_ratio_for_memory is not None:
            if self.randomized_poe_for_memory_ratio is not None:
                if np.random.random() < self.randomized_poe_for_memory_ratio:
                    additional_kwargs['scale_randomized_poe_ratio_for_memory'] = self.scale_randomized_poe_ratio_for_memory
                else:
                    additional_kwargs['scale_randomized_poe_ratio_for_memory'] = None
            additional_kwargs['scale_randomized_poe_ratio_for_memory'] = self.scale_randomized_poe_ratio_for_memory

        output = self.model(input_ids=sentence_ids,
            labels=sentence_labels,
            delta_memory=tmp_delta_memory,
            output_delta_memory=False,
            is_injection=False,
            return_dict=True, 
            training=True,
            **additional_kwargs)
        
        loss = output['loss']

        self.log_dict({'loss/avg_context_length': np.mean([x.shape[1] for x in contexts_ids])}, logger=True, on_step=True)
        self.log_dict({'loss/num_of_contexts': len(contexts_ids)}, logger=True, on_step=True)
        self.log_dict({'loss/sentence_length': sentence_ids.shape[1]}, logger=True, on_step=True)

        if self.add_selector and cat_to_maximum_memory:

            retriever_weights = output.retriever_weights

            if retriever_weights is not None:

                positive_negative_loss = self.get_positive_and_negative_loss(skip_injection, delta_memory, retriever_weights, retriever_weights_labels=retriever_weights_labels)
                if output.encoder_retriever_weights is not None:
                    positive_negative_loss += self.get_positive_and_negative_loss(skip_injection, delta_memory, output.encoder_retriever_weights, is_encoder_loss=True, retriever_weights_labels=retriever_weights_labels)

        if self.cache_data_for_longer_context:

            cached_contexts_indicators_initialized = False

            if indicator == 1:

                if self.cached_exp_label == 4:
                    self.log_dict({'repeat/unrelated': loss}, logger=True, on_step=True)
                    self.log_dict({'repeat/nuc': self.nuc_length}, logger=True, on_step=True)
                else:
                    self.log_dict({'loss/unrelated': (loss)}, logger=True, on_step=True)
                    self.log_dict({'loss/unrelated_nuc': self.nuc_length}, logger=True, on_step=True)

                # calculate the recall to evaluate the retrieval success rate of ltm
                if hasattr(self.model, 'ltm'):
                    # 1. get the age of every token in ltm
                    # 2. find the tokens between the age that satisfies the following condition:
                        #  < (sef.nuc_length + self.cached_contexts_length) and >= (self.nuc_length)
                    # 3. Check how many tokens between them are extracted from ltm

                    ltm_indices = output.ltm_indices
                    avg_precision = []
                    avg_recall = []
                    avg_normalized_precision = []
                    avg_normalized_recall = []
                    for idx in range(self.model.L):
                        if hasattr(self.model, "update_ltm_mode") and self.model.update_ltm_mode == 'immediate':
                            ground_truth_indices = np.where((self.model.ltm_ages[idx] < (self.nuc_length + self.cached_contexts_length) * self.model.num_tokens) & (self.model.ltm_ages[idx] >= self.nuc_length * self.model.num_tokens))[0]
                        else:
                            ground_truth_indices = np.where((self.model.ltm_ages[idx] < (self.nuc_length + self.cached_contexts_length)) & (self.model.ltm_ages[idx] >= self.nuc_length))[0]
                        
                        if len(ground_truth_indices) > 0 and len(ltm_indices[idx]) > 0:

                            # find the intersection between ltm_indices[idx] and ground_truth_indices:
                            intersection = np.intersect1d(ltm_indices[idx], ground_truth_indices)
                            precision = len(intersection) / len(ltm_indices[idx])
                            recall = len(intersection) / len(ground_truth_indices)

                            baseline_intersection = len(ground_truth_indices) * len(ltm_indices[idx]) / len(self.model.ltm[idx])
                            baseline_precision = baseline_intersection / len(ltm_indices[idx])
                            baseline_recall = baseline_intersection / len(ground_truth_indices)

                            self.log_dict({'ltm_recall/precision_{}'.format(idx): precision}, logger=True, on_step=True)
                            self.log_dict({'ltm_recall/recall_{}'.format(idx): recall}, logger=True, on_step=True)
                            self.log_dict({'ltm_recall/normalized_precision_{}'.format(idx): precision - baseline_precision}, logger=True, on_step=True)
                            self.log_dict({'ltm_recall/normalized_recall_{}'.format(idx): recall - baseline_recall}, logger=True, on_step=True)

                            avg_precision.append(precision)
                            avg_recall.append(recall)
                            avg_normalized_precision.append(precision - baseline_precision)
                            avg_normalized_recall.append(recall - baseline_recall)
                    
                    if len(avg_precision) > 0:
                        self.log_dict({'ltm/avg_precision': np.mean(avg_precision)}, logger=True, on_step=True)
                        self.log_dict({'ltm/avg_recall': np.mean(avg_recall)}, logger=True, on_step=True)
                        self.log_dict({'ltm/avg_normalized_precision': np.mean(avg_normalized_precision)}, logger=True, on_step=True)
                        self.log_dict({'ltm/avg_normalized_recall': np.mean(avg_normalized_recall)}, logger=True, on_step=True)

            else:

                if self.last_indicator == 1:
                    if self.cached_sentence_ids is None or np.random.random() < self.empty_prob:
                        if self.put_cache_on_cpu:
                            self.cached_sentence_ids = sentence_ids.cpu()
                            self.cached_sentence_labels = sentence_labels.cpu()
                            self.cached_exp_label = labels[0].cpu()
                        else:
                            self.cached_sentence_ids = sentence_ids
                            self.cached_sentence_labels = sentence_labels
                            self.cached_exp_label = labels[0]
                        self.cached_contexts_length = len(contexts_ids)
                        
                        self.nuc_length = 0
                        if self.add_selector:
                            if retriever_weights_labels is not None:
                                self.cached_contexts_indicators = retriever_weights_labels
                            else:
                                self.cached_contexts_indicators = torch.zeros([self.model.memory.shape[0], self.model.memory.shape[1]])
                                self.cached_contexts_indicators[:, -delta_memory.shape[2]:] = True
                                cached_contexts_indicators_initialized = True

                else:
                    self.nuc_length += len(contexts_ids)

                self.log_loss(labels, cat_to_maximum_memory, len(contexts_ids), loss)

            self.last_indicator = indicator

        else:

            warnings.warn("cache_data_for_longer_context not set to True, this is not in correct training mode")

        # else:
        #     # log the loss according to the number of contexts:
        #     if labels[0] == 3 or labels[0] == 4:
        #         if num_of_contexts == 1:
        #             if cat_to_maximum_memory:
        #                 self.log_dict({'repeat/c_with_memory_{}'.format(num_of_contexts): loss}, logger=True, on_step=True)
        #             else:
        #                 self.log_dict({'repeat/c_{}'.format(num_of_contexts): loss}, logger=True, on_step=True)
        #         else:
        #             self.log_dict({'repeat/c_{}'.format(num_of_contexts): loss}, logger=True, on_step=True)
        #     else:
        #         if num_of_contexts == 1:
        #             if cat_to_maximum_memory:
        #                 self.log_dict({'loss/related_c_with_memory_{}'.format(num_of_contexts): loss}, logger=True, on_step=True)
        #             else:
        #                 self.log_dict({'loss/{}_c_{}'.format("related" if (labels[0] ==1 or num_of_contexts == 1) else "unrelated", num_of_contexts): loss}, logger=True, on_step=True)
        #         else:
        #             self.log_dict({'loss/{}_c_{}'.format("related" if (labels[0] ==1 or num_of_contexts == 1) else "unrelated", num_of_contexts): loss}, logger=True, on_step=True)

        if self.add_selector and output.retriever_weights is not None:
            loss += positive_negative_loss

        if self.update_memory_during_training and delta_memory is not None:

            with torch.no_grad():

                if use_retriever_when_dropping:
                    # get retriever weights
                    all_retriever_weights = []
                    for idx in range(self.model.memory.shape[0]):
                        delta_memory_queries = self.model.model.model.layers[idx].self_attn.encoder_query_proj(
                            self.model.model.model.layers[idx].input_layernorm(delta_memory[0, idx]))
                        if self.model.maintain_memory_keys:
                            memory_keys = self.model.memory_keys[idx]
                        else:
                            memory_keys = self.model.model.model.layers[idx].self_attn.key_proj(
                                self.model.model.model.layers[idx].input_layernorm(self.model.memory[idx]))
                        retriever_weights = (delta_memory_queries @ memory_keys.transpose(-2, -1)).sigmoid().mean(dim=0)
                        all_retriever_weights.append(retriever_weights)
                    retriever_weights = torch.stack(all_retriever_weights)
                else:
                    retriever_weights = None

                if self.cache_data_for_longer_context and not cached_contexts_indicators_initialized:
                    self.cached_contexts_indicators = self.model.update_memory_with_delta_memory(delta_memory, 
                        self.cached_contexts_indicators, retriever_weights=retriever_weights)
                else:
                    self.model.update_memory_with_delta_memory(delta_memory, retriever_weights=retriever_weights)

        self.log_dict({"loss": loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if hasattr(self.model, 'ltm'):
            # log some long-term memory statistics
            if self.trainer.global_step % 50 == 0:
                # log per layer
                for idx in range(self.model.L):
                    # log len(self.model.ltm[idx])
                    self.log_dict({'ltm_per_layer/size_{}'.format(idx): float(len(self.model.ltm[idx]))}, logger=True, on_step=True)
                    if isinstance(self.model.ltm_recall_frequencies[idx], torch.Tensor):
                        self.log_dict({'ltm_per_layer/total_rf_{}'.format(idx): torch.sum(self.model.ltm_recall_frequencies[idx]).item()}, logger=True, on_step=True)
                    else:
                        self.log_dict({'ltm_per_layer/total_rf_{}'.format(idx): np.sum(self.model.ltm_recall_frequencies[idx])}, logger=True, on_step=True)
            
            np.mean([len(self.model.ltm[idx]) for idx in range(self.model.L)])
            self.log_dict({'ltm/size': np.mean([len(self.model.ltm[idx]) for idx in range(self.model.L)])}, logger=True, on_step=True)
            if isinstance(self.model.ltm_recall_frequencies[0], torch.Tensor):
                self.log_dict({'ltm/total_rf': np.mean([torch.sum(self.model.ltm_recall_frequencies[idx]).item() for idx in range(self.model.L)])}, logger=True, on_step=True)
            else:
                self.log_dict({'ltm/total_rf': np.mean([np.sum(self.model.ltm_recall_frequencies[idx]) for idx in range(self.model.L)])}, logger=True, on_step=True)

        
        if self.lr_scheduler is not None:
            sch = self.lr_schedulers()
            sch.step()

        return loss

    def short_validation_step(self, batch, batch_idx):
        
        loss_no_memory = None
        if self.detach_memory_ratio > 0:
            self.model.detach_memory()
            loss_no_memory = self.model(input_ids=batch, labels=batch.clone()).loss.item()
            self.model.attach_memory()
        
        loss = self.model(input_ids=batch, labels=batch.clone()).loss.item()
        
        if loss_no_memory is not None:

            self.validation_step_outputs[-1].append({
                'loss_with_memory': loss,
                'loss_no_memory': loss_no_memory
            })

        else:

            self.validation_step_outputs[-1].append({
                'loss_with_memory': loss
            })

    def longcontext_validation_step(self, batch, batch_idx):

        contexts_ids, sentence_ids, answers = batch

        if self.backup_memory is not None:
            self.model.memory.data = self.backup_memory.clone().to(self.model.memory.device)

        for context_ids in contexts_ids:
            if self.add_selector and self.use_retriever_during_validation:
                self.model.inject_memory(context_ids, 
                                     update_memory=True,
                                     use_retriever=True)
            else:
                self.model.inject_memory(context_ids, 
                                     update_memory=True)

        input_sentence_and_answer_ids = torch.cat([sentence_ids, self.tokenizer(" " + answers[0][0], return_tensors='pt', add_special_tokens=False).input_ids.to(sentence_ids.device)], dim=1)
        input_sentence_and_answer_labels = input_sentence_and_answer_ids.clone()
        input_sentence_and_answer_labels[:, :sentence_ids.shape[1]] = -100
        generation_loss = self.model(
            input_sentence_and_answer_ids, labels=input_sentence_and_answer_labels
        ).loss

        # output = self.model.generate(
        #         inputs=sentence_ids, 
        #         max_new_tokens=2048 - len(sentence_ids[0]),
        #         pad_token_id=self.tokenizer.pad_token_id,
        #     )[:, len(sentence_ids[0]):][0].detach().cpu()

        self.validation_step_outputs[-1].append({
            # 'prediction': self.tokenizer.decode(output),
            # 'target': [a[0] for a in answers],
            'loss': generation_loss.item()
        })

    def longcontext_pretrain_validation_step(self, batch, batch_idx):
        
        doc_ids = batch[0]
        token_idx = 1
        step = 0

        losses = {idx: [] for idx in range(64)}

        while token_idx < len(doc_ids[0]):

            if not (step % 2 == 0 and step > 0):
                # inject into memory
                self.model.inject_memory(doc_ids[:, token_idx:token_idx+(2 * self.model.num_tokens)].cuda(), update_memory=True)
                token_idx += (2 * self.model.num_tokens)
                step += 1
                continue
            
            if step >= 64:
                break

            #### for memory-based models:
            input_ids = doc_ids[:, token_idx:token_idx+2048]
            labels = input_ids.clone()
            loss = self.model(input_ids=input_ids.cuda(), labels=labels.cuda()).loss.item()
            losses[step].append(loss)

            self.model.inject_memory(doc_ids[:, token_idx:token_idx+(2 * self.model.num_tokens)].cuda(), update_memory=True)
            token_idx += (2 * self.model.num_tokens)
            step += 1
        
        losses = {k: np.mean(v) for k, v in losses.items() if len(v) > 0}
        self.validation_step_outputs[-1].append({
            'all_losses': losses
        })


    def cqa_validation_step(self, batch, batch_idx):

        context_ids, sentence_ids, answer_ids = batch[:3]
        unrelated_contexts = batch[3:]
        unrelated_contexts = [x for x in unrelated_contexts]

        if self.model.num_tokens == 1024:
            while context_ids.shape[1] < 1536:
                unrelated_context = unrelated_contexts.pop(0)
                context_ids = torch.cat([context_ids, unrelated_context], dim=1)

        self.model.inject_memory(
            context_ids,
            update_memory=True,
        )

        output = self.model.generate(
            inputs=sentence_ids, 
            max_new_tokens=50,
            pad_token_id=self.tokenizer.pad_token_id,
        )[:, len(sentence_ids[0]):][0].detach().cpu()
        
        middle_outputs = []

        unrelated_context = unrelated_contexts.pop(0)

        idx = 0

        while len(unrelated_contexts) > 0:
            
            if self.model.num_tokens == 1024:

                if unrelated_context is None or unrelated_context.shape[1] < 1536:
                    if unrelated_context is None:
                        unrelated_context = unrelated_contexts.pop(0)
                    else:
                        unrelated_context = torch.cat([unrelated_context, unrelated_contexts.pop(0)], dim=1)
                    continue
                    
                if unrelated_context.shape[1] > 2048:
                    unrelated_context = unrelated_context[:, :2048]
                
            else:

                unrelated_context = unrelated_contexts.pop(0)
                
            self.model.inject_memory(
                unrelated_context,
                update_memory=True,
            )

            unrelated_context = None

            if idx % self.val_injection_steps_per_eval == 0:

                middle_out = self.model.generate(
                    inputs=sentence_ids, 
                    max_new_tokens=50,
                    pad_token_id=self.tokenizer.eos_token_id
                )[:, len(sentence_ids[0]):][0].detach().cpu()

                middle_outputs.append(middle_out)

            idx += 1
            
            if idx > (120 if self.model.num_tokens == 256 else 40):
                break

        outputs = {
            'prediction': output,
            'target': answer_ids[0].cpu(),
            "middle_outputs": middle_outputs,
        }

        self.validation_step_outputs[-1].append(outputs)


    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        if self.validation_dataset_names[dataloader_idx] in ['slimpajama', 'fineweb']:
            if dataloader_idx > len(self.validation_step_outputs) - 1:
                self.validation_step_outputs.append([])
            return self.short_validation_step(batch, batch_idx)
    
        if self.validation_dataset_names[dataloader_idx] in ['narrativeqa', '2wikimqa', 'hotpotqa', 'qasper', 'musique', 'multifieldqa_en']:
            if dataloader_idx > len(self.validation_step_outputs) - 1:
                self.validation_step_outputs.append([])
            return self.longcontext_validation_step(batch, batch_idx)

        if self.validation_dataset_names[dataloader_idx] in ['slimlong']:
            if dataloader_idx > len(self.validation_step_outputs) - 1:
                self.validation_step_outputs.append([])
            return self.longcontext_pretrain_validation_step(batch, batch_idx)

        if self.validation_dataset_names[dataloader_idx] in ['squad', 'nqa']:
            if dataloader_idx > len(self.validation_step_outputs) - 1:
                self.validation_step_outputs.append([])
            return self.cqa_validation_step(batch, batch_idx)

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []
        if self.backup_memory_when_validating:
            self.backup_memory = self.model.memory.data.clone().detach()
        else:
            self.backup_memory = None
    
    def summarize_cqa_results(self, output, dataloader_idx):
        
        preds = self.tokenizer.batch_decode([x['prediction'].tolist() for x in output])
        targets = self.tokenizer.batch_decode([x['target'].tolist() for x in output])
        accuracy = calculate_exact_hit_accuracy(preds, targets)
        qa_f1 = calculate_qa_f1_score(preds, targets)

        # how many ground-truth memory tokens are left
        if self.add_selector and self.use_retriever_during_validation:
            num_memory_tokens_left = np.mean([x['memory_indicators'] for x in output])
            self.log_dict({f'val/{self.validation_dataset_names[dataloader_idx]}_num_remaining_memory_tokens': num_memory_tokens_left}, logger=True, on_step=False, on_epoch=True)

        # log
        if self.validation_dataset_names is not None:
            self.log(f'val/{self.validation_dataset_names[dataloader_idx]}', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'val/{self.validation_dataset_names[dataloader_idx]}_qa_f1', qa_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        else:
            self.log(f'val/{dataloader_idx}', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'val/{dataloader_idx}_qa_f1', qa_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # log middle outputs
        for step in range(len(output[0]['middle_outputs'])):
            preds_cur_step = []
            for idx in range(len(preds)):
                preds_cur_step.append(output[idx]['middle_outputs'][step].tolist())
            
            preds_cur_step = self.tokenizer.batch_decode(preds_cur_step)

            accuracy = calculate_exact_hit_accuracy(preds_cur_step, targets)
            qa_f1 = calculate_qa_f1_score(preds_cur_step, targets)

            if self.validation_dataset_names is not None:
                self.log(f'val/m_{self.validation_dataset_names[dataloader_idx]}_{(step+1) * self.val_injection_steps_per_eval}', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'val/m_{self.validation_dataset_names[dataloader_idx]}_{(step+1) * self.val_injection_steps_per_eval}_qa_f1', qa_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            else:
                self.log(f'val/m_{idx}_{(step+1) * self.val_injection_steps_per_eval}', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'val/m_{idx}_{(step+1) * self.val_injection_steps_per_eval}_qa_f1', qa_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return 0

    def summarize_slimpajama_results(self, output, dataloader_idx):

        loss_with_memory = np.mean([x['loss_with_memory'] for x in output])
        self.log(f'val/{self.validation_dataset_names[dataloader_idx]}', loss_with_memory, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        if "loss_no_memory" in output[0]:
            loss_no_memory = np.mean([x['loss_no_memory'] for x in output])
            self.log(f'val/{self.validation_dataset_names[dataloader_idx]}_no_memory', loss_no_memory, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            return loss_with_memory + loss_no_memory

        return loss_with_memory

    def summarize_longcontext_results(self, output, dataloader_idx):

        losses = [x['loss'] for x in output]
        self.log(f'val/{self.validation_dataset_names[dataloader_idx]}', np.mean(losses), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return np.mean(losses)

        # predictions = [x['prediction'] for x in output]
        # all_targets = [x['target'] for x in output]

        # total_score = 0.
        # for prediction, targets in zip(predictions, all_targets):
        #     score = 0.
        #     prediction = prediction.lstrip('\n').split('\n')[0].strip()

        #     for target in targets:
        #         score = max(score, qa_f1_score(target, prediction))
        #     total_score += score
        
        # self.log(f'val/{self.validation_dataset_names[dataloader_idx]}', 100 * total_score / len(predictions), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    def summarize_pretrain_longcontext_results(self, output, dataloader_idx):
        all_losses = {}
        for out in output:
            out = out['all_losses']
            for key in out:
                if not key in all_losses:
                    all_losses[key] = [out[key]]
                else:
                    all_losses[key].append(out[key])
        all_losses = {k: np.mean(v) for k, v in all_losses.items()}

        for key in all_losses:
            self.log(f'val_longcontext/{self.validation_dataset_names[dataloader_idx]}_{key}', all_losses[key], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return np.mean(list(all_losses.values()))

    def on_validation_epoch_end(self):

        outputs = self.validation_step_outputs
        total_loss = 0
        all_long_context_loss = []

        for dataloader_idx, output in enumerate(outputs):

            if self.validation_dataset_names[dataloader_idx] in ['slimpajama', 'fineweb']:
                total_loss += self.summarize_slimpajama_results(output, dataloader_idx)
            elif self.validation_dataset_names[dataloader_idx] in ['narrativeqa', '2wikimqa', 'hotpotqa', 'qasper', 'musique', 'multifieldqa_en']:
                all_long_context_loss.append(self.summarize_longcontext_results(output, dataloader_idx))
            elif self.validation_dataset_names[dataloader_idx] in ['slimlong']:
                total_loss += self.summarize_pretrain_longcontext_results(output, dataloader_idx)
            else:
                total_loss += self.summarize_cqa_results(output, dataloader_idx)
        
        if len(all_long_context_loss) > 0:
            total_loss += np.mean(all_long_context_loss)
            self.log('val/avg_long_context_loss', np.mean(all_long_context_loss), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):

        if self.optimizer is None:

            # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

            if self.embedding_learning_rate is None and self.selector_learning_rate is None:
                optimizer = torch.optim.AdamW([param for param in self.model.parameters() if param.requires_grad], 
                                        lr=self.learning_rate)

                if self.lr_scheduler is not None:
                    scheduler = eval(self.lr_scheduler['target'])(optimizer, **self.lr_scheduler['params'])
                    return [optimizer], [scheduler]
        
            else:

                param_groups = []

                remaining_params = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]

                if self.embedding_learning_rate is not None:
                    embedding_params = [param for name, param in self.model.named_parameters() if 'embeddings' in name and param.requires_grad]
                    remaining_params = [(name, param) for name, param in remaining_params if not 'embeddings' in name]
                    param_groups.append({
                        'name': 'embedding',
                        'params': embedding_params,
                        'lr': self.embedding_learning_rate
                    })

                if self.selector_learning_rate is not None:
                    selector_params = [param for name, param in self.model.named_parameters() if ("query_proj" in name or "key_proj" in name) and param.requires_grad]
                    remaining_params = [(name, param) for name, param in remaining_params if not ("query_proj" in name or "key_proj" in name)]
                    param_groups.append({
                        'name': 'selector',
                        'params': selector_params,
                        'lr': self.selector_learning_rate
                    })
                
                param_groups.append({
                    'name': 'main',
                    'params': [param for name, param in remaining_params],
                    'lr': self.learning_rate
                })

                optimizer = torch.optim.AdamW(param_groups)

                if self.lr_scheduler is not None:
                    scheduler = eval(self.lr_scheduler['target'])(optimizer, **self.lr_scheduler['params'])
                    return [optimizer], [scheduler]

        elif 'fsdp' in self.optimizer:

            optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.learning_rate)

        else:

            parameters = [param for param in self.model.parameters() if param.requires_grad]

            # optimizer: deepspeed.ops.adam.DeepSpeedCPUAdam
            if "deepspeed" in self.optimizer:
                import deepspeed
                import ninja
            if "onebit" in self.optimizer:
                from deepspeed.runtime.fp16 import onebit
            if "FusedAdam" in self.optimizer:
                from deepspeed.ops.adam import FusedAdam
                optimizer = FusedAdam(parameters, lr=self.learning_rate)

            if "DeepSpeedCPUAdam" in self.optimizer:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                optimizer = DeepSpeedCPUAdam(parameters, lr=self.learning_rate)

            # optimizer = optimizer([param for param in self.model.parameters() if param.requires_grad], lr=self.learning_rate)


        return optimizer
        
    def init_from_lora_weights(self, weights):

        if self.initialize_decoder_lora_from_default:
            new_weights = {}
            for name, param in weights.items():
                new_weights[name] = param
                if 'default' in name:
                    new_weights[name.replace('default', 'decoder_adapter')] = param
            weights = new_weights

        model_dict = self.state_dict()
        for name, param in weights.items():
            if name in model_dict:
                if (model_dict[name].shape == param.shape):
                    model_dict[name].copy_(param)
                else:
                    print(f"{name} shape not match")

        self.load_state_dict(model_dict)
        self.model.initialized += 1

    def init_from_ckpt(self, ckpt_path):

        # load from lora weights
        weights = torch.load(ckpt_path, map_location='cpu')
        self.init_from_lora_weights(weights)
        print(f"Restored from {ckpt_path}")

        if hasattr(self.model.model, "ltm") and os.path.exists(os.path.join(os.path.dirname(ckpt_path), 'ltm.pt')):
            self.model.model.ltm = torch.load(os.path.join(os.path.dirname(ckpt_path), 'ltm.pt'))
            ltm_recall_frequencies = np.load(os.path.join(os.path.dirname(ckpt_path), 'ltm_recall_frequency.npz'))
            self.model.model.ltm_recall_frequencies = [
                ltm_recall_frequencies[f'arr_{i}']  for i in range(len(ltm_recall_frequencies))
            ]
            if os.path.exists(os.path.join(os.path.dirname(ckpt_path), 'ltm_ages.npy')):
                self.model.model.ltm_ages = np.load(os.path.join(os.path.dirname(ckpt_path), 'ltm_ages.npy'))
            else:
                self.model.model.ltm_ages = [np.zeros(len(ltm)) for ltm in self.model.model.ltm]
            
            with torch.no_grad():
                if self.model.maintain_memory_keys:
                    self.model.model.memory_keys = []
                    for idx in range(len(self.model.model.memory)):
                        self.model.model.memory_keys.append(self.model.model.model.layers[idx].self_attn.key_proj(self.model.model.model.layers[idx].input_layernorm(self.model.model.memory[idx])).data)
                    self.model.model.memory_keys = torch.stack(self.model.model.memory_keys)
                
                self.model.model.memory_ages = [np.zeros(len(memory)) for memory in self.model.model.memory]
    