import os
import re
import string
import importlib

import torch
import numpy as np
from collections import abc
from einops import rearrange
from functools import partial
from collections import Counter

import multiprocessing as mp
from threading import Thread
from queue import Queue
import random 

from inspect import isfunction
from metrics import qa_f1_score
from pytorch_lightning.callbacks import ModelCheckpoint



def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
        func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res


class ModelCheckpointLLM(ModelCheckpoint):
    
    def _save_checkpoint(self, trainer, filepath):
        weights = {}
        for name, param in trainer.model.module.named_parameters():
            if param.requires_grad or "memory" in name or "lora" in name:
                weights[name] = param.data
        
        torch.save(weights, filepath.replace("epoch=", "lora_epoch="))

        # save the long term memory, only save the latest ltm
        if hasattr(trainer.model.module.model, "ltm"):
            torch.save(trainer.model.module.model.ltm, os.path.join(os.path.dirname(filepath), "ltm.pt"))
            # Save multiple arrays into a single .npz file
            np.savez(os.path.join(os.path.dirname(filepath), 'ltm_recall_frequency.npz'), *trainer.model.module.model.ltm_recall_frequencies)

def select_mask_span(total_length, remaining_mask_count, max_span_length=20):
    start = random.randint(0, total_length - 1)  # random starting point
    length = min(random.randint(1, max_span_length), remaining_mask_count)
    end = min(start + length, total_length)
    return start, end

def apply_mask(token_ids, start, end, mask_token_id):
    token_ids[start:end] = mask_token_id

def mask_span(token_ids, mask_ratio, mask_token_id, length=None, replace=False):
    
    if length is None:
        length = len(token_ids)

    total_tokens_to_mask = int(mask_ratio * length)
    if replace:
        while total_tokens_to_mask > 0:
            start, end = select_mask_span(length, total_tokens_to_mask)
            # apply_mask(token_ids, start, end, mask_token_id)
            token_ids = torch.cat([token_ids[:start], torch.tensor([mask_token_id]), token_ids[end:]])
            length -= (end - start) + 1
            total_tokens_to_mask -= (end - start)
        return token_ids, length
    else:
        mask_indicators = np.zeros(length)
        while total_tokens_to_mask > np.sum(mask_indicators):
            start, end = select_mask_span(length, total_tokens_to_mask)
            apply_mask(token_ids, start, end, mask_token_id)
            mask_indicators[start:end] = 1


def mask_tokens(context_ids, contexts_attention_mask, mask_strategy, mask_ratio, tokenizer):
    
    if mask_strategy == 'span_replace':
        new_context_ids = []
        new_lengths = []

    for idx in range(len(context_ids)):

        if mask_strategy == 'word':
            context_length = torch.sum(contexts_attention_mask[idx]).item()
            indices = np.random.choice(np.arange(context_length), 
                            int(context_length * mask_ratio), 
                            replace=False)
            context_ids[idx][indices] = tokenizer.mask_token_id

        elif mask_strategy == 'span':
            mask_span(context_ids[idx], mask_ratio, tokenizer.mask_token_id, length=contexts_attention_mask[idx].sum().item())
        
        elif mask_strategy == 'span_replace':
            context = context_ids[idx][:contexts_attention_mask[idx].sum().item()]
            context, length = mask_span(context, mask_ratio, tokenizer.mask_token_id, replace=True)
            new_context_ids.append(context)
            new_lengths.append(length)

        else:
            raise NotImplementedError
    
    if mask_strategy == 'span_replace':
        
        max_length = max([len(x) for x in new_context_ids])
        new_context_ids = [torch.cat([x, torch.tensor([tokenizer.pad_token_id]*(max_length - len(x)))]) for x in new_context_ids]
        context_ids = torch.stack(new_context_ids).long()
        contexts_attention_mask = torch.tensor([[1]*x + [0]*(max_length - x) for x in new_lengths]).long()

    return context_ids, contexts_attention_mask


def add_context_to_list(new_context_ids, cur_context_ids, all_contexts, max_length):

    # new_context_ids: new_sentence
    # cur_context_ids: the sentence concatenated so far

    if len(new_context_ids) + len(cur_context_ids) > max_length:

        if len(cur_context_ids) > 0:
            all_contexts.append(cur_context_ids)

        while len(new_context_ids) > max_length:
            all_contexts.append(new_context_ids[:max_length])
            new_context_ids = new_context_ids[max_length:]
        
        cur_context_ids = new_context_ids
    
    else:
        cur_context_ids = torch.cat([cur_context_ids, new_context_ids])


    return cur_context_ids.long()

def collate_fn_longcqa(data, tokenizer, max_length, num_tokens, 
                    skip_first_token=False, 
                    end_special_token=None):

    contexts, questions, answers = zip(*data)

    questions_tokenized = tokenizer(list(questions), 
                                    max_length=max_length + start_pos,
                                    truncation=True, 
                                    padding='longest',
                                    return_tensors='pt',
                                    add_special_tokens=False)
    
    answers_tokenized = tokenizer(list(answers),
                                    max_length=max_length + start_pos,
                                    truncation=True,
                                    padding='longest',
                                    return_tensors='pt',
                                    add_special_tokens=False)

    # assert batch_size == 1
    assert len(contexts) == 1
    contexts = contexts[0]

    contexts = sent_tokenize(contexts)

    cur_context_ids = torch.tensor([])
    all_sentences = []

    for sen in contexts:
        cur_context_ids = add_context_to_list(
                    torch.tensor(pl_model.tokenizer(sen, add_special_tokens=False)['input_ids']), 
                    cur_context_ids, all_sentences, max_length)
    
    if len(cur_context_ids) > 0:
        all_sentences.append(cur_context_ids)
    
    contexts = torch.stack(all_sentences)

    # Create attention masks with total_length
    contexts_attention_mask = torch.stack(
            [torch.ones(num_tokens + sentence.shape[0]) for sentence in all_sentences]
    )

    return contexts, contexts_attention_mask, \
        questions_tokenized.input_ids, questions_tokenized.attention_mask, \
        answers_tokenized.input_ids, answers_tokenized.attention_mask

def collate_fn_qa(data, tokenizer, max_length, num_tokens, 
                eval_max_length=None,
                add_special_tokens=False, 
                end_special_token=None,
                mask_strategy=None,
                mask_ratio=None,
                padding='longest',
                is_ift=False,
                return_idx=False):
                
    eval_max_length = max_length if eval_max_length is None else eval_max_length

    contexts, questions, answers, unrelated_contexts, idx = zip(*data)

    if end_special_token is not None:
        answers = [x + end_special_token for x in list(answers)]
    
    contexts_tokenized = tokenizer(list(contexts), 
                                   max_length=max_length, 
                                   padding=padding,
                                   truncation=True, 
                                   return_tensors='pt',
                                   add_special_tokens=add_special_tokens)
    
    if is_ift:
        questions_tokenized = []
        for question in list(questions):
            question_tokenized = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                                               add_generation_prompt=True,
                                                               return_tensors='pt')[0][1:]
            questions_tokenized.append(question_tokenized)
        questions_tokenized = torch.stack(questions_tokenized)

    else:
        questions_tokenized = tokenizer(list(questions), 
                                        max_length=eval_max_length,
                                        truncation=True, 
                                        padding='longest',
                                        return_tensors='pt',
                                        add_special_tokens=add_special_tokens).input_ids
    
    answers_tokenized = tokenizer(list(answers),
                                    max_length=eval_max_length,
                                    truncation=True,
                                    padding='longest',
                                    return_tensors='pt',
                                    add_special_tokens=add_special_tokens)

    unrelated_contexts = np.array(unrelated_contexts).transpose().tolist()

    all_unrelated_contexts = {}
    for i in range(len(unrelated_contexts)):
        all_unrelated_contexts[i] = tokenizer(unrelated_contexts[i],
                                    max_length=max_length,
                                    truncation=True,
                                    padding=padding,
                                    return_tensors='pt',
                                    add_special_tokens=add_special_tokens)

    outputs = (contexts_tokenized.input_ids, \
        questions_tokenized, \
        answers_tokenized.input_ids)
    
    for i in range(len(all_unrelated_contexts)):
        outputs += (all_unrelated_contexts[i].input_ids,)
    
    if return_idx:
        outputs += (idx,)

    return outputs



def collate_fn_qa_bs(data, tokenizer, max_length, num_tokens, with_context=True, related_position='begin'):
    contexts, questions, answers, unrelated_contexts = zip(*data)

    questions_tokenized = {
        'input_ids': [],
        'attention_mask': []
    }
    if with_context:
        if len(unrelated_contexts) > 0:
            for context, question, unrelated_context in zip(contexts, questions, unrelated_contexts):
                if len(unrelated_context) > 0:

                    if related_position == 'begin':
                        context = context + ' ' + " ".join(unrelated_context)
                    elif related_position == 'end':
                        context = " ".join(unrelated_context) + ' ' + context
                    else:
                        assert related_position == 'random'
                        # find the index to insert context into unrelated_context
                        index = random.randint(0, len(unrelated_context))
                        context = " ".join(unrelated_context[:index]) + ' ' + context + ' ' + " ".join(unrelated_context[index:])

                encoded_input = tokenizer.encode_plus(
                    context,
                    question,
                    max_length=max_length,
                    truncation="only_first",
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                questions_tokenized['input_ids'].append(encoded_input['input_ids'])
                questions_tokenized['attention_mask'].append(encoded_input['attention_mask'])
            questions_tokenized['input_ids'] = torch.cat(questions_tokenized['input_ids'], dim=0)
            questions_tokenized['attention_mask'] = torch.cat(questions_tokenized['attention_mask'], dim=0)

        else:
            for context, question in zip(contexts, questions):
                encoded_input = tokenizer.encode_plus(
                    context,
                    question,
                    max_length=max_length,
                    truncation="only_first",
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                questions_tokenized['input_ids'].append(encoded_input['input_ids'])
                questions_tokenized['attention_mask'].append(encoded_input['attention_mask'])
            questions_tokenized['input_ids'] = torch.cat(questions_tokenized['input_ids'], dim=0)
            questions_tokenized['attention_mask'] = torch.cat(questions_tokenized['attention_mask'], dim=0)
        
        questions_tokenized['input_ids'] = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), questions_tokenized['input_ids']], dim=1)
        questions_tokenized['attention_mask'] = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), questions_tokenized['attention_mask']], dim=1)
        
    else:
        questions_tokenized = tokenizer(list(questions),
                                    max_length=max_length,
                                    truncation=True,
                                    padding='longest',
                                    return_tensors='pt')

    answers_tokenized = tokenizer(list(answers),
                                    max_length=max_length,
                                    # padding='max_length',
                                    truncation=True,
                                    padding='longest',
                                    return_tensors='pt',
                                    add_special_tokens=False)

    return questions_tokenized['input_ids'], questions_tokenized['attention_mask'], \
        answers_tokenized['input_ids'], answers_tokenized['attention_mask']

def calculate_exact_hit_accuracy(preds, targets):
    hit = 0
    for pred, target in zip(preds, targets):
        if target.replace("<s>", "") in pred:
            hit += 1
    return hit / len(preds)

def calculate_qa_f1_score(preds, targets):
    return np.mean([qa_f1_score(pred, target) for pred, target in zip(preds, targets)])

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)