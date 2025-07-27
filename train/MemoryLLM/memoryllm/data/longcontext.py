import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
import gzip
import os

class LongContextDataset(Dataset):
    def __init__(self, 
                dataset,
                num=None,
                is_ift=False,
                tokenizer_path=None,
                max_chunk_length=512,
                max_length=8192,
                prompt_format_path="longbench_config/dataset2prompt_mem.json",
                maxlen_path="longbench_config/dataset2maxlen.json",):
        
        self.type = 'long_context'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.ds = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
        if num is not None:
            self.ds = self.ds.select(list(range(num)))
        self.is_ift = is_ift
        dataset2prompt = json.load(open(prompt_format_path))
        dataset2maxlen = json.load(open(maxlen_path))
        self.max_length = max_length
        self.max_chunk_length = max_chunk_length
        self.prompt_format = dataset2prompt[dataset]
        self.gen_maxlen = dataset2maxlen[dataset]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        
        json_obj = self.ds[idx]

        prompt = self.prompt_format.format(**json_obj)
        tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        prompt = self.tokenizer.decode(
            tokenized_prompt[-(self.max_length - self.gen_maxlen - 1):], 
            skip_special_tokens=True)

        contexts_ids = []
        prompt_ids = self.tokenizer(prompt, 
                                    add_special_tokens=False,
                                    truncation=False,
                                    return_tensors='pt').input_ids[0]
        
        while len(prompt_ids) > 0:
            if contexts_ids == []:
                contexts_ids.append(prompt_ids[-(2048-self.gen_maxlen):])
                prompt_ids = prompt_ids[:-(2048-self.gen_maxlen)]
            else:
                contexts_ids.append(prompt_ids[-self.max_chunk_length:])
                prompt_ids = prompt_ids[:-self.max_chunk_length]

        contexts_ids = contexts_ids[::-1]
        
        sentence_ids = contexts_ids[-1]
        contexts_ids = contexts_ids[:-1]

        answers = json_obj["answers"]

        return contexts_ids, sentence_ids, answers

if __name__ == '__main__':

    dataset = LongContextDataset('narrativeqa',
                                 prompt_format_path='../../../longbench_config/dataset2prompt_mem.json',
                                 maxlen_path='../../../longbench_config/dataset2maxlen.json',
                                 tokenizer_path="meta-llama/Meta-Llama-3.1-8B")
    
    for contexts_ids, sentence_ids in dataset:
        import ipdb; ipdb.set_trace()