import os
import json
import time
import torch
import gzip
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from itertools import islice

output_sampled_files = {
    "4k-8k": "local_data/long_data/sampled_long_data_4k_8k.jsonl",
    "8k-16k": "local_data/long_data/sampled_long_data_8k_16k.jsonl",
    "16k-32k": "local_data/long_data/sampled_long_data_16k_32k.jsonl",
    "32k-64k": "local_data/long_data/sampled_long_data_32k_64k.jsonl",
}

def split_sequence(seq_length, min_length, max_length):

    if seq_length < min_length:
        return [seq_length]
    
    # Initialize the chunks
    chunks = []
    remaining_length = seq_length

    while remaining_length > 0:
        # Calculate the maximum length for the current chunk
        max_chunk_length = min(remaining_length, max_length)
        
        # Ensure the chunk is at least min_length characters long and handle the remaining length
        if remaining_length <= max_length:
            if remaining_length < min_length and chunks:
                # Adjust previous chunks to make the last chunk at least min_length
                needed = min_length - remaining_length
                for i in range(len(chunks) - 1, -1, -1):
                    if chunks[i] - needed >= min_length:
                        chunks[i] -= needed
                        remaining_length += needed
                        break
                    else:
                        needed -= (chunks[i] - min_length)
                        remaining_length += (chunks[i] - min_length)
                        chunks[i] = min_length

            chunk_length = remaining_length if remaining_length >= min_length else min_length

        else:
            chunk_length = random.randint(min_length, max_chunk_length)
        
        # Append the chunk length to the list
        chunks.append(chunk_length)
        
        # Reduce the remaining length
        remaining_length -= chunk_length

    return chunks

class MixedDataset(Dataset):
    def __init__(self, root="HuggingFaceFW/fineweb-edu", 
                 name="CC-MAIN-2024-10", 
                 split="train",
                 tokenizer_path=None,
                 min_length=256,
                 max_length=768,
                 num_tokens=768,
                 max_seq_length=None,
                 end_special_token="",
                 add_special_tokens=False,
                 target_is_context=False,
                 shuffle_first_context=False,
                 overlap_contexts=False,
                 negative_contexts_ratio=0.0,
                 overlap_contexts_ratio=0.0,
                 target_is_context_ratio=0.0,
                 repeat_with_unrelated=False,
                 max_unrelated_at_one_step=20,
                 force_num_of_contexts=None,
                 long_documents_ratios={
                     '4k-8k': 0.3,
                     '8k-16k': 0.2,
                     '16k-32k': 0.2,
                     '32k-64k': 0.1
                 },
                 short_document_start=0,
                 short_document_end=None,
                 ):

        self.root = root
        self.max_length = max_length
        self.min_length = min_length
        self.num_tokens = num_tokens
        self.end_special_token = end_special_token
        self.add_special_tokens = add_special_tokens
        self.max_seq_length = max_seq_length if max_seq_length is not None else max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.target_is_context = target_is_context
        self.shuffle_first_context = shuffle_first_context
        self.overlap_contexts = overlap_contexts
        self.negative_contexts_ratio = negative_contexts_ratio
        self.overlap_contexts_ratio = overlap_contexts_ratio
        self.target_is_context_ratio = target_is_context_ratio
        # self.instruction = instruction
        self.last_contexts = None
        self.repeat_with_unrelated = repeat_with_unrelated
        self.max_unrelated_at_one_step = max_unrelated_at_one_step
        self.force_num_of_contexts = force_num_of_contexts
        if self.repeat_with_unrelated:
            self.unrelated_contexts = []

        assert not (self.target_is_context_ratio > 0 and self.target_is_context), "you cannot set target_is_context_ratio > 0 and target_is_context=True at the same time!"

        self.ds = load_dataset(root, name=name, split=split, trust_remote_code=True)
         # select the examples from 50000 to the last
        self.ds = self.ds.select(range(50000, len(self.ds)))
        short_document_end = short_document_end if short_document_end is not None else len(self.ds)
        self.ds = self.ds.select(range(short_document_start, short_document_end))
        self.short_document_length = len(self.ds)

        # Calculate the number of long documents needed for each category
        self.long_documents_ratios = long_documents_ratios
        total_long_ratio = sum(long_documents_ratios.values())
        self.total_length = len(self.ds) / (1 - total_long_ratio)

        self.long_documents_counts = {
            key: int(self.total_length * ratio) for key, ratio in long_documents_ratios.items()
        }
        self.total_length = self.short_document_length + sum(self.long_documents_counts.values())

        start_time = time.time()
        # Load the dataset from files
        # self.documents = []
        # for key, count in self.long_documents_counts.items():
        #     with open(output_sampled_files[key], 'r') as f:
        #         lines = f.readlines()
        #         lines = lines[:count]
        #         lines = [json.loads(line)['text'] for line in lines]
        #         self.documents.extend(lines)

        self.documents = []
        for key, count in self.long_documents_counts.items():
            with open(output_sampled_files[key], 'r') as f:


                # Use islice to read only the first 'count' lines
                # for line in islice(f, count):
                #     self.documents.append(json.loads(line)['text']
                # )

                # from count * (end / (end - start)) - count to count * (end / (end - start))
                for line in islice(f, int(count * (short_document_end / (short_document_end - short_document_start))) - count, count):
                    self.documents.append(json.loads(line)['text']
                )

        end_time = time.time()
        print("Reading long documents took:", end_time - start_time)

        print("Total Long documents loaded:", len(self.documents))
        print("Total short documents loaded:", self.short_document_length)
        print("Total documents loaded:", self.total_length)

    def get_context_and_sentence(self, doc, return_doc=False):
        
        if return_doc:
            contexts = [self.tokenizer(doc + self.end_special_token, 
                                       return_tensors='pt', 
                                       truncation=True, 
                                       add_special_tokens=self.add_special_tokens, 
                                       max_length=self.max_length).input_ids[0]]
            sentence = contexts[0]
            return contexts, sentence
        
        doc = self.tokenizer(doc + self.end_special_token, return_tensors='pt', truncation=False, add_special_tokens=False).input_ids[0]

        if len(doc) < self.min_length:
            return None

        if len(doc) < self.max_seq_length:
            chunks = split_sequence(len(doc), self.min_length, self.max_length)
            if len(chunks) == 1:
                return None
                
            else:
                contexts = []
                for chunk in chunks:
                    contexts.append(doc[:chunk])
                    doc = doc[chunk:]
                sentence = contexts.pop()

        else:

            if len(doc) < self.max_seq_length + self.min_length:
                contexts = [doc[:self.min_length]]
                sentence = doc[self.min_length:]
            else:

                seq_length = random.randint(self.min_length, self.max_seq_length)
                
                sentence = doc[-seq_length:]
                doc = doc[:-seq_length]

                assert len(doc) >= self.min_length
                chunks = split_sequence(len(doc), self.min_length, self.max_length)

                contexts = []
                for chunk in chunks:
                    contexts.append(doc[:chunk])
                    doc = doc[chunk:]

        assert len(contexts) > 0

        if self.add_special_tokens:
            contexts = [
                torch.cat([torch.tensor([self.tokenizer.bos_token_id]), context[:self.max_length-1]]) for context in contexts
            ]
            sentence = torch.cat([torch.tensor([self.tokenizer.bos_token_id]), sentence[:self.max_length-1]])

        return contexts, sentence, 1


    def get_repeat_context_sentence(self, doc):

        if self.repeat_with_unrelated:

            contexts, sentence = self.get_context_and_sentence(doc, return_doc=True)

            # randomly pick some contexts from self.unrelated_contexts:
            num_unrelated = min(self.max_unrelated_at_one_step, len(self.unrelated_contexts))

            np.random.shuffle(self.unrelated_contexts)
            unrelated_contexts = self.unrelated_contexts[:num_unrelated]
            contexts.extend(unrelated_contexts)
            
            # save the context to construct the set of unrelated_contexts
            if self.repeat_with_unrelated:
                if len(self.unrelated_contexts) < 200:
                    self.unrelated_contexts.append(contexts[0])
                else:
                    self.unrelated_contexts.pop(0)
                    self.unrelated_contexts.append(contexts[0])

            return contexts, sentence, 3
        
        else:

            output = self.get_context_and_sentence(doc)
            if output is None:
                return None
            contexts, sentence, _ = output
            # sentence = torch.cat(contexts)
            
            return contexts, sentence, 4
        

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < self.short_document_length:
            doc = self.ds[idx]['text']
        else:
            doc = self.documents[idx - self.short_document_length]
        
        if self.target_is_context or random.random() < self.target_is_context_ratio:
            output = self.get_repeat_context_sentence(doc)
        else:
            output = self.get_context_and_sentence(doc)

        if output is None:
            return self.__getitem__(random.randint(0, len(self.ds) - 1))

        contexts, sentence, label = output
        return contexts, sentence, label

if __name__ == '__main__':

    output_sampled_files = {
        "4k-8k": "../../../local_data/long_data/sampled_long_data_4k_8k.jsonl",
        "8k-16k": "../../../local_data/long_data/sampled_long_data_8k_16k.jsonl",
        "16k-32k": "../../../local_data/long_data/sampled_long_data_16k_32k.jsonl",
        "32k-64k": "../../../local_data/long_data/sampled_long_data_32k_64k.jsonl",
    }
    
    dataset = MixedDataset(
        tokenizer_path="meta-llama/Meta-Llama-3.1-8B",
        max_length=512,
        min_length=16,
        max_seq_length=2048,
        add_special_tokens=False,
        long_documents_ratios={
            '4k-8k': 0.3,
            '8k-16k': 0.2,
            '16k-32k': 0.2,
            '32k-64k': 0.1
        },
        short_document_end=70000,
        short_document_start=0,
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=True,
                            num_workers=8)
    

    for contexts, sentence, label in tqdm(dataloader):
        pass

