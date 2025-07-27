import os
import json
import torch
import gzip
import random
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset

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

class MixedStreamDataset(IterableDataset):
    def __init__(self, 
                 root="HuggingFaceFW/fineweb-edu", 
                 name="CC-MAIN-2024-10", 
                 longdoc_root='./local_data/long_data/',
                 split="train",
                 tokenizer_path=None,
                 min_length=256,
                 longdoc_min_length=None,
                 max_length=768,
                 num_tokens=768,
                 max_seq_length=None,
                 min_seq_length_for_full_context_and_sentence=None,
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
                 long_doc_start_chunk={
                    '4k-8k': 0,
                    '8k-16k': 0,
                    '16k-32k': 0,
                    '32k-64k': 0
                 },
                 short_document_start=0,
                 short_document_end=None,
                 seed=42,
                 ):
        
        self.root = root
        self.longdoc_root = longdoc_root
        self.max_length = max_length
        self.min_length = min_length
        self.longdoc_min_length = longdoc_min_length if longdoc_min_length is not None else min_length
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
        self.min_seq_length_for_full_context_and_sentence = min_seq_length_for_full_context_and_sentence

        self.long_doc_start_chunk = long_doc_start_chunk

        if self.repeat_with_unrelated:
            self.unrelated_contexts = []

        assert not (self.target_is_context_ratio > 0 and self.target_is_context), "you cannot set target_is_context_ratio > 0 and target_is_context=True at the same time!"

        if "local_data" in name:
            self.ds = load_from_disk(name)
        else:
            self.ds = load_dataset(root, name=name, split=split, trust_remote_code=True)

         # select the examples from 50000 to the last
        self.ds = self.ds.select(range(50000, len(self.ds)))
        short_document_end = short_document_end if short_document_end is not None else len(self.ds)
        self.ds = self.ds.select(range(short_document_start, short_document_end))
        self.short_document_length = len(self.ds)

        # shuffle self.ds
        self.ds = self.ds.shuffle(seed=seed)

        # Calculate the number of long documents needed for each category
        self.long_documents_ratios = long_documents_ratios
        total_long_ratio = sum(long_documents_ratios.values())
        self.total_length = len(self.ds) / (1 - total_long_ratio)

        self.long_documents_counts = {
            key: int(self.total_length * ratio) for key, ratio in long_documents_ratios.items()
        }
        self.total_length = self.short_document_length + sum(self.long_documents_counts.values())

        # TODO: check if the length of long documents surpasss the existing long documents

        self.key2chunk = {}
        # set the order of documents
        keys = ['short'] * self.short_document_length
        for key, count in self.long_documents_counts.items():
            keys += [key] * count
            # determine how many chunks to use
            num_chunks = int(np.ceil(count / 1000))
            self.key2chunk[key] = [str(i + self.long_doc_start_chunk[key]) for i in range(num_chunks)]
        
        # shuffle the keys
        # random.shuffle(keys)
        # shuffle the keys with the seed
        random.Random(seed).shuffle(keys)
        self.keys = keys

        self.key2count = {key: 0 for key in ['short'] + list(self.long_documents_counts.keys())}
        self.key2chunkid = {key: 0 for key in self.long_documents_counts}
        self.key2doc = {key:[] for key in self.long_documents_counts}

    def set_worker(self, worker_id, num_workers):
        # reset the keys
        
        # select the examples from self.ds
        
        print("worker_id:", worker_id, "short select from:", (len(self.ds) // num_workers) * worker_id, (len(self.ds) // num_workers) * (worker_id + 1))
        self.ds = self.ds.select(range((len(self.ds) // num_workers) * worker_id, (len(self.ds) // num_workers) * (worker_id + 1)))

        # select the chunk from self.key2chunk
        for key in self.key2chunk:
            chunks = self.key2chunk[key]
            total_chunks = len(chunks)
            self.key2chunk[key] = chunks[(total_chunks // num_workers) * worker_id: (total_chunks // num_workers) * (worker_id + 1)]

        # reset self.keys
        keys = ['short'] * len(self.ds)
        for key, chunks in self.key2chunk.items():
            keys += [key] * (len(chunks) * 1000)
        random.shuffle(keys)

        print("worker_id:", worker_id)
        print("key2chunk:", self.key2chunk)
    
    def __iter__(self):

        for key in self.keys:

            if key == "short":
                doc = self.ds[self.key2count[key]]['text']
            else:            
                if self.key2count[key] >= len(self.key2doc[key]):
                    with open(os.path.join(self.longdoc_root, f"{key}/chunk_{self.key2chunk[key][self.key2chunkid[key]]}.jsonl"), "r") as f:
                        lines = f.readlines()
                        lines = [json.loads(line)['text'] for line in lines]
                    self.key2doc[key] = lines
                    self.key2count[key] = 0
                    self.key2chunkid[key] += 1
                
                doc = self.key2doc[key][self.key2count[key]]
            
            self.key2count[key] += 1
            
            output = self.get_context_and_sentence(doc, min_length=self.min_length if key == "short" else self.longdoc_min_length)
            if output is None:
                continue

            contexts, sentence, label = output
            if label == 1:
                if self.target_is_context or random.random() < self.target_is_context_ratio:
                    label = 4

            yield contexts, sentence, label

    def get_context_and_sentence(self, doc, min_length, return_doc=False):
        
        if return_doc:
            contexts = [self.tokenizer(doc + self.end_special_token, 
                                       return_tensors='pt', 
                                       truncation=True, 
                                       add_special_tokens=self.add_special_tokens, 
                                       max_length=self.max_length).input_ids[0]]
            sentence = contexts[0]
            return contexts, sentence
        
        doc = self.tokenizer(doc + self.end_special_token, return_tensors='pt', truncation=False, add_special_tokens=False).input_ids[0]

        if len(doc) < min_length:
            if len(doc) > self.min_seq_length_for_full_context_and_sentence:
                return [doc], doc, 5
            else:
                return None

        if len(doc) < self.max_seq_length:
            chunks = split_sequence(len(doc), min_length, self.max_length)
            if len(chunks) == 1:
                return None
                
            else:
                contexts = []
                for chunk in chunks:
                    contexts.append(doc[:chunk])
                    doc = doc[chunk:]
                sentence = contexts.pop()

        else:

            if len(doc) < self.max_seq_length + min_length:
                contexts = [doc[:min_length]]
                sentence = doc[min_length:]
            else:

                seq_length = random.randint(min_length, self.max_seq_length)
                
                sentence = doc[-seq_length:]
                doc = doc[:-seq_length]

                assert len(doc) >= min_length
                chunks = split_sequence(len(doc), min_length, self.max_length)

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

    def get_repeat_context_sentence(self, doc, min_length):

        if self.repeat_with_unrelated:

            contexts, sentence = self.get_context_and_sentence(doc, min_length=min_length, return_doc=True)

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

            output = self.get_context_and_sentence(doc, min_length=min_length)
            if output is None:
                return None
            contexts, sentence, _ = output
            # sentence = torch.cat(contexts)
            
            return contexts, sentence, 4


if __name__ == '__main__':

    def worker_init_fn(worker_id):

        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset # the dataset copy in this worker process
        dataset.set_worker(worker_id, worker_info.num_workers)

        # all_dataset = len(dataset.all_paths)
        # per_worker = all_dataset // worker_info.num_workers
        # dataset.all_paths = dataset.all_paths[worker_id * per_worker: (worker_id + 1) * per_worker]
    
    dataset = MixedStreamDataset(longdoc_root='../../../local_data/long_data/',
                                 tokenizer_path="meta-llama/Meta-Llama-3.1-8B",
                                max_length=512,
                                min_length=16,
                                max_seq_length=2048,
                                add_special_tokens=False,
                                long_documents_ratios={
                                    '4k-8k': 0.2,
                                    '8k-16k': 0.2,
                                    '16k-32k': 0.2,
                                    '32k-64k': 0.2
                                },
                                short_document_end=200000,
                                short_document_start=0,)
    
    # for i, (contexts, sentence, label) in enumerate(dataset):
    #     print(i, contexts, sentence, label)
    #     if i == 10:
    #         break
    # for i, doc in enumerate(dataset):
    #     print(i, doc)
    #     if i == 10:
    #         break

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, worker_init_fn=worker_init_fn)    

    for idx, batch in enumerate(dataloader):
        # pass
        print(batch)
        # break

        # import ipdb; ipdb.set_trace()

        if idx % 10 == 0:
            import ipdb; ipdb.set_trace()
