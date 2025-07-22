import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import gzip
import os

class CQADataset(Dataset):
    def __init__(self, 
            root,
            max_length=768,
            num=None,
            num_unrelated_contexts=0,
            expand_to_max_length=False,
            is_ift=False,
        ):
        self.type = 'cqa'
        self.max_length = max_length
        self.num_unrelated_contexts = num_unrelated_contexts
        self.expand_to_max_length = expand_to_max_length
        self.questions = []
        self.long_answers = []
        self.short_answers = []
        self.is_ift = is_ift
        self.unrelated_text = json.load(open(os.path.join(root, 'unrelated_contexts_from_train.json'), 'r'))
        self.data = json.load(open(os.path.join(root, 'val_data.json'), 'r'))
        if num is not None:
            self.data = self.data[:num]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        context, question, answer = self.data[idx].values()

        if self.num_unrelated_contexts > 0:
            unrelated_contexts = np.random.choice(self.unrelated_text, self.num_unrelated_contexts, replace=False)
        else:
            unrelated_contexts = []

        if self.is_ift:
            return context, \
                question, \
                answer,\
                unrelated_contexts,\
                idx

        else:
            return context, \
                "Question: " + question + " Answer:", \
                " " + answer,\
                unrelated_contexts,\
                idx


if __name__ == '__main__':

    dataset = CQADataset(root="../../../data/nq-p", 
                        #  num=500,
                        #  tokenizer='llama',
                        #  tokenizer_path="meta-llama/Meta-Llama-3-8B"
                         )

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    all_indices = []
    for idx, (ctx, question, answer, unrelated_ctxs) in enumerate(dataset):
        ctx_input_ids_length = len(tok(ctx, add_special_tokens=False)['input_ids'])
        if ctx_input_ids_length < 512 and ctx_input_ids_length > 256:
            all_indices.append(idx)
    

    data = dataset.data
    data = [data[i] for i in all_indices]
    with open("/u/wangyu/wangyu/MemoryLLM-llama3/data/nq-p-2/val_data.json", 'w') as f:
        json.dump(data, f)
    