import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os


class SQuADDataset(Dataset):
    def __init__(self, filename,
                 max_length=768,
                 num=None,
                 num_unrelated_contexts=0,
                 tokenizer='llama', 
                 tokenizer_path=None,):

        self.num_unrelated_contexts = num_unrelated_contexts
        self.max_length = max_length
        with open(filename, 'r') as file:
            raw_data = json.load(file)['data']
        if tokenizer == 'gpt2':
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif tokenizer == 'llama':
            from transformers import LlamaTokenizer

            # debug:
            if tokenizer_path is None:
                self.tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")
            else:
                self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        else:
            # raise ValueError("tokenizer must be one of ['gpt2', 'llama']")
            self.tokenizer = tokenizer
            self.tokenizer.pad_token = self.tokenizer.eos_token

        indices = np.load(os.path.join(os.path.dirname(filename), 'indices_squad_3.npy'))

        # load unrelated contexts:
        if num_unrelated_contexts > 0:
            with open(filename.replace("dev", "train"), 'r') as file:
                raw_data_train = json.load(file)['data']
            self.unrelated_contexts = []
            flag = False
            for entry in raw_data_train:
                for paragraph in entry['paragraphs']:
                    context = paragraph['context']
                    # make sure every context is long enough
                    if self.unrelated_contexts == []:
                        self.unrelated_contexts.append(context)
                        continue
                    if len(self.tokenizer(self.unrelated_contexts[-1], add_special_tokens=False).input_ids) < self.max_length:
                        self.unrelated_contexts[-1] += ' ' + context
                    else:
                        if len(self.unrelated_contexts) == num_unrelated_contexts:
                            flag = True
                            break
                        self.unrelated_contexts.append(context)
                if flag:
                    break
        else:
            self.unrelated_contexts = []
        # Flatten the data
        flag = False
        self.data = []
        for entry in raw_data:
            for paragraph in entry['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = qa['answers'][0]['text'] if not qa['is_impossible'] else ''
                    if answer == '': continue
                    self.data.append({
                        'context': context,
                        'question': question,
                        'answer': answer
                    })
                
                    if num is not None and (num < len(indices)) and len(self.data) == indices[num] + 1:
                    # if num is not None and len(self.data) == num:
                        
                        flag = True
                        break
                if flag:
                    break
            if flag:
                break
        if num is not None:
            indices = indices[:num]
        self.data = [self.data[i] for i in indices]
        print("Length of squad dataset:", len(self.data))

        # truncate unrelated_contexts into less than 512 tokens
        new_unrelated_contexts = []
        for unrelated_context in self.unrelated_contexts:
            new_unrelated_contexts.append(self.tokenizer.decode(self.tokenizer(unrelated_context, add_special_tokens=False).input_ids[:512]))
        self.unrelated_contexts = new_unrelated_contexts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, question, answer = self.data[idx].values()

        # unrelated_contexts = []
        # selected_indices = [idx]
        # for i in range(self.num_unrelated_contexts):
        #     random_idx = torch.randint(0, len(self.data), (1,)).item()
            
        #     while random_idx in selected_indices:
        #         random_idx = torch.randint(0, len(self.data), (1,)).item()
            
        #     unrelated_contexts.append(list(self.data[random_idx].values())[0])
        #     selected_indices.append(random_idx)
        if self.num_unrelated_contexts > 0:
            # unrelated_contexts = np.random.choice(
            #     self.unrelated_contexts,
            #     size=self.num_unrelated_contexts,
            #     replace=False
            # )
            unrelated_contexts = self.unrelated_contexts
        else:
            unrelated_contexts = []

        return "Context: " + context, \
            "Questions: " + question + " Answer:", \
            answer,\
            unrelated_contexts
        
        # return "Context: " + context, "Question: " + question + " Answer:", answer