import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
import json
import os

class NQDataset(Dataset):
    def __init__(self, 
            filename,
            tokenizer='llama', 
            max_length=768,
            tokenizer_path=None,
            num=None,
            num_unrelated_contexts=0,
            expand_to_max_length=False,
        ):
        self.filename = filename
        self.max_length = max_length
        self.num_unrelated_contexts = num_unrelated_contexts
        self.expand_to_max_length = expand_to_max_length
        self.questions = []
        self.long_answers = []
        self.short_answers = []

        if tokenizer == 'gpt2':
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif tokenizer == 'llama':
            from transformers import LlamaTokenizer

            # debug:
            if tokenizer_path is None:
                self.tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        else:
            # raise ValueError("tokenizer must be one of ['gpt2', 'llama']")
            self.tokenizer = tokenizer
            self.tokenizer.pad_token = self.tokenizer.eos_token

        count = 0

        indices = np.load(os.path.join(os.path.dirname(filename), 'indices_nq_4.npy'))


        # get some unrelated context anyways:
        unrelated_text = []
        train_filename = os.path.join("./data/nq", 'v1.0-simplified_simplified-nq-train.jsonl')
        with open(train_filename, 'r') as file:
            for line in file:
                json_obj = json.loads(line)
                question_text = json_obj['document_text']
                unrelated_text.append(question_text)
                if len(unrelated_text) == 3:
                    break
        unrelated_text = " ".join(unrelated_text)


        with open(filename, 'r') as file:
            for line in file:
                json_obj = json.loads(line)

                if json_obj['annotations'][0]['yes_no_answer'] == 'None':
                    continue

                question_text = json_obj['question_text']
                long_answer = json_obj['annotations'][0]['long_answer']
                
                start_token = long_answer['start_token']
                end_token = long_answer['end_token']

                # Get the list of tokens
                tokens = json_obj['document_tokens']

                # Extract the tokens of the long answer
                long_answer_tokens = tokens[start_token:end_token]

                # Concatenate the token texts to get the long answer text
                long_answer_text = " ".join(token['token'] for token in long_answer_tokens if not token['html_token'])

                short_answers = json_obj['annotations'][0]['short_answers']
                if len(short_answers) == 0: continue
                # TODO: Not sure if this is the correct way
                short_answer = short_answers[0]
                start_token = short_answer['start_token']
                end_token = short_answer['end_token']
                short_answer_tokens = tokens[start_token:end_token]
                short_answer_text = " ".join(token['token'] for token in short_answer_tokens if not token['html_token'])

                self.questions.append(question_text)
                if expand_to_max_length:
                    current_length = len(self.tokenizer(question_text, add_special_tokens=False).input_ids)
                    if current_length <= 512:
                        self.long_answers.append(long_answer_text + ' ' + self.tokenizer.decode(
                            self.tokenizer(unrelated_text, add_special_tokens=False).input_ids[:512 - current_length]
                        ))
                        count += 1

                else:
                    self.long_answers.append(long_answer_text)
                self.short_answers.append(short_answer_text)

                # if num is not None and len(self.questions) == num: break

                if (num is not None) and (num < len(indices)) and len(self.questions) == indices[num] + 1: break
        if num is not None:
            indices = indices[:num]
        self.questions = [self.questions[i] for i in indices]
        self.long_answers = [self.long_answers[i] for i in indices]
        self.short_answers = [self.short_answers[i] for i in indices]
        print("Length of dataset:", len(self.questions))

        if num_unrelated_contexts > 0:
            self.unrelated_contexts = []
            # train_filename = filename.replace("dev-all", "train")
            train_filename = os.path.join(os.path.dirname(filename), 'v1.0-simplified_simplified-nq-train.jsonl')
            with open(train_filename, 'r') as file:
                for line in file:
                    json_obj = json.loads(line)

                    if json_obj['annotations'][0]['yes_no_answer'] == 'None':
                        continue

                    question_text = json_obj['question_text']
                    long_answer = json_obj['annotations'][0]['long_answer']
                    
                    start_token = long_answer['start_token']
                    end_token = long_answer['end_token']

                    # Get the list of tokens
                    if 'document_tokens' in json_obj:
                        tokens = json_obj['document_tokens']
                        long_answer_tokens = tokens[start_token:end_token]
                        # Concatenate the token texts to get the long answer text
                        long_answer_text = " ".join(token['token'] for token in long_answer_tokens if not token['html_token'])
                    else:
                        tokens = json_obj['document_text']
                        # Extract the tokens of the long answer
                        long_answer_text = tokens[start_token-1:end_token]
                    
                    if len(self.unrelated_contexts) == 0:
                        self.unrelated_contexts.append(long_answer_text)
                        continue
                        
                    if len(self.tokenizer(self.unrelated_contexts[-1], add_special_tokens=False).input_ids) <= 512:
                        self.unrelated_contexts[-1] += ' ' + long_answer_text
                    else:
                        if len(self.unrelated_contexts) == self.num_unrelated_contexts:
                            break
                        self.unrelated_contexts.append(long_answer_text)
        
        else:
            self.unrelated_contexts = []
        print("Length of unrelated contexts:", len(self.unrelated_contexts))
        
        # truncate unrelated_contexts into less than 512 tokens
        new_unrelated_contexts = []
        for unrelated_context in self.unrelated_contexts:
            new_unrelated_contexts.append(self.tokenizer.decode(self.tokenizer(unrelated_context, add_special_tokens=False).input_ids[:512]))
        self.unrelated_contexts = new_unrelated_contexts

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):

        # unrelated_contexts = []
        # selected_indices = [idx]
        # for i in range(self.num_unrelated_contexts):
        #     random_idx = torch.randint(0, len(self.questions), (1,)).item()
            
        #     while random_idx in selected_indices:
        #         random_idx = torch.randint(0, len(self.questions), (1,)).item()
            
        #     unrelated_contexts.append(self.long_answers[random_idx])
        #     selected_indices.append(random_idx)
        
        return "Context: " + self.long_answers[idx], "Questions: " + self.questions[idx] + "? Answer:", \
        self.short_answers[idx], self.unrelated_contexts
