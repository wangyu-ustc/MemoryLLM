import os
import json
import torch
import gzip
import random
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def get_chunks(lengths, max_length):

    if len(lengths) == 1:
        return [lengths]

    if lengths[0] > max_length:
        return [[lengths[0]], *get_chunks(lengths[1:], max_length)]

    if sum(lengths) <= max_length:
        return [lengths]

    if lengths[0] + lengths[1] > max_length:
        return [[lengths[0]], *get_chunks(lengths[1:], max_length)]
    else:
        # if we merge the first two numbers
        solution1 = [[lengths[0], lengths[1]], *get_chunks(lengths[2:], max_length)]
        
        # if we don't merge the first two numbers
        solution2 = [[lengths[0]], *get_chunks(lengths[1:], max_length)]

        if len(solution1) > len(solution2):
            return solution2
        else:
            return solution1

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

class InstructionTuningDataset(Dataset):
    def __init__(self,root="./data/ift", 
                 files=None,
                 tokenizer_path=None,
                 min_length=256,
                 max_length=768,
                 num_tokens=768,
                 max_seq_length=None,
                 add_end_special_token=False,
                 add_special_tokens=False,
                #  redpajama_length=None,
                 redpajama_ratio=0.1,
                 redpajama_config=None
                 ):
        '''
        split: ['sentence', 'random']
        '''
        self.root = root
        self.max_length = max_length
        self.min_length = min_length
        self.num_tokens = num_tokens
        self.add_end_special_token = add_end_special_token
        self.add_special_tokens = add_special_tokens
        self.max_seq_length = max_seq_length if max_seq_length is not None else max_length
        self.redpajama_ratio = redpajama_ratio

        if self.redpajama_ratio > 0:
            self.redpajama_dataset = RedPajamaDataset(**redpajama_config)
            self.redpajama_dataset_iter = iter(self.redpajama_dataset)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.data = []
        if files is None:
            for file in os.listdir(root):
                data = json.load(open(f"{root}/{file}", "r"))
                print("Loaded", len(data), "examples from", file)
                self.data.extend(data)

        else:
            for file in files:
                if file == 'allenai/tulu-3-sft-mixture':
                    try:
                        from datasets import load_dataset
                        ds = load_dataset("allenai/tulu-3-sft-mixture")['train']
                    except:
                        from datasets import load_from_disk
                        ds = load_from_disk("./local_data/allenai-tulu-3-sft-mixture")['train']
                    self.data.extend([x['messages'] for x in ds])
                else:
                    assert os.path.exists(f"{root}/{file}"), "File does not exist"
                    data = json.load(open(f"{root}/{file}", "r"))
                    print("Loaded", len(data), "examples from", file)
                    self.data.extend(data)
        
        print("Loaded", len(self.data), "examples")
        new_data = []
        for exp in self.data:
            flag = False
            for turn in exp:
                if turn['role'] == 'user' and len(turn['content']) > 20000:
                    flag = True
                    break
                if turn['role'] == 'assistant' and len(turn['content']) == 0:
                    flag = True
                    break
            if flag:
                continue
            new_data.append(exp)
        
        self.data = new_data
        print(f"Loaded {len(self.data)} examples after filtering")

    def get_context_and_sentence(self, doc, return_doc=False):
        
        # tokenizer doc:
        if return_doc:
            sentence = self.tokenizer(doc, 
                                       return_tensors='pt', 
                                       truncation=True, 
                                       add_special_tokens=self.add_special_tokens, 
                                       max_length=self.max_length).input_ids[0]
            return sentence
        
        doc = self.tokenizer(doc, return_tensors='pt', truncation=False, add_special_tokens=False).input_ids[0]

        chunks = split_sequence(len(doc), self.min_length, self.max_length)
        contexts = []
        for chunk in chunks:
            contexts.append(doc[:chunk])
            doc = doc[chunk:]
        
        return contexts

    def __len__(self):
        return int(len(self.data) / (1 - self.redpajama_ratio))

    def get_ids_and_labels_for_conversation(self, messages, remove_additional_contexts=False):

        contexts_ids = []

        sentence_ids = self.tokenizer.apply_chat_template(messages,
                                return_tensors="pt")[0][1:]
        
        if remove_additional_contexts:
            while len(sentence_ids) > self.max_seq_length:
                messages = messages[:-2]
                sentence_ids = self.tokenizer.apply_chat_template(messages,
                                    return_tensors="pt")[0][1:]
                if len(messages) == 2:
                    sentence_ids = sentence_ids[:self.max_seq_length]
                    break

        if sentence_ids.shape[0] > self.max_seq_length:

            # first identify how many tokens are put in sentence_ids
            all_message_ids = [self.tokenizer(x['content'], add_special_tokens=False, return_tensors='pt').input_ids[0] for x in messages]

            idx = 0
            contexts_ids = [torch.tensor([])]
            # for message, message_ids in zip(messages[:-2], all_message_ids[:-2]):
            last_header_and_message = []

            while idx < len(messages):

                if len(contexts_ids[-1]) > self.min_length:
                    contexts_ids.append(torch.tensor([]))
                    last_header_and_message = []
                
                if messages[idx]['role'] == 'user':
                    header = self.tokenizer("[User]\n\n", add_special_tokens=False, return_tensors='pt').input_ids[0]
                else:
                    header = self.tokenizer("[Assistant]\n\n", add_special_tokens=False, return_tensors='pt').input_ids[0]
            
                current_length = len(contexts_ids[-1])

                if current_length + len(header) + len(all_message_ids[idx]) > self.max_length:
                    contexts_ids[-1] = torch.cat([
                        contexts_ids[-1],
                        header,
                        all_message_ids[idx][:self.max_length - len(header) - current_length]
                    ])
                    contexts_ids.append(torch.tensor([]))
                    last_header_and_message = []
                    all_message_ids[idx] = all_message_ids[idx][self.max_length - len(header) - current_length:]
                
                else:
                    last_header_and_message.append((messages[idx]['role'], all_message_ids[idx]))
                    contexts_ids[-1] = torch.cat([
                        contexts_ids[-1],
                        header,
                        all_message_ids[idx]
                    ])
                    idx += 1

            contexts_ids = [x.long() for x in contexts_ids]
            contexts_ids = contexts_ids[:-1]

            messages = []
            for role, message in last_header_and_message:
                messages.append({
                    'role': role,
                    'content': self.tokenizer.decode(message)
                })

            turn_ids = []
            sentence_labels = []
            for turn in messages:
                role_token_id = self.tokenizer(turn['role'], add_special_tokens=False).input_ids
                cur_turn_ids = torch.tensor([128006, role_token_id[0], 128007, 271] + self.tokenizer(turn['content'], add_special_tokens=False).input_ids + [128009])
                turn_ids.append(cur_turn_ids)
                if turn['role'] == 'assistant':
                    sentence_labels.extend([-100] * 4 + cur_turn_ids[4:].tolist())
                elif turn['role'] == 'user':
                    sentence_labels.extend([-100] * len(cur_turn_ids))
                else:
                    raise ValueError("Invalid role")
            sentence_ids = torch.cat(turn_ids)
            sentence_labels = torch.tensor(sentence_labels)

        else:
            sentence_labels = torch.ones_like(sentence_ids) * -100
            count = 0
            for turn in messages:
                if turn['role'] == 'assistant':
                    input_ids = self.tokenizer(turn['content'], add_special_tokens=False, return_tensors='pt').input_ids[0]
                    found = False
                    while count < len(sentence_ids) - len(input_ids) + 1:
                        if (sentence_ids[count:count+len(input_ids)] == input_ids).all():
                            sentence_labels[count:count+len(input_ids)+1] = sentence_ids[count:count+len(input_ids)+1]
                            found = True
                            count += 1
                            break
                        count += 1
                    assert found

        return contexts_ids, sentence_ids, sentence_labels

    def __getitem__(self, idx):

        if idx < len(self.data):
            exp = self.data[idx]
            context = [x['content'] for x in exp if x['role'] == 'context']
            exp = [x for x in exp if x['role'] != 'context']

            for turn in exp:
                turn['content'] = turn['content'].strip()

            if len(context) > 0:
                context = context[0]
                contexts = self.get_context_and_sentence(context)
                if contexts[-1].shape[0] < self.min_length:
                    exp = [{
                        'role': 'context',
                        'content': self.tokenizer.decode(contexts[-1])
                    }] + exp
                    contexts = contexts[:-1]
                additional_contexts, sentence_ids, sentence_labels = self.get_ids_and_labels_for_conversation(exp, remove_additional_contexts=True)

            else:
                # no context, all conversations
                contexts = []
                additional_contexts, sentence_ids, sentence_labels = self.get_ids_and_labels_for_conversation(exp)

            return contexts + additional_contexts, sentence_ids, sentence_labels, 2
            
        else:
            
            contexts, sentence_ids, label = next(self.redpajama_dataset_iter)

            if label == 4:
                sentence_ids = torch.cat(contexts)
                sentence = "Repeat: " + self.tokenizer.decode(sentence_ids)
                sentence_ids = self.tokenizer(sentence, 
                                              return_tensors='pt', 
                                              add_special_tokens=False,
                                              truncation=True,
                                              max_length=self.max_seq_length).input_ids[0]
                sentence_labels = sentence_ids.clone()
                sentence_labels[:len(self.tokenizer("Repeat:", add_special_tokens=False).input_ids)] = -100
                
            else:
                sentence_labels = sentence_ids.clone()

            return contexts, sentence_ids, sentence_labels, label

if __name__ == '__main__':

    from tqdm import tqdm

    dataset = InstructionTuningDataset(root="../../../local_data/ift",
                               tokenizer_path="meta-llama/Meta-Llama-3-8B-Instruct",
                            #    files=['redpajama_ift.json'],
                               files=[
                                        'allenai/tulu-3-sft-mixture', 
                                        'squad.json', 
                                        'narrativeqa.json'
                                        ],
                               add_end_special_token=True,
                            #    min_length=256,
                            #    max_length=512,
                            #    max_seq_length=512,
                               min_length=1536,
                               max_length=2048,
                               max_seq_length=2048,
                               redpajama_ratio=0.0
    )
                            #    redpajama_ratio=0.01,
                            #    redpajama_config={
                            #         'root': "../../../data/redpajama",
                            #         'tokenizer': 'llama',
                            #         'tokenizer_path': "meta-llama/Meta-Llama-3-8B-Instruct",
                            #         'snapshots': ['2023-14'],
                            #         'shuffle': True,
                            #         'partition': 'head_middle',
                            #         'add_special_tokens': False,
                            #         'target_is_context_ratio': 0.1,
                            #         'max_length': 512,
                            #         'max_seq_length': 2048,
                            #         'num_tokens': 256
                            #    })
                            #    files=['ift_data_combined.json', "redpajama_ift.json", "LongAlpaca-12k-processed.json"])

    print("len dataset:", len(dataset))
    print("len dataset.data:", len(dataset.data))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    count = 0
    examples = []
    for batch in tqdm(dataloader):

        contexts, sentence_ids, sentence_labels, label = batch

        for ctx_ids in contexts:
            if not ctx_ids.shape[1] >= 1536 and ctx_ids.shape[1] <= 2048:
                import ipdb; ipdb.set_trace()
        
        if sentence_ids.shape[1] > 2048:
            import ipdb; ipdb.set_trace()

        # import ipdb; ipdb.set_trace()

        # if label == 1:
        #     print(dataset.tokenizer.decode(output[0])[-100:])
        #     import ipdb; ipdb.set_trace()

        # if not input.dtype == torch.int64 or not output.dtype == torch.int64:
        #     import ipdb; ipdb.set_trace()
