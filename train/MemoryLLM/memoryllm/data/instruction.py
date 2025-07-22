import os
import json
import torch
import gzip
import random
import numpy as np
from torch.utils.data import Dataset
try:
    from .redpajama import RedPajamaDataset
except:
    from redpajama import RedPajamaDataset


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
                 tokenizer='gpt2', 
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

        if tokenizer == 'gpt2':
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        elif tokenizer == 'llama':
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = tokenizer

        self.data = []
        if files is None:
            for file in os.listdir(root):
                
                # if 'redpajama' in file:
                #     if redpajama_length is not None:
                #         data = json.load(open(f"{root}/{file}", "r"))
                #         data = data[:redpajama_length]
                #     else:
                #         continue

                data = json.load(open(f"{root}/{file}", "r"))
                print("Loaded", len(data), "examples from", file)
                self.data.extend(data)
        else:
            for file in files:

                # if 'redpajama' in file:
                #     if redpajama_length is not None:
                #         data = json.load(open(f"{root}/{file}", "r"))
                #         data = data[:redpajama_length]
                #     else:
                #         continue
                    
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

    def merge_contexts(self, contexts):

        lengths = [len(x) for x in contexts]
        try:
            chunks = get_chunks(lengths, self.max_length - 1)
        except:
            import ipdb; ipdb.set_trace()

        new_contexts = []
        for chunk in chunks:
            if len(chunk) > 1:
                new_contexts.append(
                    self.tokenizer("\n\n".join(self.tokenizer.batch_decode(contexts[:len(chunk)])), 
                                    return_tensors='pt', 
                                    add_special_tokens=False).input_ids[0]
                )
            else:
                try:
                    new_contexts.append(contexts[0])
                except:
                    import ipdb; ipdb.set_trace()

            contexts = contexts[len(chunk):]

        return new_contexts

    def cut_contexts(self, all_sentence_ids):
        
        additional_contexts = []

        for sids in all_sentence_ids:

            sids = torch.cat(sids)
            
            if len(sids) > self.max_length:

                context_ids = sids[4:-1]
                if self.tokenizer.decode(sids[1]) == 'user':
                    header = self.tokenizer("[User]\n\n", add_special_tokens=False, return_tensors='pt').input_ids[0]
                # elif self.tokenizer.decode(sids[1]) == 'assistant':
                else:
                    header = self.tokenizer("[Assistant]\n\n", add_special_tokens=False, return_tensors='pt').input_ids[0]

                num_chunks = int(np.ceil(len(context_ids) / (self.max_length - len(header))))
                chunk_length = int(np.ceil(len(context_ids) / num_chunks))

                # S 
                # N = ceil(S / L)
                # ceil(S / N) <= L

                # if ceil(S / N) > L, then ceil (S / N) = L+1 => S / N > L => S > N * L => ceil (S / L) > N => contradicion

                for _ in range(num_chunks):
                    additional_contexts.append(
                        torch.cat([
                            header, 
                            context_ids[:chunk_length]
                        ])
                    )
                    context_ids = context_ids[chunk_length:]

            else:

                s_str = self.tokenizer.decode(sids).replace("<|start_header_id|>user<|end_header_id|>", "[User]").replace("<|start_header_id|>assistant<|end_header_id|>", "[Assistant]").replace(self.tokenizer.eos_token, "\n\n").strip()
                additional_contexts.append(self.tokenizer(s_str, return_tensors='pt', add_special_tokens=False).input_ids[0])

        return additional_contexts

    def get_ids_and_labels_for_conersation(self, messages):

        sentence_ids = self.tokenizer.apply_chat_template(messages,
                                return_tensors="pt")[0][1:]
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

        # TODO: There might be problems when user has too long an input
        
        if sentence_ids.shape[0] > self.max_seq_length:

            all_sentence_ids = [[]]
            sentence_labels = []
            cur_length = 0
            for turn in messages:

                role_token_id = self.tokenizer(turn['role'], add_special_tokens=False).input_ids
                assert len(role_token_id) == 1

                cur_turn_ids = torch.tensor([128006, role_token_id[0], 128007, 271] + self.tokenizer(turn['content'], add_special_tokens=False).input_ids + [128009])

                if cur_length > 0 and cur_length + len(cur_turn_ids) > self.max_length:
                    all_sentence_ids.append([])
                    cur_length = 0
                
                all_sentence_ids[-1].append(cur_turn_ids)

                if turn['role'] == 'assistant':
                    sentence_labels.extend([-100] * 4 + cur_turn_ids[4:].tolist())
                elif turn['role'] == 'user':
                    sentence_labels.extend([-100] * len(cur_turn_ids))
                else:
                    raise ValueError("Invalid role")
                
                cur_length += len(cur_turn_ids)

            # make sure the generation part is a conversation turn
            if len(all_sentence_ids[-1]) % 2 != 0:
                if len(all_sentence_ids[-2][-1]) + sum([len(x) for x in all_sentence_ids[-1]]) > self.max_seq_length:
                    x = all_sentence_ids.pop()
                    all_sentence_ids.append([x.pop(0)])
                    all_sentence_ids.append(x[1:])
                else:
                    all_sentence_ids[-1] = [all_sentence_ids[-2].pop()] + all_sentence_ids[-1]
                
            all_sentence_ids = [x for x in all_sentence_ids if len(x) > 0]

            try:
                additional_contexts = self.cut_contexts(all_sentence_ids[:-1])
            except:
                import ipdb; ipdb.set_trace()

            additional_contexts = self.merge_contexts(additional_contexts)

            sentence_ids = torch.cat(all_sentence_ids[-1])
            sentence_labels = torch.tensor(sentence_labels)[-len(sentence_ids):]

            # assert (self.tokenizer.apply_chat_template(messages[-len(all_sentence_ids[-1]):], return_tensors='pt')[0, 1:] == sentence_ids).all()

        else:

            additional_contexts = []

        # # TODO: delete this
        # # debugging feature
        # all_sentence_ids = []
        # all_sentence_labels = []
        # for turn in messages:
        #     all_sentence_ids += [128006, self.tokenizer(turn['role'], add_special_tokens=False).input_ids[0], 128007, 271] + self.tokenizer(turn['content'], add_special_tokens=False).input_ids + [128009]
        # assert (self.tokenizer.apply_chat_template(messages, return_tensors="pt")[0][1:] == torch.tensor(all_sentence_ids)).all()

        return additional_contexts, sentence_ids, sentence_labels

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
                additional_contexts, sentence_ids, sentence_labels = self.get_ids_and_labels_for_conersation(exp)

            else:
                contexts = []
                additional_contexts, sentence_ids, sentence_labels = self.get_ids_and_labels_for_conersation(exp)

            if len(additional_contexts) > 0 and len(contexts) > 0:
                import ipdb; ipdb.set_trace()

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

    dataset = InstructionTuningDataset(root="../../../data/ift",
                               tokenizer='llama',
                               tokenizer_path="meta-llama/Meta-Llama-3-8B-Instruct",
                            #    files=['redpajama_ift.json'],
                               files=['0_lima.json', 
                                        # '1_alpaca_cleaned.json', 
                                        # '2_chain_of_thought.json', 
                                        # '3_code_alpaca.json', 
                                        # '4_instinwild.json', 
                                        '6_refgpt.json', 
                                        # '7_longalign.json',
                                        # "LongAlpaca-12k-processed.json",
                                        '8_ultrachat_200k.json'
                                        ],
                               add_end_special_token=True,
                               max_length=512,
                               max_seq_length=2048,
                               redpajama_ratio=0.01,
                               redpajama_config={
                                    'root': "../../../data/redpajama",
                                    'tokenizer': 'llama',
                                    'tokenizer_path': "meta-llama/Meta-Llama-3-8B-Instruct",
                                    'snapshots': ['2023-14'],
                                    'shuffle': True,
                                    'partition': 'head_middle',
                                    'add_special_tokens': False,
                                    'target_is_context_ratio': 0.1,
                                    'max_length': 512,
                                    'max_seq_length': 2048,
                                    'num_tokens': 256
                               })
                            #    files=['ift_data_combined.json', "redpajama_ift.json", "LongAlpaca-12k-processed.json"])

    print("len dataset:", len(dataset))
    print("len dataset.data:", len(dataset.data))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    count = 0
    examples = []
    for batch in tqdm(dataloader):

        contexts, input, output, label = batch

        # import ipdb; ipdb.set_trace()

        # if label == 1:
        #     print(dataset.tokenizer.decode(output[0])[-100:])
        #     import ipdb; ipdb.set_trace()

        # if not input.dtype == torch.int64 or not output.dtype == torch.int64:
        #     import ipdb; ipdb.set_trace()
