import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SlimLongDataset(Dataset):
    def __init__(self, 
                 root="local_data/long_data/sampled_1k_long_data_32k_64k.jsonl", 
                 num=1000,
                 tokenizer_path=None,
                 max_length=32768,
                 ):

        self.type = 'longdoc'
        self.root = root
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        with open(self.root, 'r') as f:
            documents = f.readlines()
            self.documents = [json.loads(line)['text'] for line in documents]
        self.documents = self.documents[:num]
        
    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.tokenizer(self.documents[idx], return_tensors="pt").input_ids[:, :self.max_length]

if __name__ == '__main__':
    
    dataset = SlimLongDataset(root="../../../local_data/long_data/sampled_10k_long_data_32k_64k.jsonl", tokenizer_path='meta-llama/Meta-Llama-3.1-8B')
    print(dataset[0].shape)