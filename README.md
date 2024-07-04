# MemoryLLM: Towards Self-Updatable Large Language Models

This is the official code for the paper: **MemoryLLM: Towards Self-Updatable Large Language Models**.   
The model is open-sourced at https://huggingface.co/YuWangX/memoryllm-7b

## Environment Setup
```
conda create --name memoryllm
conda activate memoryllm
pip install -r requirements.txt
```

## Load Model
First clone the repository and get into the repository: 
```
git clone git@github.com:wangyu-ustc/MemoryLLM.git
cd MemoryLLM
```
Then simply use the following code to load the model:
```
from modeling_memoryllm import MemoryLLM
from configuration_memoryllm import MemoryLLMConfig
from transformers import LlamaTokenizer
config = MemoryLLMConfig.from_pretrained("YuWangX/memoryllm-7b")
model = MemoryLLM.from_pretrained("YuWangX/memoryllm-7b")
tokenizer = LlamaTokenizer.from_pretrained("YuWangX/memoryllm-7b")
```

## How to use the model
Inject a piece of context into the model using the following script:
```
model = model.cuda()

# Self-Update with the new context
context = "David likes eating apples."
model.inject_memory(tokenizer(context, return_tensors='pt', add_special_tokens=False).input_ids.cuda(), update_memory=True)

# Generation
import torch
input_ids = tokenizer("What fruits does David like? Answer:", return_tensors='pt', add_special_tokens=False).input_ids
attention_mask = torch.cat([
    torch.ones(input_ids.shape[0], model.num_tokens * model.num_blocks),
    torch.ones_like(input_ids)
], dim=1)
outputs = model.generate(inputs=input_ids.cuda(), attention_mask=attention_mask.cuda(), max_new_tokens=10)
print(tokenizer.decode(outputs[0]))
```

## Evaluation

### Model Editing Evaluations
We put our reimplementation of various model-editing baselines and `MemoryLLM` in the repo [EditingLlama](https://github.com/wangyu-ustc/EditingLlama). 

### Customized Experiments
To prepare the dataset, please download from [here](YuWangX/KnowledgeRetention). Please download the dataset and put them as the following structure: 
```
- data
  - squad
    - indices_squad_3.npy
    - dev-v2.0.json
    - train-v2.0.json
  - nq 
    - indices_nq_4.npy
    - v1.0-simplified_nq-dev-all.jsonl
    - v1.0-simplified_simplified-nq-train.jsonl
```
We will evaluate our model on the validation set where the unrelated contexts are sampled from the training set. To evaluate the model, we could use the following script: 

```
mkdir results
python test_qa_memory.py --model YuWangX/memoryllm-7b --nuc 10 --datasets naturalqa squad --num_samples 100
```
here `nuc` means the the number of irrelevant contexts, and `naturalqa squad` means the datasets to evaluate the model on.

### Evaluation on Longbench

```
python longbench_pred.py --model memoryllm-7b --datasets hotpotqa --max_length 12384
```
Here `max_length` is the maximum length used when truncating the context.
Then the generated results are all saved in the folder `longbench` for evaluation.


## Training
In our implementations, we train Llama2-7B on C4 dataset. However, this may lead to the poor performance on the benchmark `qasper` (see Figure 4 in the [paper](https://arxiv.org/pdf/2402.04624)). Thus we put the script of training on red-pajama here, which is the dataset we have been using in the models we are currently exploring. 

Please check the folder `train` using the following command:
```
cd train
```
### Dataset Preparation
Please follow the instructions below to prepare the datasets: (make sure you have the datasets from [here](https://github.com/wangyu-ustc/MemoryLLM?tab=readme-ov-file#customized-experiments) prepared.)
```
cd data

# Please use the softlink to link the validation datasets into the current directory.
ln -s ../../data/nq ./
ln -s ../../data/squad ./

# Then please download the redpajama dataset
cd redpajama
sh download.sh
```

After preparing all the datasets, you can run the following code to start training:
```
python main.py -t --base MemoryLLM/configs/llama/llama_30x256.yaml
```
We have not conducted training on openllama but we do have the script on openllama for debugging purposes. So if you want to see the training on openllama, please run the following command:
```
python main.py -t --base MemoryLLM/configs/openllama/openllama_4x256.yaml
```

## Citations
If you find this repo helpful, please consider cite our paper:
```
@article{wang2024memoryllm,
  title={MEMORYLLM: Towards Self-Updatable Large Language Models},
  author={Wang, Yu and Chen, Xiusi and Shang, Jingbo and McAuley, Julian},
  journal={arXiv preprint arXiv:2402.04624},
  year={2024}
}
```
