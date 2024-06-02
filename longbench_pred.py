import os
from datasets import load_dataset
from datasets import load_from_disk
import torch
import nltk
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from modeling_memoryllm import MemoryLLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["memoryllm-7b", "memory-openllama-3b", "longlora-7b-16k", "longllama-3b", "longllama-3b-v2", "openllama-3b-2k", "openllama-3b-v2-2k", "llama2-7b-4k", "llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--max_length", default=None, type=int)
    parser.add_argument("--split_model", default=False, action='store_true')
    parser.add_argument("--part", default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--retrieval', default=None, help="Retrieval method", type=str)
    parser.add_argument('--exclude_or', default=False, action='store_true')
    parser.add_argument('--force_run', default=False, action='store_true')
    return parser.parse_known_args(args)[0]

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name and 'chat' in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, retrieval=None, exclude_or=False):

    preds = []

    if 'memory' in model_name:
        backup_memory = model.memory.clone().detach().cpu()

    count = 0
    for json_obj in tqdm(data):

        count += 1
        # if count == 5: break

        if exclude_or:
            if "or" in json_obj['input']:
                continue

        if 'memory' in model_name:
            model.memory.data = backup_memory.clone().detach().to(device)

        prompt = prompt_format.format(**json_obj)

        if retrieval is None:
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        else:
            # if "Question" in prompt.split("\n\n")[-1]:
            #     tokenized_prompt = tokenizer("\n\n".join(prompt.split("\n\n")[:-1]), truncation=False, return_tensors="pt").input_ids[0]
            # else:
            #     tokenized_prompt = tokenizer("\n\n".join(prompt.split("\n\n")[:-2]), truncation=False, return_tensors="pt").input_ids[0]
            prompt_context = prompt.split(prompt_format.split("{context}")[-1].split("Question")[0])[0]
            tokenized_prompt = tokenizer(prompt_context, truncation=False, return_tensors="pt").input_ids[0]

        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
        if max_length > 0 and len(tokenized_prompt) > max_length:

            # half = int(max_length/2)
            # prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

            if retrieval is None:
                # truncate at the beginning:
                prompt = tokenizer.decode(tokenized_prompt[-(max_length - max_gen):], skip_special_tokens=True)
            
            else:
                tokenized_prompt = tokenizer(
                    tokenizer.decode(tokenized_prompt[-(max_length - max_gen):], skip_special_tokens=True),
                    truncation=False, return_tensors="pt", add_special_tokens=False
                ).input_ids[0]

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            input = prompt

        elif 'memory' in model_name:

            contexts_ids = []

            if dataset == 'gov_report' or dataset == 'multi_news' or dataset == 'qmsum':
                sentence = tokenizer("\n\n".join(prompt.split("\n\n")[-2:]), add_special_tokens=False).input_ids
                prompt_ids = tokenizer(prompt.replace("\n\n".join(prompt.split("\n\n")[-2:]), "").strip(), add_special_tokens=False, truncation=False).input_ids
                while len(prompt_ids) > 0:
                    if contexts_ids == []:
                        contexts_ids.append(prompt_ids[-512:])
                        prompt_ids = prompt_ids[:-512]
                    else:
                        contexts_ids.append(prompt_ids[-512:])
                        prompt_ids = prompt_ids[:-512]
                contexts_ids = contexts_ids[::-1]
                contexts_ids = [torch.tensor(context_ids).to(device) for context_ids in contexts_ids]
                sentence = torch.tensor(sentence).to(device)

            else:
                if retrieval is not None:
                    parts = []
                    for i in range(0, len(tokenized_prompt), 512):
                        parts.append(tokenized_prompt[i:i+512])
                    parts = [tokenizer.decode(part) for part in parts]
                    
                    # if "Question" in prompt.split("\n\n")[-1]:
                    #     query = prompt.split("\n\n")[-1]
                    # else:
                    #     query = "\n\n".join(prompt.split("\n\n")[-2:])
                    query = prompt[len(prompt_context):]
            
                    # retriever = BM25Retriever.from_texts(parts)
                    # retrieve the context from pre_prompt:
                    retriever = BM25Retriever.from_documents([Document(page_content=part) for part in parts])

                    result = retriever.get_relevant_documents(json_obj['input'], top_k=8)
                    result = " ".join([r.page_content for r in result])

                    prompt = result + query

                prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids
                while len(prompt_ids) > 0:
                    if contexts_ids == []:
                        contexts_ids.append(prompt_ids[-(512-max_gen):])
                        prompt_ids = prompt_ids[:-(512-max_gen)]
                    else:
                        contexts_ids.append(prompt_ids[-512:])
                        prompt_ids = prompt_ids[:-512]

                contexts_ids = contexts_ids[::-1]
                contexts_ids = [torch.tensor(context_ids).to(device) for context_ids in contexts_ids]
                
                sentence = contexts_ids[-1]
                contexts_ids = contexts_ids[:-1]

        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            raise NotImplementedError
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:

            if 'memory' in model_name:

                with torch.no_grad():

                    for context in contexts_ids:
                        model.inject_memory(
                            context.unsqueeze(0).to(device),
                            torch.ones(context.shape[0] + model.num_tokens).long().unsqueeze(0).to(device),
                            update_memory=True
                        )
                    
                    context_length = sentence.shape[0]
                    output = model.generate(
                        input_ids=sentence.unsqueeze(0).cuda(),
                        attention_mask=torch.ones(sentence.shape[0] + model.num_blocks * model.num_tokens).unsqueeze(0).long().cuda(),
                        max_new_tokens=max_gen,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                    )[0]

            else:

                context_length = input.input_ids.shape[-1]
                    
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif model_name == 'memoryllm-7b':
        tokenizer = LlamaTokenizer.from_pretrained(path)
        if args.split_model:
            model = MemoryLLM.from_pretrained(path, device_map='auto')
        else:
            model = MemoryLLM.from_pretrained(path).to(device)
    elif "llama2" in model_name or 'openllama' in model_name:
        # replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path).to(device)
    elif "longllama" in model_name:
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(device)
        tokenizer = LlamaTokenizer.from_pretrained(path)
    elif "longlora" in model_name:
        # path = Yukang/Llama-2-7b-longlora-100k-ft
        tokenizer = AutoTokenizer.from_pretrained(path)
        if args.split_model:
            model = AutoModelForCausalLM.from_pretrained(path, device_map='auto', torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(path).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    model2path = json.load(open("longbench_config/model2path.json", "r"))
    model2maxlen = json.load(open("longbench_config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    if args.path is not None:
        model2path[model_name] = args.path
    if args.max_length is not None:
        model2maxlen[model_name] = args.max_length
        print("Override max length")
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        if args.dataset is None:
            datasets = ["hotpotqa", "narrativeqa", "qasper", "multifieldqa_en", "2wikimqa", "musique"]
        else:
            datasets = [args.dataset]
         
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("longbench_config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("longbench_config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists(f"longbench/pred_seed{args.seed}"):
        os.makedirs(f"longbench/pred_seed{args.seed}")
    if not os.path.exists(f"longbench/pred_seed{args.seed}_e"):
        os.makedirs(f"longbench/pred_seed{args.seed}_e")
    
    for dataset in datasets:

        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            data.save_to_disk(f"longbench/data_e/{dataset}")

            if not os.path.exists(f"longbench/pred_seed{args.seed}_e/{model_name}"):
                os.makedirs(f"longbench/pred_seed{args.seed}_e/{model_name}")
            out_path = f"longbench/pred_seed{args.seed}_e/{model_name}/{dataset}.jsonl"
            # save dataset

        else:
            # data = load_from_disk(f"longbench/data/{dataset}")
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            # data.save_to_disk(f"longbench/data/{dataset}")

            if args.path is None:
                if args.max_length is None:
                    if not os.path.exists(f"longbench/pred_seed{args.seed}/{model_name}"):
                        os.makedirs(f"longbench/pred_seed{args.seed}/{model_name}")
                    out_path = f"longbench/pred_seed{args.seed}/{model_name}/{dataset}.jsonl"
                else:
                    if not os.path.exists(f"longbench/pred_seed{args.seed}/{model_name}_{args.max_length}"):
                        os.makedirs(f"longbench/pred_seed{args.seed}/{model_name}_{args.max_length}")
                    out_path = f"longbench/pred_seed{args.seed}/{model_name}_{args.max_length}/{dataset}.jsonl"
            else:
                if not os.path.exists(f"longbench/pred_seed{args.seed}/{os.path.basename(args.path)}_{args.max_length}"):
                    os.makedirs(f"longbench/pred_seed{args.seed}/{os.path.basename(args.path)}_{args.max_length}")
                out_path = f"longbench/pred_seed{args.seed}/{os.path.basename(args.path)}_{args.max_length}/{dataset}.jsonl"

        if args.exclude_or:
            out_path = out_path.split("/")
            out_path[-2] += '_exor'
            out_path = "/".join(out_path)

            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))

        if os.path.exists(out_path) and not args.force_run:
            continue

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(model, tokenizer, data, args.max_length, max_gen, prompt_format, dataset, device, model_name, args.retrieval, exclude_or=args.exclude_or)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')