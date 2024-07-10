from transformers import AutoTokenizer
from MemoryLLM.memoryllm.models.base import BaseMemoryModelPL
from MemoryLLM.memoryllm.modules.memory_llama import *
from MemoryLLM.memoryllm.modules.configuration_llama import LlamaConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import warnings
import torch

class LlamaMemoryModelPL(BaseMemoryModelPL):
    def __init__(self, model_path,
                       num_tokens=None,
                       num_blocks=None,
                       module_name='LlamaMemoryModel',
                       max_length=None,
                       ckpt_path=None,
                       add_mask_token=False,
                       pad_to_max_length=False,
                       add_bos_embedding=False,
                       shrink_to_one_embedding=False,
                       lora_config=None,
                       max_position_embeddings=None,
                       rope_scaling=None,
                       num_memory_tokens=None,
                       drop_memory_per_layer=False,
                       *args,
                       **kwargs):
        
        super(LlamaMemoryModelPL, self).__init__(*args, **kwargs)

        config = LlamaConfig.from_pretrained(model_path)
        config.num_blocks = num_blocks
        config.num_tokens = num_tokens
        config.add_bos_embedding = add_bos_embedding
        config.shrink_to_one_embedding = shrink_to_one_embedding
        config.num_memory_tokens = num_tokens * num_blocks if num_memory_tokens is None else num_memory_tokens
        config.drop_memory_per_layer = drop_memory_per_layer

        if max_position_embeddings is not None:
            config.max_position_embeddings = max_position_embeddings

        if max_length is not None:
            config.max_length = max_length
        else:
            warnings.warn("max_length not provided, setting it to 512")
            max_length = 512

        if rope_scaling is not None:
            if not hasattr(rope_scaling, 'factor'):
                rope_scaling['factor'] =  (config.num_memory_tokens + max_length) / config.max_position_embeddings
                if rope_scaling['factor'] < 1:
                    warnings.warn(f"rope_scaling factor is less than 1, setting it to 1")
                    rope_scaling['factor'] = 1
            config.rope_scaling = rope_scaling

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        config.pad_token_id = self.tokenizer.pad_token_id

        model = eval(module_name).from_pretrained(model_path, config=config)

        if add_mask_token or pad_to_max_length:
            model.resize_token_embeddings(len(self.tokenizer))

        if lora_config is not None:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=lora_config['inference_mode'], 
                r=lora_config['r'], 
                lora_alpha=lora_config['lora_alpha'], 
                lora_dropout=lora_config['lora_dropout'],
                target_modules=lora_config.get('target_modules', None)
            )

            model = get_peft_model(model, peft_config)
            if hasattr(model.base_model, "new_memory_positional_emb"):
                model.base_model.new_memory_positional_emb.requires_grad=True

        self.model = model

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        if max_length is not None:
            self.max_length = max_length
        else:
            self.max_length = 2048
