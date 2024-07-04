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
                       add_positional_embedding=False,
                       add_bos_embedding=False,
                       add_memory_token=False,
                       add_pad_token=False, # This parameter is already deprecated and not useful any more
                       split_positional_embedding=False,
                       new_memory_embedding_fullset=True,
                       shrink_to_one_embedding=False,
                       put_memory_on_cpu=False,
                       delta_memory_ratio=1.0,
                       lora_config=None,
                       new_delta_memory_length=None,
                       drop_memory_token=False,
                       share_position_ids=False,
                       use_negative_position_ids=False,
                       interpolate_position_ids=False,
                       max_position_embeddings=None,
                       additional_transformer=False,
                       bos_embedding_file=None,
                       rope_scaling=None,
                       num_memory_tokens=None,
                       split_encoder_decoder=False,
                       only_update_layer=None,
                       save_last_layer=False,
                       reuse_memory_for_every_layer=False,
                       half_layers=False,
                       drop_memory_per_layer=False,
                       *args,
                       **kwargs):
        
        super(LlamaMemoryModelPL, self).__init__(*args, **kwargs)

        config = LlamaConfig.from_pretrained(model_path)
        config.num_blocks = num_blocks
        config.num_tokens = num_tokens
        config.add_positional_embedding = add_positional_embedding
        config.pad_to_max_length = pad_to_max_length
        config.add_bos_embedding = add_bos_embedding
        config.add_pad_token = add_pad_token
        config.split_positional_embedding = split_positional_embedding
        config.new_memory_embedding_fullset = new_memory_embedding_fullset
        config.shrink_to_one_embedding = shrink_to_one_embedding
        config.put_memory_on_cpu = put_memory_on_cpu
        config.delta_memory_ratio = delta_memory_ratio
        config.new_delta_memory_length = new_delta_memory_length if new_delta_memory_length is not None else np.ceil(num_tokens if num_tokens is not None else num_memory_tokens / config.num_hidden_layers)
        config.drop_memory_token = drop_memory_token
        config.share_position_ids = share_position_ids
        config.use_negative_position_ids = use_negative_position_ids
        config.interpolate_position_ids = interpolate_position_ids
        config.additional_transformer = additional_transformer
        config.num_memory_tokens = num_tokens * num_blocks if num_memory_tokens is None else num_memory_tokens
        config.split_encoder_decoder = split_encoder_decoder
        config.only_update_layer = only_update_layer
        config.save_last_layer = save_last_layer
        config.reuse_memory_for_every_layer = reuse_memory_for_every_layer
        config.half_layers = half_layers
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
        if add_mask_token:
            self.tokenizer.add_special_tokens({'mask_token': '<mask>'})
        
        if pad_to_max_length or add_pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        else:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        config.pad_token_id = self.tokenizer.pad_token_id

        model = eval(module_name).from_pretrained(model_path, config=config)

        if bos_embedding_file is not None:
            # bos_embedding_file = "/fsx-Training/shopqa-training-fsx-prod-us-east-1/wangyuu/transformer_models/llama2-7b/llama2_bos.pt"
            model.bos_embedding.data = torch.load(bos_embedding_file)
            
        if add_mask_token or pad_to_max_length:
            # assert lora_config is None, "resize token embeddings not supported with lora"
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
                # model.base_model.new_memory_positional_emb.weight.requires_grad=True
                model.base_model.new_memory_positional_emb.requires_grad=True
            if hasattr(model.base_model, "memory_token_start_indication_embedding"):
                model.base_model.memory_token_start_indication_embedding.requires_grad=True
            if hasattr(model.base_model, 'transformer_layer'):
                for param in model.base_model.transformer_layer.parameters():
                    param.requires_grad=True
            if hasattr(model.base_model, 'gate') and model.base_model.gate is not None:
                for param in model.base_model.gate.parameters():
                    param.requires_grad=True
            if add_mask_token or add_memory_token:
                model.base_model.model.model.embed_tokens.requires_grad=True

        self.model = model

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        if max_length is not None:
            self.max_length = max_length
        else:
            self.max_length = 2048
