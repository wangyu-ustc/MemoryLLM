from MemoryLLM.memoryllm.models.base import BaseMemoryModelPL
from MemoryLLM.memoryllm.modules.memory_llama import *
from transformers import AutoTokenizer, LlamaConfig
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import warnings

class LlamaMemoryModelPL(BaseMemoryModelPL):
    def __init__(self, model_path,
                       num_tokens=None,
                       num_blocks=None,
                       tokenizer_path=None,
                       module_name='LlamaMemoryModel',
                       max_length=None,
                       ckpt_path=None,
                       add_bos_embedding=False,
                       new_memory_embedding_fullset=True,
                       shrink_to_one_embedding=False,
                       lora_config=None,
                       max_position_embeddings=None,
                       add_decoder_lora=False,
                       initialize_decoder_lora_from_default=False,
                       rope_scaling=None,
                       num_memory_tokens=None,
                       drop_memory_per_layer=False,
                       max_seq_length=None,
                       max_seq_length_when_detaching_memory=None,
                       gradient_checkpointing=False,
                       fix_encoder=False,
                    #    instruction=None,
                       tune_special_tokens=False,
                       special_token_ids=None,
                       num_ltm_blocks=None,
                       update_ltm_frequency=None,
                       update_ltm_from=None,
                       update_ltm_num_tokens=None,
                       converge_ltm_number_tokens=None,
                       attn_implementation=None,
                       add_selector=False,
                       selector_hidden_dim=1024,
                       map_from_hidden_states=True,
                       detach_hidden_state=False,
                       reinit_memory=False,
                       num_selector_layers=None,
                       add_decoder_selector=False,
                       selector_layers=None,
                       add_encoder_retriever=False,
                       maintain_memory_keys=False,
                       initial_rf_when_moving_stm_to_ltm=None,
                       decay_frequency=None,
                       dropping_interval=None,
                       add_memory_embedding=False,
                       spread_embeddings=False,
                       min_num_tokens=25,
                       important_tokens='right',
                       update_ltm_mode='decouple',
                       ltm_configs=None,
                       put_cached_dropped_memory_on_cpu=None,
                       initialized=False,
                       wrap_memory=False,
                       all_params_require_grad=False,
                       fix_poe_for_encoder=False,
                       virtual_num_blocks=None,
                       *args,
                       **kwargs):
        
        super(LlamaMemoryModelPL, self).__init__(*args, **kwargs)

        config = LlamaConfig.from_pretrained(model_path)
        config.num_blocks = num_blocks
        config.num_tokens = num_tokens
        config.add_bos_embedding = add_bos_embedding
        config.new_memory_embedding_fullset = new_memory_embedding_fullset
        config.shrink_to_one_embedding = shrink_to_one_embedding
        config.num_memory_tokens = num_tokens * num_blocks if num_memory_tokens is None else num_memory_tokens
        config.drop_memory_per_layer = drop_memory_per_layer
        config.add_decoder_lora = add_decoder_lora
        config.add_encoder_selector = add_decoder_selector
        config.tune_special_tokens = tune_special_tokens
        config.add_selector = add_selector
        config.detach_hidden_state = detach_hidden_state
        config.maintain_memory_keys = maintain_memory_keys
        config.min_num_tokens = min_num_tokens
        config.important_tokens = important_tokens
        config.wrap_memory = wrap_memory
        config.fix_poe_for_encoder = fix_poe_for_encoder
        
        if virtual_num_blocks is not None:
            config.virtual_num_blocks = virtual_num_blocks
        
        if put_cached_dropped_memory_on_cpu is not None:
            config.put_cached_dropped_memory_on_cpu = put_cached_dropped_memory_on_cpu
        
        self.important_tokens = important_tokens
        
        if add_memory_embedding is not None:
            config.add_memory_embedding = add_memory_embedding
            config.spread_embeddings = spread_embeddings

        self.selector_layers = None
        if add_selector:
            config.selector_hidden_dim = selector_hidden_dim
            config.map_from_hidden_states = map_from_hidden_states
            if num_selector_layers is not None:
                config.num_selector_layers = num_selector_layers
            if selector_layers is not None:
                config.selector_layers = list(selector_layers)
                self.selector_layers = torch.tensor(list(selector_layers))
            config.add_encoder_retriever = add_encoder_retriever
            self.add_encoder_retriever = add_encoder_retriever

        if special_token_ids is not None:
            config.special_token_ids = list(special_token_ids)
        else:
            config.special_token_ids = None
        if num_ltm_blocks is not None:
            config.num_ltm_blocks = num_ltm_blocks
        if update_ltm_frequency is not None:
            config.update_ltm_frequency = update_ltm_frequency
        if update_ltm_num_tokens is not None:
            config.update_ltm_num_tokens = update_ltm_num_tokens
        if update_ltm_from is not None:
            config.update_ltm_from = update_ltm_from
        if converge_ltm_number_tokens is not None:
            config.converge_ltm_number_tokens = converge_ltm_number_tokens
        if initial_rf_when_moving_stm_to_ltm is not None:        
            config.initial_rf_when_moving_stm_to_ltm = initial_rf_when_moving_stm_to_ltm
        if decay_frequency is not None:
            config.decay_frequency = decay_frequency
        config.update_ltm_mode = update_ltm_mode
        if ltm_configs is not None:
            config.ltm_configs = dict(ltm_configs)

        # for tree-structured dropping
        if dropping_interval is not None:
            config.dropping_interval = dropping_interval

        self.add_selector = add_selector
        self.initialize_decoder_lora_from_default = initialize_decoder_lora_from_default

        if max_position_embeddings is not None:
            config.max_position_embeddings = max_position_embeddings

        if max_length is not None:
            max_length = max_length
        else:
            warnings.warn("max_length not provided, setting it to 512")
            max_length = 512
        
        config.max_length = max_length

        if max_seq_length is not None:
            config.max_seq_length = max_seq_length
        else:
            config.max_seq_length = max_length
        
        self.max_seq_length = config.max_seq_length
        self.max_seq_length_when_detaching_memory = max_seq_length_when_detaching_memory if max_seq_length_when_detaching_memory is not None else self.max_seq_length

        if rope_scaling is not None:
            if not hasattr(rope_scaling, 'factor'):
                rope_scaling['factor'] =  (config.num_memory_tokens + self.max_seq_length_when_detaching_memory) / config.max_position_embeddings
                if rope_scaling['factor'] < 1:
                    warnings.warn(f"rope_scaling factor is less than 1, setting it to 1")
                    rope_scaling['factor'] = 1
                else:
                    print(f"rope_scaling factor is set to {rope_scaling['factor']}, i.e. max length:", config.num_memory_tokens + config.max_seq_length)
            config.rope_scaling = dict(rope_scaling)

        if attn_implementation is not None:
            config._attn_implementation = attn_implementation

        self.tokenizer = AutoTokenizer.from_pretrained(model_path if tokenizer_path is None else tokenizer_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # if instruction is not None:
        #     self.instruction_ids = self.tokenizer(instruction, return_tensors='pt', add_special_tokens=False)
        # else:
        #     self.instruction_ids = None

        config.bos_token_id = self.tokenizer.bos_token_id
        config.pad_token_id = self.tokenizer.pad_token_id

        if lora_config is not None:
            config.lora_config = dict(lora_config)
            config.lora_config['target_modules'] = list(config.lora_config['target_modules'])

        model = eval(module_name).from_pretrained(model_path, config=config)

        if tune_special_tokens:
            for token_id in special_token_ids:
                model.special_token_embeddings.data[model.special_token_ids.index(token_id)] = model.model.embed_tokens.weight.data[token_id]
        
        if add_bos_embedding:
            model.bos_embedding.data[:, :] = model.model.embed_tokens.weight.data[model.bos_token_id]

        if gradient_checkpointing:
            model.model.gradient_checkpointing = True
            model.model.gradient_checkpointing_enable()

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

            if add_decoder_lora:
                get_peft_model(model, peft_config, adapter_name="decoder_adapter")
                if not fix_encoder:
                    for name, param in model.named_parameters():
                        if "lora" in name:
                            param.requires_grad = True
                
        if hasattr(model.base_model, "new_memory_positional_emb"):
            model.base_model.new_memory_positional_emb.requires_grad=True
        if hasattr(model.base_model, "memory_token_start_indication_embedding"):
            model.base_model.memory_token_start_indication_embedding.requires_grad=True
        if hasattr(model.base_model, "bos_embedding"):
            model.base_model.bos_embedding.requires_grad=True
        if hasattr(model.base_model, "special_token_embeddings"):
            model.base_model.special_token_embeddings.requires_grad=True
        if add_memory_embedding:
            model.base_model.memory_embeddings.requires_grad=True
        if self.add_selector:
            for name, param in model.named_parameters():
                if "query_proj" in name or "key_proj" in name:
                    param.requires_grad = True

        if all_params_require_grad:
            for name, param in model.named_parameters():
                param.requires_grad = True

        self.model = model

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        if reinit_memory:
            self.model.initialized.data -= self.model.initialized.item()

        if initialized:
            if not self.model.initialized:
                self.model.initialized += 1

        if max_length is not None:
            self.max_length = max_length
        else:
            self.max_length = 2048
