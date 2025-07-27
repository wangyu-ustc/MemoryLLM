import os
import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import copy
import torch
from tqdm import tqdm
import deepspeed
import pytorch_lightning as pl
from collections import OrderedDict
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import IterableDataset

from MemoryLLM.memoryllm.util import instantiate_from_config, collate_fn_qa

os.environ["WANDB_CONFIG_DIR"] = os.path.join(os.getcwd (),"tmp")
os.makedirs(os.path.join(os.getcwd(), "tmp"), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "wandb"), exist_ok=True)

import wandb
wandb.login(key="your_wandb_key")


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        '--n_train_steps',
        default=10000,
        type=int
    )
    parser.add_argument(
        '--num_samples',
        default=None,
        type=int
    )
    parser.add_argument(
        '--backup_memory',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--related_position',
        default='begin',
        type=str
    )
    parser.add_argument(
        '--nuc',
        default=9,
        type=int
    )
    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)

def worker_init_fn_mixed(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset # the dataset copy in this worker process
    dataset.set_worker(worker_id, worker_info.num_workers)

def worker_init_fn_ift(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    all_dataset = len(dataset.redpajama_dataset.all_paths)
    per_worker = all_dataset // worker_info.num_workers
    dataset.redpajama_dataset.all_paths = dataset.redpajama_dataset.all_paths[worker_id * per_worker: (worker_id + 1) * per_worker]

def worker_init_fn_redpajama(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset # the dataset copy in this worker process
    all_dataset = len(dataset.all_paths)
    per_worker = all_dataset // worker_info.num_workers
    dataset.all_paths = dataset.all_paths[worker_id * per_worker: (worker_id + 1) * per_worker]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, num_tokens, eval_batch_size=None, eval_max_length=None, run_qa=False, add_special_tokens=False, end_special_token=None, 
                 train=None, validation=None, test=None, predict=None, wrap=False, num_workers=None, 
                 shuffle_test_loader=False, use_worker_init_fn=False, mask_strategy=None, mask_ratio=0.0,
                 shuffle_val_dataloader=False, pad_to_max_length=False, add_mask_token=False, add_pad_token=False, 
                 add_memory_token=False, collate_fn='none', worker_init_fn="worker_init_fn_redpajama", is_ift=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.add_special_tokens = add_special_tokens
        self.end_special_token = end_special_token
        self.pad_to_max_length = pad_to_max_length
        self.add_pad_token = add_pad_token
        self.add_memory_token = add_memory_token
        self.run_qa = run_qa
        self.mask_strategy = mask_strategy
        self.mask_ratio = mask_ratio
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.worker_init_fn = worker_init_fn
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.add_mask_token = add_mask_token or mask_ratio > 0
        # TODO: not sure if max_length can be accessed
        self.eval_max_length = self.datasets['train'].max_length if eval_max_length is None else eval_max_length
        self.collate_fn = collate_fn
        self.is_ift = is_ift

        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        print("dataset loaded")
        self.wrap = wrap

    def prepare_data(self):
        print("preparing data for no reason")
        for k, data_cfg in self.dataset_configs.items():
            if k != 'validation':
                instantiate_from_config(data_cfg)
            else:
                for x in data_cfg:
                    instantiate_from_config(x)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k])) if k != 'validation' else
            (k, [instantiate_from_config(x) for x in self.dataset_configs[k]])
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        if self.use_worker_init_fn:
            init_fn = eval(self.worker_init_fn)
        else:
            init_fn = None

        if self.add_mask_token:
            if self.datasets['train'].tokenizer.mask_token is None:
                self.datasets['train'].tokenizer.add_special_tokens({'mask_token': '<mask>'})
        
        if self.add_memory_token: 
            if not hasattr(self.datasets['train'].tokenizer, 'memory_token_id'):
                self.datasets['train'].tokenzer.add_tokens("<mem>")
                self.datasets['train'].tokenzer.memory_token_id = self.datasets['train'].tokenzer.convert_tokens_to_ids('<mem>')

        if self.add_pad_token:
            if self.datasets['train'].tokenizer.pad_token is None or self.datasets['train'].tokenizer.pad_token == self.datasets['train'].tokenizer.eos_token:
                self.datasets['train'].tokenizer.add_special_tokens({'pad_token': '<pad>'})
        else:
            self.datasets['train'].tokenizer.pad_token = self.datasets['train'].tokenizer.eos_token

        if self.collate_fn != 'none':
            collate_fn_with_params = partial(eval(self.collate_fn), 
                tokenizer=self.datasets['train'].tokenizer, 
                max_length=self.datasets['train'].max_length, 
                padding='max_length' if self.pad_to_max_length else 'longest',
                num_tokens=self.num_tokens,
                add_special_tokens=self.add_special_tokens,
                end_special_token=self.end_special_token,
                mask_strategy=self.mask_strategy,
                mask_ratio=self.mask_ratio)
        else:
            collate_fn_with_params = None

        if isinstance(self.datasets['train'], IterableDataset):
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            worker_init_fn=init_fn, collate_fn=collate_fn_with_params)
        else:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=True,
                            worker_init_fn=init_fn, collate_fn=collate_fn_with_params)

    def _val_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        if self.mask_ratio > 0:
            if self.datasets['train'].tokenizer.mask_token is None:
                self.datasets['train'].tokenizer.add_special_tokens({'mask_token': '<mask>'})
        
        if self.add_pad_token:
            if self.datasets['train'].tokenizer.pad_token is None or self.datasets['train'].tokenizer.pad_token == self.datasets['train'].tokenizer.eos_token:
                self.datasets['train'].tokenizer.add_special_tokens({'pad_token': '<pad>'})
        else:
            self.datasets['train'].tokenizer.pad_token = self.datasets['train'].tokenizer.eos_token

        if self.add_memory_token:
            if not hasattr(self.datasets['train'].tokenizer, 'memory_token_id'):
                self.datasets['train'].tokenizer.add_tokens("<mem>")
                self.datasets['train'].tokenizer.memory_token_id = self.datasets['train'].tokenizer.convert_tokens_to_ids('<mem>')

        collate_fn_with_params = partial(collate_fn_qa, 
            tokenizer=self.datasets['train'].tokenizer, 
            max_length=self.datasets['train'].max_length,
            eval_max_length=self.eval_max_length, 
            num_tokens=self.num_tokens,
            padding='max_length' if self.pad_to_max_length else 'longest',
            add_special_tokens=self.add_special_tokens,
            end_special_token=self.end_special_token,
            is_ift=self.is_ift)

        if isinstance(self.datasets["validation"], list):            
            dataloaders = []
            for dataset in self.datasets['validation']:
                if dataset.type == 'cqa':
                    dataloaders.append(
                        DataLoader(dataset, batch_size=self.eval_batch_size,
                                num_workers=self.num_workers,
                                worker_init_fn=init_fn,
                                shuffle=shuffle, collate_fn=collate_fn_with_params),
                    )
                else:
                    dataloaders.append(
                        DataLoader(dataset, batch_size=self.eval_batch_size,
                                num_workers=self.num_workers,
                                worker_init_fn=init_fn,
                                shuffle=shuffle),
                    )

            return dataloaders

        else:
            return DataLoader(self.datasets["validation"],
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            worker_init_fn=init_fn,
                            shuffle=shuffle,
                            collate_fn=collate_fn_with_params)


    def _test_dataloader(self, shuffle=False):
        if self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=None)



class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            # trainer.save_checkpoint(ckpt_path)
            trainer.model.model.save_pretrained(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            pass
            # # ModelCheckpoint callback created log directory --- remove it
            # if not self.resume and os.path.exists(self.logdir):
            #     dst, name = os.path.split(self.logdir)
            #     dst = os.path.join(dst, "child_runs", name)
            #     os.makedirs(os.path.split(dst)[0], exist_ok=True)
            #     try:
            #         os.rename(self.logdir, dst)
            #     except FileNotFoundError:
            #         pass



class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()

    # 1.9.0:
    # parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            if "checkpoints" in opt.resume:
                logdir = opt.resume[:opt.resume.index("checkpoints")]
                ckpt = opt.resume
            else:
                paths = opt.resume.split("/")
                # idx = len(paths)-paths[::-1].index("logs")+1
                # logdir = "/".join(paths[:idx])
                logdir = "/".join(paths[:-3])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume

            if "checkpoints" in opt.resume:
                logdir = opt.resume[:opt.resume.index("checkpoints")]
                ckpt = opt.resume
            else:
                logdir = opt.resume.rstrip("/")
                ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]

    else:
        
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        # trainer_config["accelerator"] = "cuda"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)

        if not "accelerator" in trainer_config or trainer_config['accelerator'] == 'cpu':
            cpu = True
        else:
            cpu = False

        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        if (not opt.train) and (opt.resume is not None):
            if 'ckpt_path' in config['model']['params']:
                del config['model']['params']['ckpt_path'] 

        # model
        config.model.params['backup_memory_when_validating'] = opt.backup_memory
        config.model.params['related_position_when_validation'] = opt.related_position
        model = instantiate_from_config(config.model)

        if not opt.train:
            
            if os.path.isdir(ckpt):
                # model.load_from_checkpoint
                if os.path.exists(ckpt + "/checkpoint/mp_rank_00_model_states.pt"):
                    ckpt += "/checkpoint/mp_rank_00_model_states.pt"
                    def rename_keys(state_dict):
                        new_state_dict = OrderedDict()
                        for key, value in state_dict.items():
                            new_key = key.replace("_forward_module.", "")
                            new_state_dict[new_key] = value
                        return new_state_dict
                    model.load_state_dict(rename_keys(torch.load(ckpt, map_location='cpu')['module']))
                else:
                    model.model = model.model.from_pretrained(ckpt)

            else:
                model.load_state_dict(torch.load(ckpt)['state_dict'], strict=False)

            for validation_param in config.data.params.validation:
                validation_param.params['num'] = opt.num_samples
                validation_param.params['num_unrelated_contexts'] = opt.nuc

            config.data.params.train.params['name'] = 'debug'
            data = instantiate_from_config(config.data)
            data.setup()
            print("#### Data #####")
            for k in data.datasets:
                print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
            # trainer = Trainer.from_argparse_args(trainer_opt)
            trainer_kwargs = vars(trainer_opt)
            trainer = Trainer(**{
                'accelerator': trainer_kwargs['accelerator'],
                'auto_select_gpus': trainer_kwargs['auto_select_gpus'],
                'resume_from_checkpoint': trainer_kwargs['resume_from_checkpoint'],
            })

            results = trainer.validate(model, dataloaders=data.val_dataloader())

            if not os.path.exists('/fsx-Training/shopqa-training-fsx-prod-us-east-1/wangyuu/qa_results'):
                os.makedirs('/fsx-Training/shopqa-training-fsx-prod-us-east-1/wangyuu/qa_results')

            import json
            with open(os.path.join('/fsx-Training/shopqa-training-fsx-prod-us-east-1/wangyuu/qa_results', opt.resume.replace("/", "_").replace(".", "_").strip("_") + 'results.json'), 'w') as f:
                json.dump(results, f)
            
            sys.exit(0)
        
        # trainer and callbacks
        trainer_kwargs = dict()

        os.makedirs(logdir, exist_ok=True)
        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "project": "MemoryLLM",
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        if "debug" in opt.base[0]:
            default_logger_cfg = default_logger_cfgs["testtube"]
        else:
            default_logger_cfg = default_logger_cfgs["wandb"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            # "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "target": 'MemoryLLM.memoryllm.util.ModelCheckpointLLM',
            "params": {
                "dirpath": ckptdir,
                "filename": "{step:09}",
                "verbose": True,
                "save_last": True,
                # "every_n_train_steps": 10
            }
        }

        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3
            if 'acc' in model.monitor:
                default_modelckpt_cfg["params"]["mode"] = "max"
            else:
                default_modelckpt_cfg["params"]["mode"] = "min"

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {
                        # "target": 'MemoryLLM.memoryllm.util.ModelCheckpointLLM',
                        "target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                        'params': {
                            "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                            "filename": "{epoch:06}-{step:09}",
                            "verbose": True,
                            'save_top_k': -1,
                            'every_n_train_steps': opt.n_train_steps,
                            'save_weights_only': True
                        }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

            del callbacks_cfg['metrics_over_trainsteps_checkpoint']

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer_kwargs.update(vars(trainer_opt))
        # trainer_kwargs.update({
        #     'strategy': DDPStrategy(find_unused_parameters=False)
        # })

        if trainer_kwargs['strategy'] == 'fsdp':

            # fsdp_configs = dict(trainer_kwargs['fsdp_configs'])
            # del trainer_kwargs['fsdp_configs']

            from pytorch_lightning.strategies import FSDPStrategy
            # from customized_fsdp_strategy import FSDPStrategy
            # if 'ignored_modules' in fsdp_configs:
            #     from MemoryLLM.memoryllm.modules.memory_llama import MemoryModule
            #     fsdp_configs['ignored_modules'] = [MemoryModule]
            from torch.distributed.fsdp import MixedPrecision

            trainer_kwargs['strategy'] = FSDPStrategy(
                mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
                cpu_offload=True)

        trainer = Trainer(**trainer_kwargs)
        
        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
       
        # data.prepare_data()

        data.setup()

        # print("#### Data #####")
        # for k in data.datasets:
        #     print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # ########### dataloader statistics: ######################
        # context_lengths = []
        # sequence_lengths = []
        # for idx, batch in tqdm(enumerate(data.train_dataloader().loader1), total=1000):
        #     context, context_mask, sequence, sequence_mask, _ = batch
        #     context_lengths.extend(context_mask.sum(dim=1).tolist())
        #     sequence_lengths.extend(sequence_mask.sum(dim=1).tolist())
        #     if idx > 1000:
        #         break
        
        # # print the maximum, minimum, std, average of the lengths:
        # print("Context:")
        # print(max(context_lengths))
        # print(min(context_lengths))
        # print(np.mean(context_lengths))
        # print(np.std(context_lengths))

        # print("Sequence:")
        # print(max(sequence_lengths))
        # print(min(sequence_lengths))
        # print(np.mean(sequence_lengths))
        # print(np.std(sequence_lengths))
        # #######################################################

        # set the dataset to be one attribute of the model
        # model.train_dataset = data.datasets['train'].data

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(trainer.device_ids)
            # ngpu = len(trainer.device_ids) * trainer.num_nodes
            # ngpu = len(lightning_config.trainer.devices.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            # if trainer.global_rank == 0:
            #     print("Summoning checkpoint.")
            #     ckpt_path = os.path.join(ckptdir, "last.ckpt")
            #     if hasattr(trainer.model, "save_pretrained"):
            #         trainer.model.save_pretrained(ckpt_path)
            #     elif hasattr(trainer.model, "model") and hasattr(trainer.model.model, "save_pretrained"):
            #         trainer.model.model.save_pretrained(ckpt_path)
            #     elif hasattr(trainer, "save_checkpoint"):
            #         trainer.save_checkpoint(ckpt_path)
            #     else:
            #         print("Skipping save ckpt")
            #     print(f"Checkpoint saved to {ckpt_path}")
            pass
            
        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise

        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())

