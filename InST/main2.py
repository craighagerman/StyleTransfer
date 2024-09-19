import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch

import sys

sys.path.append(os.getcwd())
from main2_modules import (
    DataLoader,
    DataModuleFromConfig,
    WrappedDataset,
    CUDACallback,
    SetupCallback,
    ImageLogger,
)

import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor


try:
    # Pytorch 1.x
    from pytorch_lightning.utilities.distributed import rank_zero_only
except ModuleNotFoundError:
    # Pytorch 2.x
    from pytorch_lightning.utilities.rank_zero import rank_zero_only


from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config


from transformers import logging

logging.set_verbosity_error()


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print(f"{len(m)} missing keys:")
        # print(m)
    if len(u) > 0 and verbose:
        print(f"{len(u)} unexpected keys:")
        # print(u)

    print(f"type(model): {type(model)}")
    # model.cuda()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    return model


# def nondefault_trainer_args(opt):
#     parser = argparse.ArgumentParser()
#     parser = Trainer.add_argparse_args(parser)
#     args = parser.parse_args([])
#     return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


################################################################################


################################################################################
def setup():
    sys.path.append(os.getcwd())


def load_config(opt, unknown):
    ### === Load yaml config file  (e.g. v1-finetune.yaml) ============= ###
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    print(f"lightning_config:\n{lightning_config}")
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "ddp"
    # for k in nondefault_trainer_args(opt):
    #     print(f"K: {k}")
    #     trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    print()
    print(f"trainer_opt: {trainer_opt}")
    print(f"benchmark in trainer_opt {'benchmark' in trainer_opt}")
    print(f"benchmark in max_steps {'max_steps' in trainer_opt}")
    print()
    lightning_config.trainer = trainer_config
    return config, lightning_config, trainer_opt, cpu


def prep_config(
    config,
    lightning_config,
    trainer_opt,
    cpu,
    opt,
    now,
    nowname,
    logdir,
    ckptdir,
    cfgdir,
):
    ### === Model === ###

    # config.model.params.personalization_config.params.init_word = opt.init_word
    # CH: not used
    config.model.params.personalization_config.params.embedding_manager_ckpt = (
        opt.embedding_manager_ckpt
    )
    # CH: not used
    config.model.params.personalization_config.params.placeholder_tokens = (
        opt.placeholder_tokens
    )
    # CH: not used
    if opt.init_word:
        config.model.params.personalization_config.params.initializer_words[
            0
        ] = opt.init_word

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "offline": opt.debug,
                "id": nowname,
            },
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            },
        },
    }
    default_logger_cfg = default_logger_cfgs["testtube"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        },
    }

    # TODO [CH] - need to get/pass in model.monitor here
    # # CH: true monitor in model config
    # if hasattr(model, "monitor"):
    #     print(f"Monitoring {model.monitor} as checkpoint metric.")
    #     default_modelckpt_cfg["params"]["monitor"] = model.monitor
    #     default_modelckpt_cfg["params"]["save_top_k"] = 1

    # CH: false no "modelcheckpoint" in lightning_config
    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

    # NOTE [CH]: pl is pytorch_lightning. We're checking the version here.
    # but not sure WHY it matters about less than or greater than 1.4.0??
    if version.parse(pl.__version__) < version.parse("1.4.0"):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main2_modules.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            },
        },
        "image_logger": {
            "target": "main2_modules.ImageLogger",
            "params": {"batch_frequency": 750, "max_images": 4, "clamp": True},
        },
        # "learning_rate_logger": {
        #     "target": "main.LearningRateMonitor",
        #     "params": {
        #         "logging_interval": "step",
        #         # "log_momentum": True
        #     },
        # },
        "cuda_callback": {"target": "main2_modules.CUDACallback"},
    }
    if version.parse(pl.__version__) >= version.parse("1.4.0"):
        default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
        print(
            "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
        )
        default_metrics_over_trainsteps_ckpt_dict = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 1,
                    #  'every_n_train_steps': 10000,
                    "save_weights_only": True,
                },
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if "ignore_keys_callback" in callbacks_cfg and hasattr(
        trainer_opt, "resume_from_checkpoint"
    ):
        callbacks_cfg.ignore_keys_callback.params[
            "ckpt_path"
        ] = trainer_opt.resume_from_checkpoint
    elif "ignore_keys_callback" in callbacks_cfg:
        del callbacks_cfg["ignore_keys_callback"]

    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]
    trainer_kwargs["max_steps"] = trainer_opt.max_steps

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ###

    # data
    # NOTE - data_root is the directory with training images
    config.data.params.train.params.data_root = opt.data_root
    config.data.params.validation.params.data_root = opt.data_root
    data = instantiate_from_config(config.data)

    # NOTE [CH]: why is this duplicated from above? seems like a mistake
    data = instantiate_from_config(config.data)

    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"\t{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate

    if not cpu:
        ngpu = lightning_config.trainer.gpus
    else:
        ngpu = 1
    if "accumulate_grad_batches" in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
            )
        )
    else:
        learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {learning_rate:.2e}")

    return data, trainer, learning_rate


def load_model(config, opt):
    # CH: we DO use actual_resume
    # TODO [CH]: `model` isn't a great name since this isn't a model. Should be model_config
    if opt.actual_resume:
        model = load_model_from_config(config, opt.actual_resume)
    else:
        model = instantiate_from_config(config.model)
    return model


def run(data, trainer, opt, model, learning_rate, logdir, ckptdir):
    try:
        model.learning_rate = learning_rate

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb

                pudb.set_trace()

        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # ========================================
        # run
        # ========================================
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            data_module = DataModuleFromConfig(bs)
            trainer.test(model, data, datamodule=data_module)

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


################################################################################
# main()
################################################################################


def main():
    setup()
    opt, unknown, logdir, now, nowname = parse_arguments()
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    print(f"\n[CH] opt: {opt}\n")  # deleteme
    print(f"[CH] unknown: {unknown}")  # deleteme
    print(f"[CH] now: {now}")
    print(f"[CH] logdir: {logdir}")
    print(f"[CH] ckptdir: {ckptdir}")
    print(f"[CH] cfgdir: {cfgdir}")

    config, lightning_config, trainer_opt, cpu = load_config(opt, unknown)

    print(f"\n[CH] lightning_config: {lightning_config}\n")
    print(f"[CH] trainer_opt: {trainer_opt}")

    seed_everything(opt.seed)

    data, trainer, learning_rate = prep_config(
        config,
        lightning_config,
        trainer_opt,
        cpu,
        opt,
        now,
        nowname,
        logdir,
        ckptdir,
        cfgdir,
    )

    print(f"[CH] type(data): {type(data)}")
    print(f"[CH] type(trainer): {type(trainer)}")
    print("\n\n")
    print("-"*80)
    model = load_model(config, opt)
    run(data, trainer, opt, model, learning_rate, logdir, ckptdir)
    print("-" * 80)
    print("* DONE *")


################################################################################
# Helper functions
################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="InST main training code")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

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
        "-p", "--project", help="name of new or path to existing project"
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
        default=23,
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
        "--datadir_in_name",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Prepend the final directory in the data_root to the output directory name",
    )

    parser.add_argument(
        "--actual_resume",
        type=str,
        default="",
        help="Path to model to actually resume from",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to directory with training images",
    )

    parser.add_argument(
        "--embedding_manager_ckpt",
        type=str,
        default="",
        help="Initialize embedding manager from a checkpoint",
    )
    parser.add_argument("--placeholder_tokens", type=str, nargs="+", default=["*"])

    parser.add_argument(
        "--init_word",
        type=str,
        help="Word to use as source for initial token embedding.",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        nargs="?",
        default=0,
        help="Number of GPUs to use",
    )

    # args = parser.parse_args()
    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
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

        if opt.datadir_in_name:
            now = os.path.basename(os.path.normpath(opt.data_root)) + now

        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    return opt, unknown, logdir, now, nowname


################################################################################
# if __name__
################################################################################
if __name__ == "__main__":
    main()
