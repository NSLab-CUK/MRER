import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import get_config_regression
from data_loader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import mrer
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('MMSA')

def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def MRER_run(model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=500, feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=4, verbose_level=1, mode='', is_training=False):
    
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    
    if config_file != "":
        config_file = Path(config_file)
    else:
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")

    if model_save_dir == "":
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]

    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    
    args = get_config_regression(model_name, dataset_name, config_file)
    args.is_training = is_training  
    args.mode = mode
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'regression'
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    if config:
        args.update(config)

    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args['cur_seed'] = i + 1
        result = _run(args, num_workers, is_tune)
        model_results.append(result)
    
    if args.is_training:
        criterions = list(model_results[0].keys())
        csv_file = res_save_dir / f"{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Model"] + criterions)
        
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            res.append(values[-1])  
        
        if len(res) == len(df.columns):
            df.loc[len(df)] = res
            df.to_csv(csv_file, index=None)
            logger.info(f"Results saved to {csv_file}.")
        else:
            logger.error(f"Column mismatch: expected {len(df.columns)}, got {len(res)}")

def _run(args, num_workers=4, is_tune=False, from_sena=False):
    dataloader = MMDataLoader(args, num_workers)
    
    if args.is_training:
        model = getattr(mrer, 'MRER')(args)
        model = model.cuda()
    else:
        model = getattr(mrer, 'MRER')(args)
        model = model.cuda()

    trainer = ATIO().getTrain(args)

    if args.mode == 'test':  
        model.load_state_dict(torch.load('./pt/mrer' + str(args.dataset_name)+'.pth'))
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        sys.stdout.flush()
        input('[Press Any Key to start another run]')
    else:
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
        model.load_state_dict(torch.load('./pt/mrer' + str(args.dataset_name)+'.pth'))
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results
