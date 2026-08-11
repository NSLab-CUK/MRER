"""
Testing script for MRER
"""

from run import MRER_run


MRER_run(
    model_name='mrer', 
    dataset_name='mosi', 
    is_tune=False, 
    seeds=[1111], 
    model_save_dir="./pt",
    res_save_dir="./result", 
    log_dir="./log", 
    mode='test', 
    is_training=False
)
