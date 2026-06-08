"""
Training script for MRER
"""

from run import MRER_run


MRER_run(
    model_name='mrer',
    dataset_name='mosi',
    is_tune=False,
    seeds=[3333],
    model_save_dir="./pt",
    res_save_dir="./result",
    log_dir="./log",
    mode='train',
    is_training=True,
    config={
        'use_mr_pgf': True,
        'use_er_dca': True,

        'use_binary_aux': False,
        'use_boundary_aux': False,
        'use_acc7_aux': False,
        'use_ordinal_aux': True,
        'use_output_calibration': False,

        'lambda_balance': 0.005,
        'lambda_rw_recon': 0.0,
        'lambda_consistency': 0.0015,
        'lambda_binary': 0.0,
        'lambda_boundary': 0.0,
        'lambda_acc7': 0.0,
        'lambda_ordinal': 0.0011,

        'calibration_scale': 1.0,
        'calibration_bias': 0.0,
        'pretrained': '/root/autodl-tmp/bert-base-uncased',
        'early_stop': 10,
        'KeyEval': 'Loss',
        'save_test_predictions': False
    }
)
