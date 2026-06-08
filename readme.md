# MRER

MRER

## Project Structure

```
MRER/
├── train.py                  # Training entry
├── test.py                   # Evaluation script
├── run.py                    # Main runner
├── config/
│   └── config.json           # Runtime configuration
├── trains/
│   └── singleTask/
│       ├── MRER.py           # Trainer
│       └── model/
│           └── mrer.py       # Core MRER architecture
├── models/
│   └── cross_modal_ssm.py    # Cross-modal interaction module
├── dataset/                  # Processed MOSI/MOSEI datasets
├── utils/                    # Logging and evaluation utilities
└── requirements.txt
```

## Datasets

- **CMU-MOSI**
- **CMU-MOSEI**

Place datasets in the `./dataset` folder, or modify the dataset path in `config/config.json`.

## Installation

1. **Create a virtual environment** (recommended):
```bash
python3 -m venv mrer_env
source mrer_env/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Execution

### Training

Set `dataset_name='mosi'` or `dataset_name='mosei'` in `train.py`, then run:

```bash
python train.py
```

The trained model will be saved in the `./pt` directory.

### Testing

Set the dataset name in `test.py` and the model path in `run.py`, then run:

```bash
python test.py
```

## Configuration

Runtime parameters can be modified in `./config/config.json`, including:
- Dataset paths
- Hyperparameters (learning rate, batch size, etc.)
- Module switches (MR-PGF, ER-DCA)

Logs and results are saved in `./log` and `./result/normal` directories.
