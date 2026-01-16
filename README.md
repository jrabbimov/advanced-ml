# BLIP team
## Setup Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Notebooks Execution Order

1. **Preprocess.ipynb** - Data preprocessing and evaluation metrics setup
2. **Baseline_Blip.ipynb** - Baseline (ResNet18+BiLSTM) and BLIP fine-tuning
3. **Curriculum.ipynb** - BLIP with curriculum learning strategy

## Configuration

### Save Directory

By default, models are saved to `./saved_models`. To change this, set the `SAVE_ROOT` environment variable:

```bash
export SAVE_ROOT="/path/to/your/models"
```

Or in Python/Jupyter:
```python
import os
os.environ["SAVE_ROOT"] = "/path/to/your/models"
```

### GPU Requirements

- Baseline training: ~4GB GPU memory
- BLIP fine-tuning: ~16GB GPU memory (batch_size=2)
- Curriculum learning: ~16GB GPU memory

## Shared Utilities

Common functions are available in `utils.py`:
- `normalize_text()` - Text normalization for VQA
- `token_f1()` - Token-level F1 metric
- `set_seed()` - Reproducibility helper

## Notes

- All notebooks use the HuggingFace `datasets` library to load SLAKE-VQA
- Outputs from training runs are preserved in the notebooks
- For Google Colab: Mount your drive and set `SAVE_ROOT` to your drive path
