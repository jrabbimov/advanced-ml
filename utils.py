"""
Shared utility functions for SLAKE VQA notebooks.
"""

import re
import unicodedata
from collections import Counter


def normalize_text(s: str) -> str:
    """
    Normalize text for training and evaluation.
    Keeps useful medical symbols: / + - . : % ( ) < >
    """
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)

    # Keep useful medical symbols
    s = re.sub(r"[^\w\s/\+\-\.:%%()<>]", "", s)

    # Yes/no canonicalization
    if s in {"y", "yeah", "yep", "true", "positive"}:
        s = "yes"
    elif s in {"n", "nope", "false", "negative"}:
        s = "no"

    # Light modality normalization
    s = s.replace("computed tomography", "ct")
    s = s.replace("magnetic resonance", "mri")
    s = s.replace("x ray", "x-ray")

    return s


def token_f1(pred: str, gt: str) -> float:
    """
    Compute token-level F1 score between prediction and ground truth.
    """
    p = normalize_text(pred).split()
    g = normalize_text(gt).split()

    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0

    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0

    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    import os
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
