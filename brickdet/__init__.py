import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'work'
NANODET = WORK / 'nanodet'

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def use_nanodet():
    """Make the patched NanoDet checkout from setup.sh importable."""
    if not (NANODET / 'nanodet').is_dir():
        sys.exit(f'{NANODET} not found, run ./setup.sh')
    if str(NANODET) not in sys.path:
        sys.path.insert(0, str(NANODET))
    import numpy as np
    np.Inf, np.NaN = np.inf, np.nan  # pytorch-lightning 1.9 on NumPy 2


def device(name='auto'):
    if name != 'auto':
        return name
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
