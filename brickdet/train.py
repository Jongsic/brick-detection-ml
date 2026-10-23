"""
Train with NanoDet's tools/train.py, choosing the device here instead of in the config.

  python -m brickdet.train configs/nanodet-plus-m_416.yml [--device auto|mps|cuda|cpu] [--seed 7]

Checkpoints and logs go to the config's save_dir. To resume, set schedule.resume in the config.
"""
import argparse
import os
import platform
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from . import NANODET, ROOT, WORK, device, use_nanodet

GPU_IDS = {'mps': 'mps', 'cuda': [0], 'cpu': -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--device', default='auto', choices=['auto', *GPU_IDS])
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    use_nanodet()

    dev = device(a.device)
    cfg = yaml.safe_load(open(a.config))
    cfg['device']['gpu_ids'] = GPU_IDS[dev]
    WORK.mkdir(exist_ok=True)
    resolved = WORK / f'{Path(a.config).stem}.{dev}.yml'
    yaml.safe_dump(cfg, open(resolved, 'w'), sort_keys=False)

    if platform.system() == 'Darwin' and shutil.which('caffeinate'):
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])  # keep the Mac awake while training
    os.chdir(ROOT)
    sys.argv = [str(NANODET / 'tools' / 'train.py'), str(resolved), '--seed', str(a.seed)]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
