"""
Measure training throughput (images/s) for a config without running a full training.

  python -m brickdet.bench configs/nanodet-plus-m_416.yml [--device auto] [--steps 40] [--batch N] [--same-batch]

--same-batch repeats one batch, so data loading is excluded.
"""
import argparse
import time

import torch

from . import device, use_nanodet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--same-batch', action='store_true')
    a = ap.parse_args()
    use_nanodet()
    from nanodet.data.batch_process import stack_batch_img
    from nanodet.data.collate import naive_collate
    from nanodet.data.dataset import build_dataset
    from nanodet.model.arch import build_model
    from nanodet.util import cfg, load_config

    load_config(cfg, a.config)
    dev = torch.device(device(a.device))
    bs = a.batch or cfg.device.batchsize_per_gpu
    ds = build_dataset(cfg.data.train, 'train')
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=cfg.device.workers_per_gpu,
                                     collate_fn=naive_collate, drop_last=True, persistent_workers=True)
    model = build_model(cfg.model).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    it = iter(dl)
    first = next(it)

    def step():
        batch = dict(first) if a.same_batch else next(it)
        batch['img'] = stack_batch_img([x.to(dev) for x in batch['img']], divisible=32)
        _, loss, _ = model.forward_train(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    for _ in range(5):  # warm-up
        step()
    t = time.time()
    for _ in range(a.steps):
        loss = step()
    dt = time.time() - t
    ips = a.steps * bs / dt
    print(f'{dev} batch {bs}: {dt / a.steps:.2f} s/step, {ips:.1f} img/s, loss {loss:.3f}')
    for epochs in (100, 150, 300):
        print(f'  {len(ds)} images x {epochs} epochs ~ {len(ds) * epochs / ips / 3600:.1f} h')


if __name__ == '__main__':
    main()
