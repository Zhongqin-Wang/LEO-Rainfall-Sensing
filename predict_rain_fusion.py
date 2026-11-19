"""Inference on prepared samples using a saved checkpoint and training scaler."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.rain_fusion import RainFusion
from train_rain_fusion import normalize


def predict(checkpoint, samples, output):
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    with np.load(samples, allow_pickle=False) as d:
        x, cloud = d['csi'].copy(), d['cloud'].copy()
        times = d['target_time_utc'].copy()
    if x.ndim != 3 or cloud.ndim != 4 or x.shape[1:] != (saved['input_csi_steps'], saved['csi_features']) or list(cloud.shape[1:]) != saved['input_cloud_shape'] or len(x) != len(cloud):
        raise ValueError('Input shape does not match checkpoint')
    if not np.isfinite(x).all() or not np.isfinite(cloud).all():
        raise ValueError('Nonfinite input')
    x = normalize(x, saved['csi_mean'].numpy(), saved['csi_scale'].numpy())
    model = RainFusion(saved['csi_features'], saved['cloud_channels'], saved['mode'])
    model.load_state_dict(saved['state_dict']); model.eval()
    with torch.no_grad():
        logits, rate = model(torch.from_numpy(x), torch.from_numpy(cloud.astype('float32')))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(target_time_utc=times, rain_probability=logits.sigmoid().numpy(),
        rain_rate_mm_h=torch.expm1(rate).numpy(), threshold_mm_h=saved['rain_threshold_mm_h'],
        synthetic_demo=saved['synthetic_demo'])).to_csv(output, index=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); predict(a.checkpoint, a.samples, a.output)
