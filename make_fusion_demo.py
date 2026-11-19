"""Artificial inputs/labels for pipeline smoke testing ONLY; no scientific skill claim."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_demo(output):
    rng=np.random.default_rng(42)
    n=60
    latent=rng.normal(size=n)
    csi=rng.normal(size=(n,16,9)).astype('float32')
    csi[:,:,:8]+=latent[:,None,None]
    csi[:,:,-1]=1
    cloud=rng.normal(size=(n,8,16,16)).astype('float32')
    cloud[:,:4]+=latent[:,None,None,None]
    cloud[:,4:]=1
    rate=np.where(latent>0,latent*3,0).astype('float32')
    times=[pd.Timestamp('2026-01-01T12:00Z')+pd.Timedelta(days=i//6, minutes=10*(i%6)) for i in range(n)]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output,csi=csi,cloud=cloud,rain_rate_mm_h=rate,synthetic_demo=True,
        issue_time_utc=np.array([t.isoformat() for t in times]),
        target_time_utc=np.array([(t+pd.Timedelta(minutes=10)).isoformat() for t in times]),
        history_start_utc=np.array([(t-pd.Timedelta(hours=1)).isoformat() for t in times]),
        event_group=np.array([t.strftime('%Y-%m-%d') for t in times]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    make_demo(p.parse_args().output)
