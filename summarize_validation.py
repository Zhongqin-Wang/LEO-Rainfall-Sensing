"""Summarize completed synthetic experiments; truth is for evaluation only."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
SCENARIOS = {"uniform": "Constant rain", "rain_time_series": "Time-varying rain",
             "moving_rain_cells": "Moving rain cells + reflection"}


def summarize(root):
    report = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  interpretation="Synthetic internal-consistency validation, not measured Starlink performance",
                  scenarios={}, versions={key:version(key) for key in
                    ("numpy","scipy","pandas","astropy","astropy-iers-data","sgp4","itur")})
    files = [*ROOT.glob("*.py"),*ROOT.glob("src/*.py"),*ROOT.glob("tests/test_*.py"),ROOT/"requirements.txt"]
    report["source_sha256"] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    test_log=root/"logs/tests.log"
    if test_log.exists():
        summaries=re.findall(r"\d+ passed[^\n]*",test_log.read_text())
        report["regression_test_summary"]=summaries[-1] if summaries else "No passing summary; inspect logs/tests.log"
    report["limitations"] = [
        "Published Starlink synchronization/pilot structure with synthetic payload and assumed RF/clock parameters",
        "Ideal simultaneous no-rain reference and exact logged AGC calibration",
        "Uniform-equivalent slant-path rain rate conditional on assumed rain-layer height, not point rainfall",
        "One fixed satellite pass and random seed per scenario; not a Monte Carlo confidence bound",
        "Independent short frames at 10-second steps, not continuously phase-coherent IQ",
        "Optional smooth local ground facet and assumed antenna pattern, not measured terrain or antenna data"]
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout="constrained")
    for i,(name,title) in enumerate(SCENARIOS.items()):
        directory=root/name
        frame=pd.read_csv(directory/"passes/satellite_11007/rain_timeseries.csv")
        metadata=json.loads((directory/"run_metadata.json").read_text())
        frame=frame.sort_values("time_utc")
        ok=frame.status.eq("ok")
        attenuation_error=(frame.estimated_rain_db-frame.true_rain_db)[ok].to_numpy()
        rate_error=(frame.estimated_equivalent_rain_rate_mm_h-frame.true_equivalent_rain_rate_mm_h)[ok].to_numpy()
        valid_rate=np.isfinite(rate_error)
        if not ok.any() or not np.isfinite(attenuation_error).all():
            raise ValueError(f"No finite successful attenuation estimates: {name}")
        stats=dict(selected_pairs=len(frame),generated_pairs=int(frame.generation_status.eq("generated").sum()),
            successful_pairs=int(ok.sum()),processing_status_counts={str(k):int(v) for k,v in frame.status.value_counts().items()},
            attenuation_rmse_db=float(np.sqrt(np.mean(attenuation_error**2))),
            attenuation_bias_db=float(np.mean(attenuation_error)),
            attenuation_max_abs_error_db=float(np.max(abs(attenuation_error))),
            equivalent_rate_resolved_count=int(valid_rate.sum()),
            equivalent_rate_rmse_mm_h=float(np.sqrt(np.mean(rate_error[valid_rate]**2))) if valid_rate.any() else None,
            satellite_name=str(frame.satellite_name.iloc[0]),norad_id=int(frame.norad_id.iloc[0]),
            orbit_sha256=metadata["geometry_metadata"]["orbit_sha256"],
            earth_orientation=metadata["geometry_metadata"]["earth_orientation"],
            configuration_sha256=hashlib.sha256(json.dumps(metadata["configuration"],sort_keys=True,default=str).encode()).hexdigest())
        report["scenarios"][name]=stats
        configuration=metadata["configuration"]
        (directory/"configuration_as_run.yaml").write_text(yaml.safe_dump(configuration,sort_keys=False))
        frozen=json.loads(json.dumps(configuration))
        frozen["geometry"]["eop_path"]=str(Path(stats["earth_orientation"]["provenance"]["archive_path"]).relative_to(ROOT))
        (directory/"configuration_replay.yaml").write_text(yaml.safe_dump(frozen,sort_keys=False))
        minutes=frame.elapsed_s/60
        for ax,true,estimate,ylabel in (
            (axes[i,0],frame.true_rain_db,frame.estimated_rain_db,"Rain attenuation (dB)"),
            (axes[i,1],frame.true_equivalent_rain_rate_mm_h,frame.estimated_equivalent_rain_rate_mm_h,"Path-equivalent rain (mm/h)")):
            ax.plot(minutes,true,"k--",label="Simulation truth",lw=1.5)
            ax.plot(minutes[ok],estimate[ok],"o-",ms=4,label="CSI estimate",lw=1.2)
            ax.set(xlabel="Minutes since first visible sample",ylabel=ylabel)
            ax.grid(alpha=.25)
        axes[i,0].set_title(f"{title} | {ok.sum()}/{len(frame)} pairs | RMSE {stats['attenuation_rmse_db']:.3f} dB",fontsize=11)
        axes[i,1].set_title(title,fontsize=11)
    axes[0,0].legend(fontsize=9)
    axes[0,1].legend(fontsize=9)
    fig.suptitle("Synthetic Starlink public-profile validation | NORAD 100520 | 18 October 2026",fontsize=13)
    fig.savefig(root/"fix_validation.png",dpi=180)
    fig.savefig(root/"fix_validation.svg")
    plt.close(fig)
    report["replay_note"]="configuration_as_run.yaml preserves the exact tested configuration; configuration_replay.yaml points to identical archived EOP bytes, so its config hash differs. Regenerate geometry before replay."
    (root/"fix_validation.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT/"outputs/final_validation")
    args=parser.parse_args()
    result=summarize(args.root)
    for name,stats in result["scenarios"].items():
        print(f"{name}: {stats['successful_pairs']}/{stats['selected_pairs']} pairs; attenuation RMSE {stats['attenuation_rmse_db']:.5f} dB")
