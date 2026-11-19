"""Process a manifest of independent short frames across satellite passes."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_csi import process_capture
from run_rain import estimate_rain


def process_passes(root, satellite):
    summary_dir = root / "passes" / satellite
    manifest = pd.read_csv(summary_dir / "captures.csv", keep_default_na=False)
    records = []
    for _, row in manifest.iterrows():
        result = row.to_dict()
        result.update(status=row["generation_status"], estimated_rain_db=np.nan,
                      original_estimated_rain_db=np.nan, error_db=np.nan)
        if row["generation_status"] == "generated":
            errors = []
            for case in ("reference", "rain"):
                path = root / row[f"{case}_capture"]
                try:
                    quality = process_capture(path, plot=False)
                    result[f"{case}_status"] = "ok"
                    result.update({f"{case}_{key}": value for key, value in quality.items()})
                except (ValueError, RuntimeError, OSError, KeyError) as error:
                    result[f"{case}_status"] = "failed"
                    errors.append(f"{case}: {error}")
            if errors:
                result.update(status="receiver_failed", error="; ".join(errors))
                (root / row["rain_capture"] / "rain_estimate.npz").unlink(missing_ok=True)
            else:
                try:
                    result.update(estimate_rain(
                        root / row["rain_capture"], root / row["reference_capture"], plot=False
                    ))
                    result.update(status="ok", error="")
                except (ValueError, RuntimeError, OSError, KeyError) as error:
                    result.update(status="rain_estimation_failed", error=str(error))
        records.append(result)
        # Keep completed and failed rows even if processing is interrupted later.
        pd.DataFrame(records).to_csv(summary_dir / "rain_timeseries.csv", index=False)
        print(f"{row['time_utc']}: {result['status']}", flush=True)

    summary = pd.DataFrame(records)
    for pass_id, group in summary.groupby("pass_id", sort=False):
        group = group.sort_values("time_utc")
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(group.elapsed_s, group.true_rain_db, "k--", label="Simulated")
        axes[0].plot(group.elapsed_s, group.estimated_rain_db, "o-", label="Estimated")
        axes[0].set_ylabel("Rain attenuation (dB)")
        axes[0].legend()
        axes[1].plot(group.elapsed_s, group.elevation_deg, color="tab:green")
        failed = group.status != "ok"
        if failed.any():
            axes[1].scatter(group.loc[failed, "elapsed_s"], group.loc[failed, "elevation_deg"],
                            color="red", marker="x", label="Skipped / failed")
            axes[1].legend()
        axes[1].set(xlabel="Seconds since first visible sample", ylabel="Elevation (deg)")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.suptitle(f"{satellite} | pass {pass_id} | sampled short frames")
        fig.tight_layout()
        name = pd.Timestamp(group.iloc[0].time_utc).strftime("%Y%m%dT%H%M%SZ")
        fig.savefig(summary_dir / f"rain_pass_{name}.png", dpi=150)
        plt.close(fig)
    print(summary.status.value_counts().to_string())
    print(f"Saved: {summary_dir / 'rain_timeseries.csv'}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellite", required=True, help="For example satellite_0006")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    summary = process_passes(args.root, args.satellite)
    if summary.empty or (summary.status != "ok").any():
        raise SystemExit(2)  # Preserve all rows, but do not report a clean CLI run.


if __name__ == "__main__":
    main()
