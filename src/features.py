"""Receiver-only features: never open propagation, weather or clock truth."""
from pathlib import Path
import numpy as np


def capture_features(directory):
    directory = Path(directory)
    with np.load(directory / "csi_pilots_corrected.npz", allow_pickle=False) as data:
        if not bool(data["agc_removed"]):
            raise ValueError("Power features require a logged/calibrated gain")
        csi, noise, mask = data["pilot_csi"], data["csi_noise_power"], data["pilot_mask"]
        frequency = data["frequency_offset_hz"]
        if csi.shape != noise.shape or csi.shape != mask.shape or csi.shape[1] != len(frequency):
            raise ValueError("Inconsistent CSI feature grid")
        valid = mask[2:] & np.isfinite(csi[2:]) & np.isfinite(noise[2:])
        signal = abs(csi[2:])**2 - noise[2:]
        coherent = data["signal_power_by_frequency"] if "signal_power_by_frequency" in data else None
        result = {}
        for name, selected in (("all", np.ones(len(frequency), bool)),
                               ("positive", frequency > 0), ("negative", frequency < 0)):
            chosen = valid & selected[None, :]
            if not chosen.any():
                raise ValueError("No known pilots for feature band: " + name)
            if coherent is not None:
                counts = chosen.sum(axis=0)
                used = counts>0
                power = float(np.average(coherent[used],weights=counts[used]))
            else:
                power = float(np.mean(signal[chosen]))
            # Keep signed linear estimates and a validity flag. Never turn a
            # failed low-SNR measurement into zero rainfall or discard its row.
            result[name + "_pilot_signal_power"] = power
            result[name + "_pilot_power_db"] = float(10*np.log10(power)) if power > 0 else np.nan
            result[name + "_pilot_power_resolved"] = power > 0
        for key in ("capture_score", "synchronization_coherence", "estimated_residual_cfo_hz",
                    "estimated_residual_sfo_ppm", "estimated_noise_power_w"):
            if key in data:
                result[key] = float(data[key])
        result["pilot_count"] = int(valid.sum())
        result["positive_minus_negative_power_db"] = result["positive_pilot_power_db"] - result["negative_pilot_power_db"]
    return result
