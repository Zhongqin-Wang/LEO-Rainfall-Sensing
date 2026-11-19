import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from itur.models import itu838
from scipy.optimize import brentq
from scipy.special import logsumexp
from src.rain_spectrum import spectral_coefficients


def invert_equivalent_rain(attenuation_db, settings, frequency_offsets_hz=None, reference_weights=None):
    """Uniform-equivalent path rate, conditional on the assumed rain-layer height."""
    length = float(settings["assumed_rain_path_km"])
    if not np.isfinite(length) or length <= 0:
        return np.nan, "no_identifiable_rain_path"
    if not np.isfinite(attenuation_db) or attenuation_db < 0:
        return np.nan, "negative_or_unresolved_attenuation"
    model = settings.get("rain_frequency_model", "center_frequency_flat")
    if model not in ("center_frequency_flat", "itu838_frequency_dependent", "mie_spherical_mp"):
        raise ValueError("Unknown rain attenuation retrieval model")
    if model in ("itu838_frequency_dependent", "mie_spherical_mp"):
        offsets = np.asarray(frequency_offsets_hz, dtype=float)
        weights = np.asarray(reference_weights, dtype=float)
        if (offsets.ndim != 1 or offsets.shape != weights.shape or not offsets.size
                or not np.isfinite(offsets).all() or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.any(weights > 0)):
            return np.nan, "unresolved_frequency_weights"
        selected = weights > 0
        weights = weights[selected] / weights.sum()
        frequency = float(settings["carrier_hz"]) + offsets[selected]
        if model == "mie_spherical_mp":
            from src.rain_complex import complex_rain_spectrum, settings_from_config
            microphysics = settings_from_config(settings)
        else:
            k, alpha = spectral_coefficients(frequency,
                float(settings["elevation_deg"]), float(settings["polarization_tilt_deg"]))
        def modeled(rate):
            losses = (complex_rain_spectrum(frequency, length, [rate], **microphysics)[0]
                      if model == "mie_spherical_mp" else np.asarray(k)*rate**np.asarray(alpha)*length)
            return -10 / np.log(10) * logsumexp(np.log(weights) - np.log(10) / 10 * losses)
        if attenuation_db == 0:
            return 0., "ok"
        upper = 1.
        limit = 500. if model == "mie_spherical_mp" else 1e6
        while modeled(upper) < attenuation_db and upper < limit:
            upper = min(2*upper, limit)
        if modeled(upper) < attenuation_db:
            return np.nan, "attenuation_outside_retrieval_range"
        return float(brentq(lambda r: modeled(r) - attenuation_db, 0, upper)), "ok"
    k,alpha = itu838.rain_specific_attenuation_coefficients(
        float(settings["carrier_hz"])/1e9,float(settings["elevation_deg"]),float(settings["polarization_tilt_deg"]))
    return float((attenuation_db/(float(k)*length))**(1/float(alpha))), "ok"


def estimate_rain(rain_dir, reference_dir, plot=True):
    rain_dir, reference_dir = Path(rain_dir), Path(reference_dir)
    (rain_dir / "rain_estimate.npz").unlink(missing_ok=True)
    rain_meta, ref_meta = rain_dir / "capture_metadata.json", reference_dir / "capture_metadata.json"
    if rain_meta.exists() != ref_meta.exists():
        raise ValueError("Capture identity metadata is missing for one member of the pair")
    if rain_meta.exists() and ref_meta.exists():
        if json.loads(rain_meta.read_text()).get('receive_channels',1) != 1:
            raise ValueError('Legacy scalar rain retrieval accepts one channel; select a channel or design an array estimator explicitly')
        rain_info = json.loads(rain_meta.read_text())
        ref_info = json.loads(ref_meta.read_text())
        for key in ("satellite_index", "pass_id", "time_utc", "norad_id", "waveform_profile", "configuration_sha256"):
            if rain_info.get(key) != ref_info.get(key):
                raise ValueError(f"Capture mismatch: {key}")
        if rain_info.get("case", "rain") != "rain" or ref_info.get("case", "reference") != "reference":
            raise ValueError("The estimator requires distinct rain and reference captures")
    if rain_dir.resolve() == reference_dir.resolve():
        raise ValueError("Rain and reference cannot be the same capture")
    filename = "csi_pilots_corrected.npz"

    # Exclude the two training symbols.
    with np.load(reference_dir / filename) as data:
        if "csi_noise_power" not in data:
            raise ValueError("Rerun run_csi.py for the reference capture")
        reference_noise = data["csi_noise_power"][2:].copy()
        selection_amplitude = np.abs(data["pilot_csi"][1]).copy()
        reference = data["pilot_csi"][2:].copy()
        reference_mask = data["pilot_mask"][2:].copy()
        reference_frequency = data["frequency_offset_hz"].copy()
        reference_agc_removed = bool(data["agc_removed"])
        reference_coherent = data["signal_power_by_frequency"].copy() if "signal_power_by_frequency" in data else None

    with np.load(rain_dir / filename) as data:
        if "csi_noise_power" not in data:
            raise ValueError("Rerun run_csi.py for the rainy capture")
        rainy_noise = data["csi_noise_power"][2:].copy()
        rainy = data["pilot_csi"][2:].copy()
        rainy_mask = data["pilot_mask"][2:].copy()
        rainy_frequency = data["frequency_offset_hz"].copy()
        rainy_agc_removed = bool(data["agc_removed"])
        rainy_coherent = data["signal_power_by_frequency"].copy() if "signal_power_by_frequency" in data else None

    if not (reference_agc_removed and rainy_agc_removed):
        raise ValueError("AGC must be removed from both captures")

    if (
        reference.shape != rainy.shape
        or not np.array_equal(reference_mask, rainy_mask)
        or not np.array_equal(reference_frequency, rainy_frequency)
    ):
        raise ValueError("Reference and rainy CSI grids do not match")

    ref_amplitude = np.abs(reference)
    rain_amplitude = np.abs(rainy)

    reference_valid = (
        reference_mask
        & np.isfinite(reference)
        & (ref_amplitude > 0)
    )

    if not np.any(reference_valid):
        raise ValueError("No valid reference pilots")

    # Select frequencies with a separate reference training symbol, not the
    # same noisy data amplitudes whose mean power enters the estimator.
    training_valid = np.isfinite(selection_amplitude) & (selection_amplitude>0)
    if not training_valid.any():
        raise ValueError("No independent reference training coefficients for fade screening")
    threshold = .2*np.median(selection_amplitude[training_valid])
    selected_frequency = training_valid & (selection_amplitude>threshold)
    valid = (
        reference_valid
        & rainy_mask
        & np.isfinite(rainy)
        & (rain_amplitude > 0)
        & selected_frequency[None,:]
        & np.isfinite(reference_noise)
        & np.isfinite(rainy_noise)
    )

    attenuation_db = np.full(reference.shape, np.nan)
    attenuation_db[valid] = 20 * np.log10(
        ref_amplitude[valid] / rain_amplitude[valid]
    )

    usable = np.any(valid, axis=1)
    if not np.any(usable):
        raise ValueError("No usable pilot pairs")

    # Keep the original estimator for comparison.
    original_per_symbol = np.full(len(reference), np.nan)
    original_per_symbol[usable] = np.nanmedian(
        attenuation_db[usable], axis=1
    )
    original_db = float(np.nanmedian(original_per_symbol))

    # Average powers before subtraction; do not clip individual pilots.
    counts = valid.sum(axis=1)
    denominator = np.maximum(counts, 1)
    reference_signal = np.sum(
        np.where(valid, ref_amplitude**2 - reference_noise, 0.0), axis=1
    ) / denominator
    rainy_signal = np.sum(
        np.where(valid, rain_amplitude**2 - rainy_noise, 0.0), axis=1
    ) / denominator
    resolved = usable & (reference_signal > 0) & (rainy_signal > 0)
    rain_db_per_symbol = np.full(len(reference), np.nan)
    rain_db_per_symbol[resolved] = 10 * np.log10(
        reference_signal[resolved] / rainy_signal[resolved]
    )

    # Frame estimate uses all selected pilots, not only positive symbols.
    frame_reference = float(np.mean(
        (ref_amplitude**2 - reference_noise)[valid]
    ))
    frame_rainy = float(np.mean(
        (rain_amplitude**2 - rainy_noise)[valid]
    ))
    incoherent_db = (float(10*np.log10(frame_reference/frame_rainy))
                     if frame_reference>0 and frame_rainy>0 else np.nan)
    estimator = "noise_corrected_mean_power_ratio"
    weights = np.sum(np.where(valid, ref_amplitude**2-reference_noise, 0.), axis=0)
    if (reference_coherent is None) != (rainy_coherent is None):
        raise ValueError("Paired captures must use the same pilot-power estimator")
    if reference_coherent is not None:
        if reference_coherent.shape != reference_frequency.shape or rainy_coherent.shape != reference_frequency.shape:
            raise ValueError("Invalid coherent pilot-power dimensions")
        frequency_counts = valid.sum(axis=0)
        chosen = frequency_counts>0
        if not np.isfinite(reference_coherent[chosen]).all() or not np.isfinite(rainy_coherent[chosen]).all():
            raise ValueError("Nonfinite coherent pilot power")
        frame_reference = float(np.average(reference_coherent[chosen],weights=frequency_counts[chosen]))
        frame_rainy = float(np.average(rainy_coherent[chosen],weights=frequency_counts[chosen]))
        weights = np.where(chosen, reference_coherent*frequency_counts, 0.)
        estimator = "multi_lag_channel_autocorrelation_power_ratio"
    if frame_reference <= 0 or frame_rainy <= 0:
        raise ValueError("Signal power is unresolved after pilot processing")
    estimated_db = float(10 * np.log10(frame_reference / frame_rainy))
    estimated_rate,rate_status = np.nan,"retrieval_geometry_not_supplied"
    parameters_path = rain_dir / "retrieval_config.json"
    if parameters_path.exists():
        parameters = json.loads(parameters_path.read_text())
        # Match the power-ratio observable across frequencies. Negative noisy
        # reference weights cannot define a physical mixture; do not clip them.
        estimated_rate,rate_status = invert_equivalent_rain(estimated_db,parameters,
                                                            reference_frequency,weights)

    np.savez_compressed(
        rain_dir / "rain_estimate.npz",
        attenuation_db=attenuation_db,
        rain_db_per_symbol=rain_db_per_symbol,
        estimated_rain_db=estimated_db,
        estimated_equivalent_rain_rate_mm_h=estimated_rate,
        rain_rate_retrieval_status=rate_status,
        rain_rate_interpretation="uniform-equivalent path rate, not point rainfall",
        original_rain_db_per_symbol=original_per_symbol,
        original_estimated_rain_db=original_db,
        reference_signal_power_per_symbol=reference_signal,
        rainy_signal_power_per_symbol=rainy_signal,
        valid_symbols=resolved,
        valid_pilots=valid,
        estimator=estimator,
        noise_subtracted_estimated_rain_db=incoherent_db,
        frequency_selection_source="separate_reference_training_symbol",
        reference_role="simultaneous_synthetic_counterfactual_not_a_single_SDR_observable",
        task_type="contemporaneous_attenuation_retrieval_not_future_rainfall_forecasting",
    )
    print(f"Original amplitude-ratio estimate: {original_db:.3f} dB")

    # Truth is used only after estimation, for comparison.
    true_db,true_rate,true_center = np.nan,np.nan,np.nan
    truth_observable = "center_frequency_attenuation"
    truth_status = "missing"
    try:
        with np.load(rain_dir / "Hprop.npz") as data:
            true_db = float(data["rain_db"])
            true_center = true_db
            truth_status = "available" if np.isfinite(true_db) else "nonfinite"
            if "rain_effective_mm_h" in data:
                true_rate = float(data["rain_effective_mm_h"])
            if "rain_db_by_frequency" in data and (reference_dir / "Hprop.npz").exists():
                with np.load(reference_dir / "Hprop.npz") as ref_truth:
                    hp_rain, hp_ref = data["Hprop"], ref_truth["Hprop"]
                if hp_rain.shape != reference_frequency.shape or hp_ref.shape != reference_frequency.shape:
                    raise ValueError("Invalid channel truth dimensions")
                counts_per_frequency = valid.sum(axis=0)
                true_db = float(10*np.log10(np.sum(counts_per_frequency*abs(hp_ref)**2)
                                           / np.sum(counts_per_frequency*abs(hp_rain)**2)))
                truth_observable = "selected_pilot_power_ratio_including_frequency_dependent_rain"
    except FileNotFoundError:
        pass
    except (OSError,ValueError,KeyError):
        truth_status = "invalid"  # Preserve inference, but expose evaluation corruption.
        true_db, true_rate, true_center = np.nan, np.nan, np.nan

    print(f"Estimated rain attenuation: {estimated_db:.3f} dB")
    print(f"Simulated rain attenuation: {true_db:.3f} dB")
    print(f"Estimation error: {estimated_db - true_db:+.3f} dB")

    result = {
        "estimated_rain_db": estimated_db,
        "original_estimated_rain_db": original_db,
        "noise_subtracted_estimated_rain_db": incoherent_db,
        "estimator": estimator,
        "true_rain_db": true_db,
        "error_db": estimated_db - true_db,
        "estimated_equivalent_rain_rate_mm_h": estimated_rate,
        "rain_rate_retrieval_status": rate_status,
        "true_equivalent_rain_rate_mm_h": true_rate,
        "true_rain_db_center_frequency": true_center,
        "truth_attenuation_observable": truth_observable,
        "reference_role": "simultaneous_synthetic_counterfactual",
        "task_type": "contemporaneous_retrieval",
        "truth_evaluation_status": truth_status,
    }
    if not plot:
        return result

    symbols = np.arange(1, len(rain_db_per_symbol) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(symbols, original_per_symbol, "o--", alpha=0.6, label="Original amplitude ratio")
    ax.plot(symbols, rain_db_per_symbol, "o-", label="Noise-corrected power ratio")
    ax.axhline(true_db, color="black", linestyle="--", label="Simulated")
    ax.set(
        xlabel="Data symbol number",
        ylabel="Rain attenuation (dB)",
        title=rain_dir.name,
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(rain_dir / "rain_estimate.png", dpi=150)
    plt.show()
    plt.close(fig)


    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellite")
    parser.add_argument("--rain", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.rain is not None and args.reference is not None:
        estimate_rain(args.rain, args.reference)
    elif args.satellite is not None and args.rain is None and args.reference is None:
        root = Path(__file__).resolve().parent / "outputs"
        estimate_rain(root / "rain" / args.satellite, root / "reference" / args.satellite)
    else:
        parser.error("Provide --satellite or both --rain and --reference")


if __name__ == "__main__":
    main()
