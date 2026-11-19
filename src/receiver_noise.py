"""Input-referred receiver noise and ITU-R P.618-14 Eq. (69) sky brightness."""
import numpy as np

K_B = 1.380649e-23


def noise_budget(settings, fs):
    model = settings.get("noise_model", "legacy_fixed_temperature")
    figure = float(settings["noise_figure_db"])
    if not np.isfinite([fs, figure]).all() or fs <= 0 or figure < 0:
        raise ValueError("Noise bandwidth must be positive and noise figure nonnegative")
    if model == "legacy_fixed_temperature":
        temperature = float(settings["temperature_k"])
        system = temperature * 10 ** (figure / 10)
        details = dict(noise_model=model, system_noise_temperature_k=system)
    elif model == "atmospheric_brightness":
        attenuation = float(settings["atmospheric_attenuation_db"])
        background = float(settings.get("background_temperature_k", 2.7))
        medium = float(settings.get("mean_radiating_temperature_k", 275.))
        spillover = float(settings.get("spillover_temperature_k", 0.))
        reference = float(settings.get("noise_reference_temperature_k", 290.))
        if (not np.isfinite([attenuation, background, medium, spillover, reference]).all()
                or min(attenuation, background, medium, spillover, reference) < 0):
            raise ValueError("Invalid atmospheric noise parameters")
        transmission = 10 ** (-attenuation / 10)
        sky = medium * (1 - transmission) + background * transmission
        electronics = reference * (10 ** (figure / 10) - 1)
        system = sky + spillover + electronics
        details = dict(noise_model=model, atmospheric_attenuation_db=attenuation,
            sky_noise_temperature_k=sky, receiver_equivalent_noise_temperature_k=electronics,
            system_noise_temperature_k=system,
            noise_approximation="center_frequency_white_PSD_no_sun_ground_pattern_or_interference")
    else:
        raise ValueError("Unknown receiver.noise_model")
    if not np.isfinite(system) or system < 0:
        raise ValueError("Invalid system noise temperature")
    return dict(details, noise_power_w=K_B * system * fs)


def case_receiver_settings(row, settings, case):
    result = dict(settings)
    if settings.get("noise_model") == "atmospheric_brightness":
        # Exclude free-space spreading, antenna gains, reflection and scintillation.
        if not all(k in row for k in ("gas_db", "cloud_db", "rain_db")):
            raise ValueError("Atmospheric noise requires gas, cloud and rain losses")
        result["atmospheric_attenuation_db"] = float(row["gas_db"] + row["cloud_db"])
        if case == "rain":
            result["atmospheric_attenuation_db"] += float(row["rain_db"])
    return result
