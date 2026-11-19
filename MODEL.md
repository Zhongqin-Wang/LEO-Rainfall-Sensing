# Model and data notes

## Satellite geometry and propagation

The geometry stage uses SGP4 with time-matched Earth-orientation data to obtain slant range, elevation, range rate and Doppler. For the moving-satellite channel, the direct component, environmental multipath and rain-scattered component are tracked separately. A fixed scatterer can still produce a time-varying path because the satellite moves. The path delay derivative determines geometric Doppler.

The rain component supports ITU-R P.838 specific attenuation and an optional spherical Mie model with a Marshall–Palmer drop size distribution. The complex model accounts for attenuation and phase on each simulated path. Finite rain cells and external precipitation grids can be sampled along a path. The retrieval is a path-equivalent rain rate, conditional on receiver calibration and the configured rain-layer height. A single link does not resolve a three-dimensional rain field.

## Waveform and receiver

The custom OFDM example uses a known preamble and pilots. The optional public Starlink profile implements the documented synchronisation and pilot structure and uses synthetic payloads. CSI is measured at known tones. Unknown payload tones are represented as missing measurements.

The receiver model includes offsets in carrier frequency, sampling clocks and timing, as well as noise and automatic gain control. Short captures represent independent frames. The continuous configuration carries the clock evolution across frames and can model one to four channels on a shared receiver clock. Channel-dependent phase and antenna geometry are retained.

Geometry, simulated weather and physical channel truth are separate from the measurements used by the receiver processing. Simulated truth serves in diagnostics and comparisons. The measured-IQ validator assesses acquisition and pilot CSI on a reference capture without producing a rainfall measurement.

## Optional data

| Input | Source and use |
| --- | --- |
| Orbital elements | [CelesTrak GP data](https://celestrak.org/NORAD/documentation/gp-data-formats.php) or a compatible dated snapshot |
| Earth orientation | [Astropy IERS tables](https://docs.astropy.org/en/stable/utils/iers.html), or a user-selected IERS-A table |
| Waveform reference | [Public signal reference package](https://arxiv.org/abs/2602.02627), installed by `fetch_reference.py` using the local reference manifest |
| Recorded IQ | The published reference capture identified by `validate_public_iq.py`. Supply the original ZIP in the selected data directory and run the validator with the same directory |
| Weather | User-supplied BOM radar precipitation grids, ERA5 fields and Himawari imagery as required by the chosen workflow |

The simulator separates synthetic link output from experimental recordings. A comparison with recorded IQ verifies acquisition under its test conditions. Validation of rainfall retrieval requires measured satellite signals and independent rainfall observations. When training a rainfall model, keep events and dates separate across training and evaluation to avoid leakage between overlapping windows.
