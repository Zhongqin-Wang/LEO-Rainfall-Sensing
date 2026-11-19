"""Coherent rain propagation: dilute spherical drops, Mie + MP size distribution.

Convention: exp(+j*2*pi*f*t), propagation exp(-j*k*s). The excess wavenumber
has a negative imaginary part for passive rain. No artificial random phase,
polarimetric KDP, or off-axis/droplet micro-Doppler is inserted by this model.
See MODEL.md for equations, sources and approximations.
"""
from functools import lru_cache

import miepython
import numpy as np
from numpy.polynomial.legendre import leggauss

C = 299792458.0
MODEL = "mie_spherical_mp"


def water_permittivity(frequency_hz, temperature_c=20.):
    """P.840-7 double-Debye liquid-water permittivity, eps' - j eps''.

    This uses the dielectric submodel, not the P.840 cloud Rayleigh-loss model.
    Restrict the new rain model to liquid water and Ku/Ka microwave scenarios.
    """
    f = np.asarray(frequency_hz, float) / 1e9
    temperature_c = float(temperature_c)
    if (not np.isfinite(f).all() or np.any((f < 1) | (f > 40))
            or not np.isfinite(temperature_c) or not 0 <= temperature_c <= 40):
        raise ValueError("Mie rain requires frequencies 1--40 GHz and water temperature 0--40 C")
    theta = 300. / (temperature_c + 273.15)
    eps0 = 77.66 + 103.3 * (theta - 1)
    eps1, eps2 = .0671 * eps0, 3.52
    fp = 20.20 - 146 * (theta - 1) + 316 * (theta - 1)**2
    fs = 39.8 * fp
    return eps2 + (eps0 - eps1)/(1 + 1j*f/fp) + (eps1 - eps2)/(1 + 1j*f/fs)


def forward_amplitude_m(frequency_hz, diameter_mm, temperature_c=20.):
    """Single-drop forward amplitude in metres, in our exp(+jwt) convention.

    miepython 3.0.2 an_bn returns the conjugate of the Wiscombe amplitude
    convention. Conjugation and -j/k convert to a dimensional forward amplitude.
    Rayleigh limit: f(0) = k**2 * a**3 * (eps-1)/(eps+2).
    Optical theorem: C_ext = -4*pi/k * Im[f(0)] >= 0.
    """
    frequency_hz, diameter_mm = float(frequency_hz), float(diameter_mm)
    eps = complex(water_permittivity(frequency_hz, temperature_c))
    if not np.isfinite(diameter_mm) or not 0 < diameter_mm <= 8:
        raise ValueError("Drop diameter must be in (0,8] mm")
    k = 2*np.pi*frequency_hz/C
    a, b = miepython.an_bn(np.sqrt(eps), k*diameter_mm*1e-3/2)
    orders = np.arange(1, len(a)+1)
    return -1j*np.conj(np.sum((2*orders+1)*(a+b))/2)/k


@lru_cache(maxsize=256)
def _kernel(frequencies, temperature_c, quadrature_order):
    # Fixed physical diameter support; Gaussian weights are in mm, matching
    # N(D)'s m^-3 mm^-1 units. Cache only frequency/drop physics, never weather.
    nodes, weights = leggauss(quadrature_order)
    diameter = .5*(8.-.05)*nodes + .5*(8.+.05)
    weights = .5*(8.-.05)*weights
    amplitude = np.array([[forward_amplitude_m(f, d, temperature_c) for d in diameter]
                          for f in frequencies])
    amplitude *= (2*np.pi/(2*np.pi*np.asarray(frequencies)/C))[:, None]
    return diameter, weights, amplitude


def complex_rain_spectrum(frequency_hz, path_km, rates_mm_h, temperature_c=20.,
                          quadrature_order=96):
    """Return (loss_dB, unwrapped_phase_rad) integrated along equal-length cells.

    N(D) = 8000 exp[-4.1 R^(-.21) D], with R in mm/h and D in mm.
    R is the empirical Marshall--Palmer rate parameter, not a measured DSD.
    At zero rate or path the excess channel is exactly one. No phase wrapping
    is used before spectral interpolation or time differentiation.
    """
    f = np.asarray(frequency_hz, float)
    rates = np.atleast_1d(np.asarray(rates_mm_h, float))
    length = float(path_km)
    if (f.ndim != 1 or not f.size or rates.ndim != 1 or not rates.size
            or not np.isfinite(rates).all() or np.any((rates < 0) | (rates > 500))
            or not np.isfinite(length) or length < 0
            or not isinstance(quadrature_order, (int, np.integer)) or not 32 <= quadrature_order <= 512):
        raise ValueError("Invalid Mie rain path/rates (0--500 mm/h) or quadrature order (32--512)")
    water_permittivity(f, temperature_c)  # Validate even for clear controls.
    if not np.any(rates) or length == 0:
        return np.zeros_like(f), np.zeros_like(f)
    diameter, weights, kernel = _kernel(tuple(f), float(temperature_c), int(quadrature_order))
    active = rates[rates > 0]
    # Average N(D), not N(mean R), for a spatially nonuniform rain field.
    density = np.sum(8000*np.exp(-4.1*active[:, None]**(-.21)*diameter), axis=0)/rates.size
    excess_k = kernel @ (weights*density)  # rad/m; dilute Foldy forward sum.
    loss = -20/np.log(10)*excess_k.imag*length*1000
    phase = -excess_k.real*length*1000
    if not np.isfinite([loss, phase]).all() or np.any(loss < -1e-10):
        raise FloatingPointError("Non-passive or nonfinite coherent rain response")
    return np.maximum(loss, 0.), phase


def settings_from_config(config):
    settings = config.get("rain_microphysics", {})
    unknown = set(settings) - {"temperature_c", "quadrature_order", "spectral_nodes"}
    if unknown:
        raise ValueError(f"Unknown rain_microphysics settings: {sorted(unknown)}")
    order = settings.get("quadrature_order", 96)
    if not isinstance(order, (int, np.integer)) or not 32 <= order <= 512:
        raise ValueError("rain_microphysics.quadrature_order must be an integer in [32,512]")
    return dict(temperature_c=float(settings.get("temperature_c", 20.)), quadrature_order=order)


def row_mie_spectrum(row, config, frequency_offsets_hz, case="rain"):
    """Evaluate smooth microwave response on RF nodes, interpolate unwrapped phase.

    The frequency of these transfer functions is the nominal transmit RF; the
    receiver motion/time resampling is applied afterwards in the IQ channel.
    """
    offsets = np.asarray(frequency_offsets_hz, float)
    if offsets.ndim != 1 or not offsets.size or not np.isfinite(offsets).all():
        raise ValueError("Frequency offsets must be a finite nonempty vector")
    if case not in ("rain", "reference"):
        raise ValueError("Unknown capture case")
    prop = config["propagation"]
    settings = settings_from_config(prop)
    count = prop.get("rain_microphysics", {}).get("spectral_nodes", 33)
    if not isinstance(count, int) or not 9 <= count <= 257:
        raise ValueError("rain_microphysics.spectral_nodes must be an integer in [9,257]")
    # Validate the configured physical domain, including a disabled/clear run.
    carrier = float(config["link"]["carrier_hz"])
    water_permittivity(carrier + offsets, settings["temperature_c"])
    if case == "reference" or not prop["enabled"]["rain"]:
        return np.zeros_like(offsets), np.zeros_like(offsets)
    rates = row.get("rain_rate_samples_mm_h", row.get("rain_rate_mm_h"))
    if rates is None:
        raise ValueError("Coherent rain generation requires weather samples")
    nodes = np.unique(np.r_[np.linspace(offsets.min(), offsets.max(), count), 0.])
    loss, phase = complex_rain_spectrum(carrier+nodes, float(row["rain_path_km"]), rates, **settings)
    return np.interp(offsets, nodes, loss), np.interp(offsets, nodes, phase)
