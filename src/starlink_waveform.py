"""Public Starlink Ku downlink structure, with explicitly synthetic payloads.

Sources: Humphreys et al. (2023), doi:10.1109/TAES.2023.3268610;
Qin et al. (2026), arXiv:2602.02627 and their published reference package.
PSS algorithm adapted from the authors' MIT-licensed genPss.m. Attribution and
MIT / CC-BY-4.0 terms: data/starlink_public_reference/LICENSE.txt.
The simulator implements the publicly documented signal structure.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat

FS = 240_000_000.0
NFFT = 1024
CP = 32
SYMBOLS = 302
BLOCK = NFFT + CP
FRAME_SAMPLES = 320_000
FRAME_RATE_HZ = 750.0
FRAME_GUARD_SAMPLES = FRAME_SAMPLES - BLOCK * SYMBOLS  # 1088
GUTTER = np.array([0, 1, 1022, 1023])
EDGE_PILOTS = np.r_[488:496, 528:536]
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data/starlink_public_reference"
PROFILE = "starlink_public_v1"


def channel_center_hz(channel):
    if int(channel) != channel or not 1 <= int(channel) <= 8:
        raise ValueError("Starlink Ku channel must be an integer in [1, 8]")
    return 10.7e9 + FS / NFFT / 2 + 250e6 * (int(channel) - 0.5)


def validate_config(config):
    link, ofdm = config["link"], config["ofdm"]
    expected = channel_center_hz(link.get("starlink_channel", 6))
    if not np.isclose(float(link["carrier_hz"]), expected, rtol=0, atol=1):
        raise ValueError(f"Selected Starlink channel requires carrier_hz={expected:.4f}")
    if float(link["bandwidth_hz"]) != FS or float(link["subcarrier_spacing_hz"]) != FS / NFFT:
        raise ValueError("Starlink public profile requires 240 MHz sampling and 234375 Hz spacing")
    if int(ofdm["symbols"]) != SYMBOLS or int(ofdm.get("cp_samples", CP)) != CP:
        raise ValueError("Starlink public profile requires 302 symbol slots and a fixed 32-sample CP")


@lru_cache(maxsize=1)
def public_sequences():
    provenance = json.loads((REFERENCE_DIR / "provenance.json").read_text())
    for item in provenance["files"]:
        if hashlib.sha256((REFERENCE_DIR / item["file"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Public Starlink reference checksum mismatch: {item['file']}")
    # Same Fibonacci LFSR and polarity convention as the published generator.
    state = np.array([0, 0, 1, 1, 0, 1, 0])
    sequence = []
    for _ in range(127):
        bit = int(state[2] ^ state[6])
        sequence.append(bit)
        state = np.r_[bit, state[:-1]]
    chips = 2 * np.r_[0, np.asarray(sequence)[::-1]] - 1
    segment = np.exp(-1j * np.pi / 4 - 1j * np.pi / 2 * np.cumsum(chips))
    useful_pss = np.r_[-segment, np.tile(segment, 7)]
    pss = np.r_[-useful_pss[-CP:], useful_pss]
    sss = loadmat(REFERENCE_DIR / "sssVec.mat")["sssVecFull"].ravel()
    rotations = loadmat(REFERENCE_DIR / "referenceTemplate.mat")["referenceTemplateRotations"].T
    if sss.shape != (NFFT,) or rotations.shape != (301, NFFT):
        raise ValueError("Unexpected dimensions in the public Starlink reference")
    source = (REFERENCE_DIR / "genEdgePilots.m").read_text()
    matches = re.findall(r"hexCodes\{(\d+)\}\s*=\s*\[(.*?)\];", source, re.S)
    if [int(m[0]) for m in matches] != list(range(1, 17)):
        raise ValueError("Expected 16 published edge-pilot codes")
    pilots = np.zeros((SYMBOLS, NFFT), dtype=complex)
    for subcarrier, (_, expression) in zip(EDGE_PILOTS, matches):
        code = "".join(re.findall(r"'([0-9A-F]+)'", expression))
        if len(code) != 150:
            raise ValueError("Each edge-pilot code must have 150 hexadecimal digits")
        number = int(code, 16)
        digits = np.array([(number >> (2 * (301 - i))) & 3 for i in range(2, 302)])
        pilots[2:, subcarrier] = np.exp(1j * np.pi / 2 * (digits + 0.5))
    for value in (pss, sss, pilots, rotations):
        value.setflags(write=False)
    return pss, sss, pilots, rotations


def make_starlink_frame(seed, payload_mode="template_qpsk"):
    """Return IQ and only publicly known pilots; payload is never a receiver input.

    The default is a synthetic all-QPSK reference-template scenario with a random
    six-symbol header. It does not model adaptive modulation, header semantics,
    encryption, or measured T-code statistics. Rectangular CP-OFDM pulse shaping
    follows the authors' reference generator; analog RF filtering is not modeled.
    """
    pss, sss, edge, rotations = public_sequences()
    rng = np.random.default_rng(seed)
    grid = np.zeros((SYMBOLS, NFFT), dtype=complex)
    if payload_mode == "template_qpsk":
        grid[1:] = np.exp(1j * np.pi / 2 * rotations.astype(float))
        grid[2:8] = np.exp(1j * np.pi / 2 * rng.integers(0, 4, (6, NFFT)))
    elif payload_mode == "random_qpsk":
        grid[1:] = np.exp(1j * np.pi / 2 * rng.integers(0, 4, (301, NFFT)))
    else:
        raise ValueError("payload_mode must be template_qpsk or random_qpsk")
    grid[:, GUTTER] = 0
    grid[1] = sss
    grid[2:, EDGE_PILOTS] = edge[2:, EDGE_PILOTS]
    blocks = np.fft.ifft(grid, axis=1, norm="ortho")
    blocks = np.c_[blocks[:, -CP:], blocks]
    blocks[0] = pss  # PSS is a native time-domain sequence, not an OFDM pilot row.
    iq = np.r_[blocks.ravel(), np.zeros(FRAME_GUARD_SAMPLES, complex)]
    known = np.zeros_like(grid)
    known[1] = sss
    known[2:] = edge[2:]
    mask = np.abs(known) > 0
    return iq, mask, known


def acquisition_template():
    pss, sss, _, _ = public_sequences()
    time = np.fft.ifft(sss, norm="ortho")
    return np.r_[pss, time[-CP:], time]
