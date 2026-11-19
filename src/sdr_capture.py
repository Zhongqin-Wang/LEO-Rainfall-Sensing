"""Full-channel SDR sample-rate interface; no invented hardware calibration.

250 Msps -> 240 Msps follows Qin et al., doi:10.1038/s44459-026-00075-6.
The antialias FIR is an implementation choice, not a measured X410 response.
"""
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import firwin, resample_poly

from src.starlink_waveform import FS


def capture_rate(config):
    settings=config.get("acquisition",{})
    from src.receive_array import channel_count
    channel_count(config)
    rate=float(settings.get("sample_rate_hz",FS))
    if not np.isfinite(rate) or rate<FS:
        raise ValueError("Full-channel SSS + both edge pilots require capture rate >=240 MSps; narrowband needs a different receiver")
    return rate


def convert_rate(iq,input_rate,output_rate):
    """Zero-phase rational resampling; keep the input time origin and amplitude."""
    iq=np.asarray(iq)
    if iq.ndim not in (1,2) or not len(iq) or not np.isfinite(iq).all() or (iq.ndim==2 and not 1<=iq.shape[1]<=4):
        raise ValueError("SDR IQ must be finite [sample] or [sample,1--4 channels]")
    a,b=float(input_rate),float(output_rate)
    if not np.isfinite([a,b]).all() or min(a,b)<=0:
        raise ValueError("Sample rates must be finite and positive")
    if a==b:
        return iq.copy()
    ratio=Fraction(b/a).limit_denominator(1000)
    if not np.isclose(float(ratio),b/a,rtol=1e-12,atol=0) or max(ratio.numerator,ratio.denominator)>1000:
        raise ValueError("Unsupported resampling ratio; choose commensurate capture and processing rates")
    width=max(ratio.numerator,ratio.denominator)
    taps=firwin(2*128*width+1,1/width,window=("kaiser",10.))
    return resample_poly(iq,ratio.numerator,ratio.denominator,window=taps,axis=0)


def read_sc16(path,start_s,duration_s,sample_rate_hz):
    """Read a bounded burst of interleaved little-endian int16 I,Q in ADC counts."""
    path=Path(path)
    if path.stat().st_size%4:
        raise ValueError("sc16 file must contain complete little-endian I,Q pairs")
    if not np.isfinite([start_s,duration_s,sample_rate_hz]).all() or start_s<0 or min(duration_s,sample_rate_hz)<=0:
        raise ValueError("Invalid SDR capture time or sample rate")
    first=int(round(start_s*sample_rate_hz))
    count=int(round(duration_s*sample_rate_hz))
    total=path.stat().st_size//4
    if count<1 or first+count>total:
        raise ValueError("Requested SDR burst extends beyond the recorded IQ")
    raw=np.memmap(path,dtype='<i2',mode='r',offset=first*4,shape=(count,2))
    iq=raw[:,0].astype(float)+1j*raw[:,1].astype(float)
    clipped=np.any((raw==-32768)|(raw==32767),axis=1)
    return iq,dict(first_sample=first,sample_count=count,actual_start_s=first/sample_rate_hz,
                   duration_s=count/sample_rate_hz,clipped_sample_fraction=float(clipped.mean()),
                   units="uncalibrated_ADC_counts",sample_format="interleaved_int16_IQ_little_endian")
