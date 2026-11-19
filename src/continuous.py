"""Continuous, finite-duration, full-band IQ for 1--4 synchronized receivers.

One global sample clock, one oscillator realization and one moving scattering
world are used throughout a record. Geometry uses interpolated SGP4 states;
non-geometric per-ray rain/gain transfer is tabulated over the short window.
Unknown transmit frame phase jumps are a declared stress scenario, not a
measured Starlink phase-jump distribution. The receiver never reads truth.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftfreq, next_fast_len
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.ndimage import map_coordinates, spline_filter
from scipy.signal import firwin, fftconvolve, resample_poly

from src.antenna import path_antenna_gains
from src.paper_channel import build_paper_paths, _ground_at
from src.path_rain import ray_rain_spectrum
from src.receive_array import array_layout, channel_scenario, channel_count, receive_array_frame
from src.receiver_noise import noise_budget, case_receiver_settings
from src.sdr_capture import capture_rate, convert_rate
from src.starlink_link import draw_clock
from src.starlink_waveform import (FS, NFFT, FRAME_SAMPLES, FRAME_RATE_HZ, SYMBOLS,
    CP, BLOCK, PROFILE, make_starlink_frame, validate_config)

C = 299792458.


class TransmitTimeline:
    """Deterministic random-access TX samples, independent of output chunking."""
    def __init__(self, seed, power_w=1., payload_mode='template_qpsk', phase_mode='independent',
                 empty_frames=(), origin_samples=2048):
        if phase_mode not in ('independent','none'):
            raise ValueError('TX frame phase mode must be independent or none')
        if not np.isfinite(power_w) or power_w <= 0:
            raise ValueError('TX power must be positive')
        self.seed,self.mode,self.phase_mode = int(seed),payload_mode,phase_mode
        self.empty,self.origin = set(int(x) for x in empty_frames),int(origin_samples)
        self.scale = np.sqrt(power_w*NFFT/(NFFT-4))
        self.cache = OrderedDict()

    def phase(self, index):
        if self.phase_mode == 'none':return 0.
        # Encode signed frame indices without depending on cache traversal.
        key = 2*index if index >= 0 else -2*index-1
        return float(np.random.default_rng(np.random.SeedSequence([self.seed,key,73])).uniform(-np.pi,np.pi))

    def frame(self, index):
        if index not in self.cache:
            key = 2*index if index >= 0 else -2*index-1
            seed = int(np.random.SeedSequence([self.seed,key,17]).generate_state(1)[0])
            x,_,_ = make_starlink_frame(seed,self.mode)
            self.cache[index] = np.zeros_like(x) if index in self.empty else x*self.scale*np.exp(1j*self.phase(index))
            if len(self.cache)>8:self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def read(self, first, count):
        first,count = int(first),int(count)
        if count < 1:raise ValueError('TX source read must be nonempty')
        out = np.empty(count,complex)
        start = first-self.origin
        for k in range(start//FRAME_SAMPLES,(start+count-1)//FRAME_SAMPLES+1):
            lo,hi = max(start,k*FRAME_SAMPLES),min(start+count,(k+1)*FRAME_SAMPLES)
            out[lo-start:hi-start] = self.frame(k)[lo-k*FRAME_SAMPLES:hi-k*FRAME_SAMPLES]
        return out


class InterpolationBank:
    """Prepare a smooth spectral basis once, reuse 8x cubic interpolation per ray."""
    def __init__(self, x, first, frequency_grid, basis):
        self.first = int(first)
        self.coefficients = []
        self.length = len(x)
        # Fixed LTI kernels make results independent of the host read/write
        # block boundaries. The 1024-tap spectral kernel is exact on the
        # declared OFDM frequency grid; halos retain its real neighbouring IQ.
        # The finite anti-imaging filter is an explicit engineering response,
        # not a measured SDR/LNB front end. Edge-pilot gain is tested.
        interpolation_filter = firwin(2*128*8+1,1/8,window=('kaiser',10.))
        for vector in basis:
            kernel = np.fft.fftshift(ifft(vector))
            center = len(kernel)//2
            filtered = fftconvolve(x,kernel,mode='full')[center:center+len(x)]
            high = resample_poly(filtered,8,1,window=interpolation_filter)
            coefficients = (spline_filter(high.real,order=3,mode='constant')
                            +1j*spline_filter(high.imag,order=3,mode='constant'))
            self.coefficients.append(coefficients)

    def sample(self, positions, k):
        return map_coordinates(self.coefficients[k],((positions-self.first)*8)[None,:],
            order=3,mode='constant',cval=0.,prefilter=False)


def spectral_basis(transfers, tolerance=2e-7, max_rank=12):
    """Rank-revealing orthogonal basis; relative error controlled per weak ray too."""
    shape = transfers.shape
    matrix = transfers.reshape(-1,shape[-1])
    norms = np.linalg.norm(matrix,axis=1)
    active = norms>0
    normalized = np.zeros_like(matrix)
    normalized[active] = matrix[active]/norms[active,None]
    residual = normalized.copy()
    vectors = []
    for _ in range(max_rank):
        errors = np.linalg.norm(residual,axis=1)
        index = int(np.argmax(errors))
        if errors[index] <= tolerance:break
        vector = residual[index]/errors[index]
        # Explicit reorthogonalization avoids loss of small rain-shape modes.
        for previous in vectors:vector -= np.vdot(previous,vector)*previous
        vector /= np.linalg.norm(vector)
        vectors.append(vector)
        weights = residual@vector.conj()
        residual -= weights[:,None]*vector
    error = float(np.max(np.linalg.norm(residual,axis=1)))
    if error>tolerance:raise ValueError(f'Spectral basis tolerance not met: {error}')
    basis = np.asarray(vectors)
    coefficients = (matrix@basis.conj().T).reshape(*shape[:-1],len(basis))
    return basis,coefficients,error


def make_channel_table(row,config,satellite_position,satellite_velocity,duration_s,case='rain',step_s=.005):
    """World-fixed rays, exact moving lengths, interpolated non-geometric transfer."""
    positions,phases = array_layout(config)
    times = np.linspace(-.002,duration_s+.002,max(4,int(np.ceil((duration_s+.004)/step_s))+1))
    if config['weather'].get('model') == 'bom' and case == 'rain':
        from src.bom_rain import reader, require_rates
        source_field = reader(config['weather'])
        endpoints = [source_field.query_rain(config['station'],
            pd.Timestamp(row.time_utc)+pd.Timedelta(seconds=float(t))) for t in (times[0], times[-1])]
        for item in endpoints:
            require_rates(item)
        if endpoints[0]['source_time_utc'] != endpoints[1]['source_time_utc']:
            raise ValueError('BOM short window crosses a weather-frame boundary; choose an interior time')
    satellites,velocities = satellite_position(times),satellite_velocity(times)
    source = build_paper_paths(row,config,case)
    npaths = len(source)+1
    frequencies = fftfreq(NFFT,1/FS)
    transfers = np.empty((len(times),len(positions),npaths,NFFT),complex)
    distances = np.empty((len(times),len(positions),npaths))
    rates = np.empty_like(distances)
    origin = row[['station_x_km','station_y_km','station_z_km']].to_numpy(float)*1000
    satellite0 = satellite_position(np.array([0.]))[0]
    reference_length = float(np.linalg.norm(satellite0-origin))
    fc = float(config['link']['carrier_hz'])
    for m in range(len(positions)):
        record,cfg,rays,_ = channel_scenario(row,source,config,m)
        receiver = record[['station_x_km','station_y_km','station_z_km']].to_numpy(float)*1000
        d0 = np.linalg.norm(satellite0-receiver)
        gain0 = path_antenna_gains(satellite0,receiver,cfg['propagation'])['combined_gain_dbi']
        # Remove reference mean rain, then integrate rain along every actual ray.
        base_loss = float(record['net_loss_db'] if case=='rain' else record['reference_loss_db'])
        if case=='rain':base_loss -= float(row.get('rain_db',0.))
        base = 10**(-base_loss/20)
        for j,(t,s,v) in enumerate(zip(times,satellites,velocities)):
            direct = np.linalg.norm(s-receiver)
            direct_gain = path_antenna_gains(s,receiver,cfg['propagation'])['combined_gain_dbi']
            direct_amplitude = base*d0/direct*10**((direct_gain-gain0)/20)
            epoch = pd.Timestamp(row.time_utc)+pd.Timedelta(seconds=float(t))
            for k,path in enumerate([None,*rays]):
                if path is None:
                    points = [s,receiver]
                    length = direct;rate = np.dot((s-receiver)/direct,v)
                    amplitude = direct_amplitude
                elif path.get('path_model')=='ground_specular':
                    reflection = _ground_at(s,receiver,cfg)
                    points = [s,np.asarray(reflection['ground_point_itrs_m']),receiver]
                    length = reflection['lr_m']
                    up = np.asarray(cfg['_ground_up_itrs'])
                    vector = s-receiver+2*cfg['station']['antenna_height_m']*up
                    rate = np.dot(vector/np.linalg.norm(vector),v)
                    amplitude = direct_amplitude*complex(reflection['relative_coefficient_real'],reflection['relative_coefficient_imag'])
                else:
                    velocity = np.asarray(path['velocity_itrs_m_s'])
                    point = np.asarray(path['position_itrs_m'])+t*velocity
                    dt,dr = np.linalg.norm(s-point),np.linalg.norm(point-receiver)
                    points = [s,point,receiver]
                    length = dt+dr
                    rate = np.dot((s-point)/dt,v-velocity)+np.dot((point-receiver)/dr,velocity)
                    gain = path_antenna_gains(s,receiver,cfg['propagation'],point-s,point-receiver)['combined_gain_dbi']
                    coefficient = complex(path['coefficient_real'],path['coefficient_imag'])
                    amplitude = base*coefficient*path['tx_leg_m']*path['rx_leg_m']/(dt*dr)
                    amplitude *= 10**((gain-path['tx_gain_dbi']-path['rx_gain_dbi'])/20)
                rain_loss,rain_phase = ray_rain_spectrum(points,receiver,epoch,cfg,frequencies,case)
                transfers[j,m,k] = amplitude*10**(-rain_loss/20)*np.exp(1j*rain_phase)
                distances[j,m,k] = length-reference_length
                rates[j,m,k] = rate
    basis,weights,error = spectral_basis(transfers)
    # Slow shape/gain is interpolated; geometric carrier phase is never interpolated as wrapped phase.
    delay = CubicHermiteSpline(times,distances/C,rates/C,axis=0)
    coefficient = CubicSpline(times,weights,axis=0)
    return dict(times=times,distances=distances,rates=rates,delays=delay,weights=coefficient,
        basis=basis,basis_error=error,frequency=frequencies,reference_length_m=reference_length,
        paths=source,path_count=npaths,phases=phases,transfers=transfers,
        max_delay_s=float(np.max(abs(distances/C))))


def continuous_chunks(source,table,config,tx_clock,rx_clock,duration_s,seed,case_receiver,
                      chunk_samples=250000,origin_phase=0.):
    """Yield (first_sample, IQ) with state carried across all block boundaries."""
    rate = capture_rate(config)
    channels = channel_count(config)
    count = int(round(duration_s*rate))
    if chunk_samples<1024 or count<1:raise ValueError('Invalid continuous record/block size')
    settings = config.get('continuous',{})
    tx_drift = float(settings.get('tx_lo_drift_hz_s',0.))
    rx_drift = float(settings.get('rx_lo_drift_hz_s',0.))
    gains = np.broadcast_to(np.asarray(settings.get('gain_linear',1.),float),(channels,))
    if np.any(gains<=0) or not np.isfinite(gains).all():raise ValueError('Continuous gains must be positive')
    noise = noise_budget(case_receiver,rate)['noise_power_w']
    phase_rng = np.random.default_rng(np.random.SeedSequence([seed,51]))
    noise_rng = [np.random.default_rng(np.random.SeedSequence([seed,m,91])) for m in range(channels)]
    phase_state = float(origin_phase)
    linewidth = tx_clock['phase_noise_linewidth_hz']+rx_clock['phase_noise_linewidth_hz']
    rx_ratio,tx_ratio = 1+rx_clock['sfo_ppm']*1e-6,1+tx_clock['sfo_ppm']*1e-6
    if not np.isfinite([tx_drift,rx_drift,linewidth]).all() or linewidth<0:
        raise ValueError('Invalid continuous oscillator parameters')
    fc = float(config['link']['carrier_hz'])
    paths = table['path_count']
    delays = [[CubicHermiteSpline(table['times'],table['distances'][:,m,k]/C,table['rates'][:,m,k]/C)
               for k in range(paths)] for m in range(channels)]
    weights = [[CubicSpline(table['times'],table['weights'](table['times'])[:,m,k,:])
               for k in range(paths)] for m in range(channels)]
    halo = 8192
    with ThreadPoolExecutor(max_workers=channels) as workers:
        for first in range(0,count,chunk_samples):
            n = np.arange(first,min(count,first+chunk_samples),dtype=float)
            nominal = n/rate
            rx_time = nominal/rx_ratio
            t = rx_time-rx_clock['timing_offset_samples']/(FS*rx_ratio)
            # Bounds include differential path delay, all clocks and FFT/filter halos.
            bounds = FS*tx_ratio*(np.array([t[0],t[-1]])+np.array([-1,1])*table['max_delay_s'])-tx_clock['timing_offset_samples']
            low,high = int(np.floor(bounds[0]))-halo,int(np.ceil(bounds[1]))+halo
            bank = InterpolationBank(source.read(low,high-low+1),low,table['frequency'],table['basis'])
            output = np.zeros((len(n),channels),complex)
            def render_channel(m):
                channel_signal = np.zeros(len(n),complex)
                for k in range(paths):
                    delay = delays[m][k](t)
                    emission = t-delay
                    indices = FS*tx_ratio*emission-tx_clock['timing_offset_samples']
                    coefficient = weights[m][k](t)
                    filtered = np.zeros(len(n),complex)
                    for r in range(len(table['basis'])):
                        filtered += coefficient[:,r]*bank.sample(indices,r)
                    phase = (-2*np.pi*fc*delay+2*np.pi*(tx_clock['lo_offset_hz']*emission
                        +.5*tx_drift*emission**2-rx_clock['lo_offset_hz']*rx_time-.5*rx_drift*rx_time**2))
                    channel_signal += filtered*np.exp(1j*phase)
                return channel_signal
            for m,signal in enumerate(workers.map(render_channel,range(channels))):
                output[:,m] = signal
            # One accumulated oscillator realization, shared across antennas, never restarted per frame.
            increments = phase_rng.normal(0.,np.sqrt(2*np.pi*linewidth/(rate*rx_ratio)),len(n))
            phases = phase_state+np.cumsum(increments)
            if len(phases):phase_state = float(phases[-1])
            output *= np.exp(1j*phases[:,None]-1j*table['phases'][None,:])
            for m in range(channels):
                draws = noise_rng[m].normal(size=(len(n),2))
                output[:,m] += np.sqrt(noise/2)*(draws[:,0]+1j*draws[:,1])
            yield first,output*gains[None,:]


def write_sc16(path,chunks,count,channels,full_scale):
    """Sample-major [sample,channel,I/Q], one fixed ADC scale for entire record."""
    if not np.isfinite(full_scale) or full_scale<=0:raise ValueError('ADC full scale must be positive')
    scale = 32767/full_scale
    raw = np.memmap(path,dtype='<i2',mode='w+',shape=(count,channels,2))
    clipped = np.zeros(channels,np.int64);peak = np.zeros(channels)
    next_sample = 0
    for first,iq in chunks:
        if first!=next_sample:raise ValueError('Noncontiguous output block')
        data = np.stack([iq.real,iq.imag],axis=-1)*scale
        clipped += np.count_nonzero(np.any((data>32767)|(data< -32768),axis=-1),axis=0)
        peak = np.maximum(peak,np.max(abs(iq),axis=0))
        raw[first:first+len(iq)] = np.clip(np.rint(data),-32768,32767).astype('<i2')
        next_sample += len(iq)
    if next_sample!=count:raise ValueError('Record length mismatch')
    raw.flush();del raw
    return dict(adc_units_per_count=1/scale,clipped_sample_fraction=(clipped/count).tolist(),
                peak_input_units=peak.tolist(),sample_count=count,receive_channels=channels,
                sample_format='little_endian_sc16_sample_channel_IQ')


def read_record(directory,first,count):
    directory = Path(directory)
    meta = json.loads((directory/'capture_metadata.json').read_text())
    total,channels = int(meta['sample_count']),int(meta['receive_channels'])
    if first<0 or first+count>total or count<1:raise ValueError('Record slice outside capture')
    raw = np.memmap(directory/'rx_iq_sc16.bin',dtype='<i2',mode='r',offset=first*channels*4,shape=(count,channels,2))
    return (raw[:,:,0].astype(float)+1j*raw[:,:,1].astype(float))*meta['adc_units_per_count']


def process_record(directory):
    """Detect successive frames from recorded IQ; no simulation truth access."""
    directory = Path(directory)
    meta = json.loads((directory/'capture_metadata.json').read_text())
    gains = np.asarray(json.loads((directory/'receiver_log.json').read_text())['agc_gain_linear'])
    with np.load(directory/'frame_config.npz') as z:known = z['pilot_values']
    with np.load(directory/'orbit_prediction.npz') as z:
        times,range_m,range_rate = z['time_s'],z['range_m'],z['range_rate_m_s']
    orbit = CubicHermiteSpline(times,range_m-range_m[0],range_rate)
    fc,rate = float(meta['carrier_hz']),float(meta['capture_sample_rate_hz'])
    total,channels = int(meta['sample_count']),int(meta['receive_channels'])
    margin = int(round(4096*rate/FS))
    length = int(np.ceil((FRAME_SAMPLES+8192)*rate/FS))
    first,attempt = 0,0
    frames,records = [],[]
    frame_directory = directory/'frames';frame_directory.mkdir(exist_ok=True)
    (directory/'csi_continuous.npz').unlink(missing_ok=True)
    for old in frame_directory.glob('frame_*.npz'):
        old.unlink()
    pilot_mask = abs(known)>0
    symbol_indices,tone_indices = np.where(pilot_mask)
    while first+int(np.ceil((FRAME_SAMPLES+512)*rate/FS)) <= total:
        search_length = length+int(np.ceil(FRAME_SAMPLES*rate/FS)) if attempt==0 else length
        count = min(search_length,total-first)
        t0 = first/rate
        midpoint = min(t0+FRAME_SAMPLES/(2*FS),meta['duration_s'])
        fd = float(-fc/C*orbit.derivative()(midpoint))
        dilation = 1+fd/fc
        quality = dict(attempt=attempt,read_first_sample=first,status='failed')
        try:
            iq = convert_rate(read_record(directory,first,count),rate,FS)
            result = receive_array_frame(iq,known,gains,fd,dilation,noise_source='frame_guard',prefer_earliest=attempt==0)
            start = first+float(result['frame_start_samples'])/dilation*rate/FS
            start_time = start/rate
            # Anchor every local orbit wipe-off to one global predicted phase.
            phase_anchor = float(-2*np.pi*fc/C*orbit(t0))
            correction = np.exp(-1j*phase_anchor)
            packed = {key:result[key][pilot_mask].astype(np.complex64)*np.complex64(correction)
                for key in ['pilot_csi','pilot_csi_orbit_only','pilot_csi_before_cpe']}
            np.savez_compressed(frame_directory/f'frame_{len(frames):04d}.npz',**packed,
                frame_start_sample=start,frame_start_s=start_time,
                global_orbit_phase_at_read_start_rad=phase_anchor,
                estimated_residual_cfo_hz=result['estimated_residual_cfo_hz'],
                estimated_residual_sfo_ppm=result['estimated_residual_sfo_ppm'],
                capture_score=result['capture_score'],cpe_rad=result['cpe_rad'].astype(np.float32))
            frames.append(dict(frame_start_sample=start,frame_start_s=start_time,**packed))
            quality.update(status='ok',frame_index=len(frames)-1,frame_start_s=start_time,
                capture_score=float(result['capture_score']),residual_cfo_hz=float(result['estimated_residual_cfo_hz']),
                residual_sfo_ppm=float(result['estimated_residual_sfo_ppm']))
            next_first = int(np.floor(start+FRAME_SAMPLES/dilation*rate/FS))-margin
        except (ValueError,RuntimeError) as error:
            quality['error'] = str(error)
            next_first = first+int(round(FRAME_SAMPLES/dilation*rate/FS))
        records.append(quality)
        first = max(first+1,next_first);attempt+=1
    if frames:
        values = {key:np.stack([f[key] for f in frames]) for key in ['pilot_csi','pilot_csi_orbit_only','pilot_csi_before_cpe']}
        np.savez_compressed(directory/'csi_continuous.npz',**values,
            frame_start_s=[f['frame_start_s'] for f in frames],
            frame_start_sample=[f['frame_start_sample'] for f in frames],
            symbol_index=symbol_indices,subcarrier_index=tone_indices,
            frequency_offset_hz=fftfreq(NFFT,1/FS)[tone_indices],
            symbol_center_s=(symbol_indices*BLOCK+CP+(NFFT-1)/2)/FS,
            axes=['detected_frame','known_pilot','antenna'],receive_channels=channels,
            invalid_payload_tones='not_stored_not_zero_filled',synchronization_reference_channel=0,
            common_frame_sync_across_antennas=True,agc_removed=True,
            cross_frame_note='global orbit mixing phase retained; TX frame phase jumps and residual clocks are not removed')
    pd.DataFrame(records).to_csv(directory/'frame_detection.csv',index=False)
    result = dict(attempted_windows=len(records),detected_frames=len(frames),receive_channels=channels,
        nominal_frame_slots=meta['duration_s']*FRAME_RATE_HZ,
        pilots_per_detected_frame_per_antenna=int(pilot_mask.sum()),
        expected_boundary_behavior='partial beginning/end frames and unavailable frames may reduce valid count',
        CSI_shape=[len(frames),int(pilot_mask.sum()),channels],raw_duration_s=meta['duration_s'],
        truth_used_by_receiver=False)
    (directory/'receiver_summary.json').write_text(json.dumps(result,indent=2))
    return result
