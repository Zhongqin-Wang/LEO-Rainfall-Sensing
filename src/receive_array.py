"""One synchronized SDR with 1--4 spatially separated RX phase centres.

All channels see the same world-fixed scatterers/weather and shared TX/RX
clocks. Geometry is spherical, including local image-source ground reflection;
no half-wavelength spacing or unambiguous AoA claim is imposed.
"""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import shutil
import numpy as np

from src.reflection import station_axes
from src.antenna import path_antenna_gains
from src.paper_channel import build_paper_paths, ground_path, path_geometry, render_components

C = 299792458.


def channel_count(config):
    n = config.get('acquisition', {}).get('receive_channels', 1)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= n <= 4:
        raise ValueError('acquisition.receive_channels must be an integer in [1,4]')
    return int(n)


def array_layout(config):
    n = channel_count(config)
    settings = config.get('receiver', {})
    layout = settings.get('array', {})
    if layout.get('clock_mode', 'shared') != 'shared':
        raise ValueError('The array requires shared LO and sample clock, including downconverters')
    positions = np.asarray(layout.get('positions_enu_m', [[float(m), 0., 0.] for m in range(n)]), float)
    phases = np.asarray(settings.get('rf_chain_phase_offsets_rad', [0.]*n), float)
    if positions.shape != (n,3) or not np.isfinite(positions).all() or not np.array_equal(positions[0], [0.,0.,0.]):
        raise ValueError('Array positions must be finite M x 3 ENU offsets with RX0 at the origin')
    if phases.shape != (n,) or not np.isfinite(phases).all():
        raise ValueError('Provide one fixed RF-chain phase per receive channel')
    if n > 1 and (np.min([np.linalg.norm(a-b) for i,a in enumerate(positions) for b in positions[i+1:]]) <= 0):
        raise ValueError('Receive phase centres must be distinct')
    if np.max(np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)) > 10.:
        raise ValueError('This local array model supports aperture at most 10 m')
    return positions, phases


def _coordinates(row):
    def vector(prefix):
        return row[[prefix+x+'_km' for x in ['x','y','z']]].to_numpy(float)*1000
    return vector('sat_'), vector('station_')


def _difference(vector, shift):
    """Stable ||vector-shift|| - ||vector||, including sub-wavelength baselines."""
    return (np.sum(shift*shift,axis=-1)-2*np.sum(vector*shift,axis=-1))/(np.linalg.norm(vector-shift,axis=-1)+np.linalg.norm(vector,axis=-1))


def channel_scenario(row, paths, config, m):
    """Move RX only; never redraw or translate scatterers for a new channel."""
    positions, phases = array_layout(config)
    satellite, origin = _coordinates(row)
    basis = np.column_stack(station_axes(origin))
    shift = basis@positions[m]
    receiver = origin+shift
    cfg, record = deepcopy(config), row.copy()
    cfg['acquisition']['receive_channels'] = 1
    cfg['receiver'].pop('array', None)
    cfg['receiver']['rf_chain_phase_offsets_rad'] = [float(phases[m])]
    cfg['_weather_origin_itrs_m'] = origin.tolist()
    cfg['_ground_up_itrs'] = basis[:,2].tolist()
    cfg['station']['antenna_height_m'] += float(positions[m,2])
    if cfg['station']['antenna_height_m'] <= 0:
        raise ValueError('All antennas must be above the common ground plane')
    cfg['propagation']['beam']['rx_frame_origin_itrs_m'] = origin.tolist()
    cfg['propagation']['beam']['tx_reference_receiver_itrs_m'] = origin.tolist()
    for k, value in zip(['station_x_km','station_y_km','station_z_km'],receiver/1000):
        record[k] = value
    old_distance = np.linalg.norm(satellite-origin)
    delta = float(_difference(satellite-origin, shift))
    new_distance = old_distance+delta
    old_gain = path_antenna_gains(satellite,origin,config['propagation'])['combined_gain_dbi']
    new_gain = path_antenna_gains(satellite,receiver,cfg['propagation'])['combined_gain_dbi']
    direct_voltage_ratio = old_distance/new_distance*10**((new_gain-old_gain)/20)
    record['_array_direct_loss_correction_db'] = -20*np.log10(direct_voltage_ratio)
    for key in ['net_loss_db','reference_loss_db']:
        if key in record:
            record[key] -= 20*np.log10(direct_voltage_ratio)
    record['slant_range_km'] = new_distance/1000
    velocity = row[['sat_vx_km_s','sat_vy_km_s','sat_vz_km_s']].to_numpy(float)*1000
    if not config['ofdm'].get('motion_enabled',True):
        velocity[:] = 0.
    record['slant_range_rate_km_s'] = np.dot((satellite-receiver)/new_distance,velocity)/1000
    rays = []
    for source in paths:
        if source.get('path_model') == 'ground_specular':
            rays.append(ground_path(record,cfg))
            continue
        path = deepcopy(source)
        point = np.asarray(path['position_itrs_m'])
        point_velocity = np.asarray(path['velocity_itrs_m_s'])
        geometry = path_geometry(satellite,velocity,receiver,np.zeros(3),point,point_velocity,config['link']['carrier_hz'])
        gain = path_antenna_gains(satellite,receiver,cfg['propagation'],point-satellite,point-receiver)
        gain_difference = gain['combined_gain_dbi']-source['tx_gain_dbi']-source['rx_gain_dbi']
        coefficient = complex(source['coefficient_real'],source['coefficient_imag'])
        coefficient *= source['rx_leg_m']/geometry['rx_leg_m']*10**(gain_difference/20)/direct_voltage_ratio
        relative = coefficient*np.exp(-2j*np.pi*config['link']['carrier_hz']*geometry['excess_delay_s'])
        antenna_ratio = float(10**((gain['combined_gain_dbi']-new_gain)/20))
        intrinsic = coefficient/antenna_ratio
        path.update(geometry,coefficient_real=coefficient.real,coefficient_imag=coefficient.imag,
            antenna_relative_amplitude=antenna_ratio,
            intrinsic_coefficient_real=intrinsic.real,intrinsic_coefficient_imag=intrinsic.imag,
            relative_coefficient_real=relative.real,relative_coefficient_imag=relative.imag,
            delay_samples=geometry['excess_delay_s']*config['link']['bandwidth_hz'],
            doppler_difference_hz=geometry['relative_doppler_hz'],
            absolute_doppler_hz=-config['link']['carrier_hz']/C*geometry['range_rate_m_s'],
            tx_gain_dbi=float(gain['tx_gain_dbi']),rx_gain_dbi=float(gain['rx_gain_dbi']))
        rays.append(path)
    return record,cfg,rays,delta/C


def render_array_components(row, paths, satellite_positions_m, times_s, frequency_offsets_hz,
                            carrier_hz, net_loss_db, config, case='rain'):
    """Physical CSI [time,frequency,antenna], with one carrier/delay reference."""
    results = []
    for m in range(channel_count(config)):
        record,cfg,rays,delta = channel_scenario(row,paths,config,m)
        shift_loss = float(record['_array_direct_loss_correction_db'])
        value = render_components(record,rays,satellite_positions_m,times_s,frequency_offsets_hz,
            carrier_hz,net_loss_db+shift_loss,config=cfg,case=case)
        phase = np.exp(-2j*np.pi*(carrier_hz+np.asarray(frequency_offsets_hz))*delta)
        results.append({k: v*phase for k,v in value.items()})
    return {k: np.stack([v[k] for v in results],axis=-1) for k in results[0]}


def generate_array_pair(row,before_row,after_row,config,prediction_batch,out,relative_dir,frame_seed):
    """Generate actual multi-channel IQ, then publish a synchronized parent capture."""
    from src.starlink_link import generate_starlink_pair
    positions,phases = array_layout(config)
    n = len(positions)
    if not config['propagation'].get('paper_channel',{}).get('enabled',False):
        raise ValueError('Multi-channel generation requires the current S/X/R paper channel')
    sources = {case: build_paper_paths(row,config,case) for case in ['rain','reference']}
    for m in range(n):
        record,cfg,_,delay = channel_scenario(row,sources['rain'],config,m)
        before = channel_scenario(before_row,[],config,m)[0]
        after = channel_scenario(after_row,[],config,m)[0]
        rays = {case: channel_scenario(row,p,config,m)[2] for case,p in sources.items()}
        generate_starlink_pair(record,before,after,cfg,prediction_batch,out,
            Path(relative_dir)/'channels'/f'rx{m}',frame_seed,
            _array_context=dict(paths=rays,delay_s=delay,noise_channel=m))
    for case in sources:
        directory = Path(out)/case/relative_dir
        children = [directory/'channels'/f'rx{m}' for m in range(n)]
        lengths = [np.load(p/'rx_iq.npy',mmap_mode='r').shape[0] for p in children]
        count = min(lengths)
        iq = np.stack([np.load(p/'rx_iq.npy')[:count] for p in children],axis=-1)
        np.save(directory/'rx_iq.npy',iq)
        for name in ['frame_config.npz','orbit_prediction.npz','retrieval_config.json']:
            shutil.copyfile(children[0]/name,directory/name)
        logs = [json.loads((p/'receiver_log.json').read_text()) for p in children]
        (directory/'receiver_log.json').write_text(json.dumps(dict(agc_gain_linear=[p['agc_gain_linear'] for p in logs])))
        truth = [json.loads((p/'simulation_truth.json').read_text()) for p in children]
        (directory/'simulation_truth.json').write_text(json.dumps(dict(channels=truth,
            shared_LO_and_sample_clock=True,independent_thermal_noise=True,rf_chain_phase_offsets_rad=phases.tolist()),indent=2))
        with np.load(children[0]/'Hprop.npz') as reference:
            values = {k:reference[k] for k in reference.files}
        values['Hprop'] = np.stack([np.load(p/'Hprop.npz')['Hprop'] for p in children],axis=-1)
        values['taps'] = np.fft.ifft(values['Hprop'],axis=0)
        for key in ['net_loss_db','rain_db','rain_db_by_frequency','rain_phase_rad_by_frequency',
                    'rain_phase_rate_rad_s']:
            values[key] = np.stack([np.load(p/'Hprop.npz')[key] for p in children],axis=-1)
        values['antenna_positions_enu_m'] = positions
        np.savez_compressed(directory/'Hprop.npz',**values)
        meta = json.loads((children[0]/'capture_metadata.json').read_text())
        meta.update(configuration_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest(),
            receive_channels=n,iq_axes=['sample','antenna'],csi_axes=['symbol','subcarrier','antenna'],
            antenna_positions_enu_m=positions.tolist(),shared_LO_and_sample_clock=True,
            array_geometry='spherical_point_paths_and_common_tangent_ground_plane',
            receiver_processing='one_shared_frame_start_CFO_SFO_CPE_solution_from_RX0',
            array_assumptions=['identical co-pointed element patterns','no mutual coupling or cable dispersion',
                'each ray rain integral evaluated against one world-fixed weather field',
                'large spacing can produce angular ambiguity; no AoA estimator claimed'])
        (directory/'capture_metadata.json').write_text(json.dumps(meta,indent=2))


def receive_array_frame(iq,known,gains,predicted_doppler_hz,predicted_time_scale,noise_source='preframe',prefer_earliest=False):
    """Share synchronization fitted on RX0; preserve differential array phase."""
    from src.starlink_sync import receive_frame
    iq,gains = np.asarray(iq),np.asarray(gains,float)
    if (iq.ndim!=2 or not 1<=iq.shape[1]<=4 or gains.shape!=(iq.shape[1],)
            or not np.isfinite(iq).all() or not np.isfinite(gains).all() or np.any(gains<=0)):
        raise ValueError('Array receiver requires finite sample x channel IQ and positive per-channel gains')
    values = []
    for m in range(iq.shape[1]):
        result = receive_frame(iq[:,m]/gains[m],known,predicted_doppler_hz,predicted_time_scale,
                               shared_sync=values[0] if m else None,noise_source=noise_source,prefer_earliest=prefer_earliest)
        values.append(result)
    reference = values[0]
    data = dict(reference)
    for key in ['pilot_csi','pilot_csi_orbit_only','pilot_csi_before_cpe','csi_noise_power',
                'signal_power_by_frequency']:
        data[key] = np.stack([value[key] for value in values],axis=-1)
    for key in ['estimated_noise_power_w','noise_sample_count']:
        data[key] = np.array([value[key] for value in values])
    data.update(receive_channels=iq.shape[1],synchronization_reference_channel=0,
        antenna_reference_product_orbit_only=data['pilot_csi_orbit_only']*data['pilot_csi_orbit_only'][...,:1].conj(),
        antenna_reference_product_note='RX_m times conjugate RX0; common rain phase also cancels; noise and ICI remain',
        synchronization_shared_across_channels=True,csi_axes=['symbol','subcarrier','antenna'],
        channel_quality_note='acquisition score and clock-fit coherence refer to RX0; other channels share its solution')
    return data


def process_array_capture(directory,iq,known,gains,plot=False):
    from src.starlink_waveform import FS,NFFT,CP,SYMBOLS,BLOCK,PROFILE
    directory = Path(directory)
    with np.load(directory/'orbit_prediction.npz') as prediction:
        doppler,scale = float(prediction['predicted_doppler_hz']),float(prediction['predicted_time_scale'])
    result = receive_array_frame(iq,known,gains,doppler,scale)
    np.savez_compressed(directory/'csi_pilots_corrected.npz',**result,waveform_profile=PROFILE,
        frequency_offset_hz=np.fft.fftfreq(NFFT,1/FS),
        symbol_center_s=(np.arange(SYMBOLS)*BLOCK+CP+(NFFT-1)/2)/FS,
        agc_removed=True,agc_gain_used=gains,sample_rate_hz=FS,nfft=NFFT,cp_samples=CP,
        predicted_doppler_hz=doppler,predicted_time_scale=scale,
        csi_definition='known_nonzero_SSS_and_edge_pilots_only_with_shared_array_synchronization')
    if plot:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        for m in range(iq.shape[1]):
            ax.plot(np.angle(result['pilot_csi_before_cpe'][1,:,m]),label=f'RX{m}')
        ax.set(xlabel='Subcarrier',ylabel='SSS CSI phase (rad)')
        ax.legend();fig.tight_layout();fig.savefig(directory/'csi.png',dpi=150);plt.close(fig)
    quality = {key:float(result[key]) for key in ['capture_score','estimated_residual_cfo_hz',
        'estimated_residual_sfo_ppm','synchronization_coherence']}
    quality['receive_channels'] = int(iq.shape[1])
    print('Shared-synchronization array CSI:',quality,flush=True)
    return quality
