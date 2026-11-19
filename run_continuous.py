"""Finite continuous SDR record, default 100 ms, with pilot-only per-frame CSI."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml
from astropy.time import Time,TimeDelta
from scipy.interpolate import CubicHermiteSpline,CubicSpline
from sgp4.api import SatrecArray

from src.continuous import (TransmitTimeline,make_channel_table,continuous_chunks,write_sc16,process_record)
from src.earth_orientation import ensure_earth_orientation
from src.models import TLEBatch
from src.orbit import load_orbit_batch,_propagate_itrs,add_beam_visibility
from src.propagation import propagation_table
from src.provenance import software_provenance
from src.receive_array import array_layout,channel_count
from src.receiver_noise import case_receiver_settings
from src.sdr_capture import capture_rate
from src.starlink_link import draw_clock
from src.starlink_waveform import FS,NFFT,FRAME_SAMPLES,make_starlink_frame,validate_config,PROFILE
from src.weather import clear_weather,KEYS
from src.bom_rain import fingerprint
from src.orbit_data import sha256_file
from src.antenna import path_antenna_gains

ROOT=Path(__file__).resolve().parent


def setup(config,geometry_dir,norad_id,duration,start_time=None):
    geometry=pd.read_parquet(geometry_dir/'geometry.parquet')
    geometry=geometry[geometry.norad_id==norad_id].sort_values('time_utc').reset_index(drop=True)
    if geometry.empty:
        raise ValueError('Selected satellite has no archived pass')
    metadata=json.loads((geometry_dir/'geometry_metadata.json').read_text())
    if metadata['config']!=json.loads(json.dumps({k:config[k] for k in ['geometry','station','weather']})):
        raise ValueError('Geometry/weather configuration mismatch')
    if config['weather'].get('model') == 'bom':
        if metadata.get('bom_data_sha256') != fingerprint(config['weather']):
            raise ValueError('BOM data changed; rebuild geometry/weather archive')
    geometry=add_beam_visibility(geometry,config['propagation'])
    if start_time is not None:
        requested=pd.Timestamp(start_time)
        if requested.tzinfo is None:
            raise ValueError('Capture start must include timezone')
        geometry=geometry[geometry.time_utc==requested]
        if geometry.empty:raise ValueError('Requested start is not a geometry sample; rebuild geometry at this time')
    geometry=geometry[geometry.in_rx_beam]
    if geometry.empty:
        raise ValueError('No selected satellite samples within RX half-power cone')
    if metadata.get('orbit_sha256',metadata.get('tle_sha256')) != sha256_file(ROOT/config['geometry']['orbit_path']):
        raise ValueError('Orbit archive changed; rebuild geometry')
    weather=pd.read_parquet(geometry_dir/'weather_truth.parquet')
    ensure_earth_orientation(geometry.time_utc.iloc[0].to_pydatetime(),ROOT/config['geometry']['eop_path'])
    if config['propagation']['enabled']['cloud'] and not config['propagation']['cloud'].get('era5_file'):
        raise ValueError('Continuous cloud attenuation requires propagation.cloud.era5_file')
    fc=float(config['link']['carrier_hz'])
    prop=propagation_table(geometry,weather,fc,config['propagation'])
    dry=propagation_table(geometry,clear_weather(weather),fc,config['propagation'])
    records=geometry.merge(prop,on=KEYS,validate='one_to_one').merge(
        weather[KEYS+[c for c in ['rain_rate_mm_h','rain_rate_samples_mm_h'] if c in weather]],on=KEYS,validate='one_to_one')
    records=records.merge(dry[KEYS+['net_loss_db']].rename(columns={'net_loss_db':'reference_loss_db'}),on=KEYS)
    row=records.iloc[int(records.elevation_deg.argmax())]
    batch=load_orbit_batch(ROOT/config['geometry']['orbit_path'], allow_history=config['geometry'].get('orbit_selection')=='latest_available')
    index=int(row.satellite_index)
    single=TLEBatch(np.array([batch.names[index]]),[batch.models[index]],SatrecArray([batch.models[index]]))
    times=np.linspace(-.01,duration+.01,int(np.ceil((duration+.02)/.005))+1)
    valid,itrs=_propagate_itrs(single,Time(row.time_utc.to_pydatetime(),scale='utc')+TimeDelta(times,format='sec'))
    if not valid.all():
        raise ValueError('Invalid SGP4 state inside observation')
    position=np.moveaxis(itrs.cartesian.xyz.to_value('m')[:,0,:],0,1)

    receiver=row[['station_x_km','station_y_km','station_z_km']].to_numpy(float)*1000
    beam=config['propagation']['beam']
    gain=path_antenna_gains(position,receiver,config['propagation'])
    width=float(beam.get('rx_half_power_beamwidth_deg',beam['half_power_beamwidth_deg']))
    if np.any(gain['rx_offaxis_deg']>width/2+1e-9):
        raise ValueError('Capture leaves RX half-power cone; select another start')
    velocity=np.moveaxis(itrs.velocity.d_xyz.to_value('m/s')[:,0,:],0,1)
    track=CubicSpline(times,position,axis=0)
    return row,track,times


def run(config,geometry_dir,output,norad_id=51737,case='rain',duration_s=.1,process=True,start_time=None):
    start=time.monotonic()
    config=deepcopy(config)
    validate_config(config);positions,phases=array_layout(config)
    if not 0<duration_s<=1.:
        raise ValueError('This short-window implementation supports duration in (0,1] seconds')
    if not config['propagation'].get('paper_channel',{}).get('enabled',False):
        raise ValueError('Continuous mode requires the S/X/R paper channel')
    settings=config.get('continuous',{})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    row,track,orbit_times=setup(config,Path(geometry_dir),norad_id,duration_s,start_time)
    if not config['ofdm'].get('motion_enabled',True):
        anchor=track(0.)
        position=lambda t:np.broadcast_to(anchor,(len(t),3)).copy()
        velocity=lambda t:np.zeros((len(t),3))
    else:
        position,velocity=track,track.derivative()
    print('Preparing shared-world continuous path/rain table...',flush=True)
    table=make_channel_table(row,config,position,velocity,duration_s,case,
        step_s=float(settings.get('physics_step_s',.005)))
    print(f"Spectral basis rank {len(table['basis'])}, maximum normalized residual {table['basis_error']:.3g}",flush=True)
    seed=int(config['experiment']['seed'])
    counterfactual=config['experiment'].get('reference_clock_mode','independent')
    if counterfactual not in ('shared','independent'):
        raise ValueError('reference_clock_mode must be shared or independent')
    stream_seed=seed+(101000 if case=='reference' and counterfactual=='independent' else 0)
    tx_clock=draw_clock(config['tx_clock'],stream_seed+711)
    rx_clock=draw_clock(config['clock'],stream_seed+712)
    source=TransmitTimeline(seed+713,float(config['ofdm']['tx_power_w']),config['ofdm'].get('payload_mode','template_qpsk'),
        phase_mode=settings.get('tx_frame_phase_mode','independent'),empty_frames=settings.get('empty_frames',[]),
        origin_samples=int(settings.get('tx_frame_origin_samples',2048)))
    origin_phase=float(np.random.default_rng(stream_seed+714).uniform(-np.pi,np.pi))
    rate=capture_rate(config);count=int(round(duration_s*rate));channels=channel_count(config)
    gains=np.broadcast_to(np.asarray(settings.get('gain_linear',1.),float),(channels,)).copy()
    receiver=case_receiver_settings(row,config['receiver'],case)
    chunks=continuous_chunks(source,table,config,tx_clock,rx_clock,duration_s,stream_seed+715,receiver,
        chunk_samples=int(settings.get('chunk_samples',250000)),origin_phase=origin_phase)
    def progress():
        last=-1
        for first,iq in chunks:
            percent=int(100*(first+len(iq))/count)
            if percent//10>last:
                print(f'Continuous IQ {percent}% ({first+len(iq):,}/{count:,} samples per antenna)',flush=True)
                last=percent//10
            yield first,iq
    metadata=write_sc16(output/'rx_iq_sc16.bin',progress(),count,channels,float(settings.get('adc_full_scale',1e-4)))
    metadata.update(duration_s=count/rate,capture_sample_rate_hz=rate,processing_sample_rate_hz=FS,
        carrier_hz=float(config['link']['carrier_hz']),time_utc=row.time_utc.isoformat(),norad_id=norad_id,
        case=case,counterfactual_clock_mode=counterfactual,
        acquisition_mode='continuous_finite_record',iq_axes=['sample','antenna','I_Q'],
        shared_LO_and_sample_clock=True,antenna_positions_enu_m=positions.tolist(),
        no_intentional_sample_gaps=True,record_frame_origin_known_to_receiver=False,
        traffic_model='configured_occupied_or_empty_slots_not_measured_Starlink_scheduling',
        tx_phase_model='unknown_per_frame_phase_stress_scenario' if source.phase_mode=='independent' else 'continuous_no_frame_jumps_control',
        receiver_gain_model='fixed_voltage_gain_for_entire_record',
        ADC_model='fixed_uniform_sc16_quantizer_not_measured_RF_frontend',waveform_profile=PROFILE,
        interpolation_model='fixed_1024_tap_spectral_basis_then_8x_Kaiser_2049_tap_anti_imaging_and_cubic_sampling',
        interpolation_response_note='declared numerical response; near-Nyquist rolloff; not a measured SDR/LNB filter',
        source_configuration_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
    metadata['orbit_record_index']=int(row.satellite_index)
    for key in ['orbit_epoch_utc','orbit_creation_utc']:
        if key in row:
            metadata[key]=pd.Timestamp(row[key]).isoformat()
    (output/'capture_metadata.json').write_text(json.dumps(metadata,indent=2))
    (output/'receiver_log.json').write_text(json.dumps(dict(agc_gain_linear=gains.tolist(),gain_mode='fixed'),indent=2))
    # Receiver references contain only known pilots, never generated payloads.
    _,mask,known=make_starlink_frame(0)
    np.savez_compressed(output/'frame_config.npz',pilot_mask=mask,pilot_values=known*source.scale,
        waveform_profile=PROFILE,sample_rate_hz=FS)
    # Public orbit prior plus declared, self-consistent position/velocity errors.
    station=row[['station_x_km','station_y_km','station_z_km']].to_numpy(float)*1000
    prediction=config.get('orbit_prediction',{})
    pe=np.asarray(prediction.get('position_error_m',[0.,0.,0.]),float)
    ve=np.asarray(prediction.get('velocity_error_m_s',[0.,0.,0.]),float)
    predicted=position(orbit_times)+pe+orbit_times[:,None]*ve
    predicted_velocity=velocity(orbit_times)+ve
    vectors=predicted-station;distances=np.linalg.norm(vectors,axis=1)
    rates=np.sum(vectors/distances[:,None]*predicted_velocity,axis=1)
    np.savez_compressed(output/'orbit_prediction.npz',time_s=orbit_times,range_m=distances,range_rate_m_s=rates,
        basis='public_GP_with_declared_prediction_errors_not_simulated_clock_or_channel_truth')
    truth=output/'truth';truth.mkdir(exist_ok=True)
    if config['weather'].get('model') == 'bom':
        from src.bom_rain import reader
        label = reader(config['weather']).query_rain(config['station'], row.time_utc)
        label['interpretation'] = 'Nearest BOM surface grid rate, not path-effective rain or measured CSI'
        label['query_time_utc'] = row.time_utc.isoformat()
        label['bom_data_sha256'] = json.loads((Path(geometry_dir)/'geometry_metadata.json').read_text())['bom_data_sha256']
        (truth/'bom_rain_label.json').write_text(json.dumps(label, indent=2))
    cloud = config['propagation']['cloud']
    cloud_truth = dict(source='era5' if 'cloud_source_time_utc' in row else 'off',
        attenuation_db=float(row.cloud_db), elevation_deg=float(row.elevation_deg),
        temperature_c=float(cloud['temperature_c']))
    if 'cloud_liquid_water_kg_m2' in row:
        cloud_truth['liquid_water_kg_m2'] = float(row.cloud_liquid_water_kg_m2)
    if 'cloud_source_time_utc' in row:
        from src.orbit_data import sha256_file
        path = ROOT/cloud['era5_file']
        cloud_truth.update(source_file=str(path), source_sha256=sha256_file(path),
            source_time_utc=row.cloud_source_time_utc.isoformat(),
            grid_lat_deg=float(row.cloud_grid_lat_deg), grid_lon_deg=float(row.cloud_grid_lon_deg))
    (truth/'cloud_attenuation.json').write_text(json.dumps(cloud_truth,indent=2))
    np.savez_compressed(truth/'path_table.npz',time_s=table['times'],relative_length_m=table['distances'],
        range_rate_m_s=table['rates'],spectral_basis=table['basis'],spectral_weights=table['weights'](table['times']),
        reference_length_m=table['reference_length_m'],frequency_offset_hz=table['frequency'])
    (truth/'simulation_truth.json').write_text(json.dumps(dict(tx_clock=tx_clock,rx_clock=rx_clock,
        receiver_initial_phase_rad=origin_phase,rf_chain_phase_offsets_rad=phases.tolist(),
        tx_frame_indices=list(range(-2,int(np.ceil(duration_s*750))+3)),
        tx_frame_phases_rad=[source.phase(k) for k in range(-2,int(np.ceil(duration_s*750))+3)],
        empty_frames=sorted(source.empty),paths=table['paths'],reference_length_m=table['reference_length_m']),indent=2))
    (output/'configuration_as_run.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    quality=process_record(output) if process else None
    summary=dict(duration_s=metadata['duration_s'],receive_channels=channels,samples_per_antenna=count,
        raw_bytes=(output/'rx_iq_sc16.bin').stat().st_size,receiver=quality,
        spectral_basis_rank=len(table['basis']),spectral_basis_max_relative_error=table['basis_error'],
        geometry='SGP4 position spline; velocity is its derivative; spherical path lengths and rates',
        nongeometric_update_step_s=float(settings.get('physics_step_s',.005)),
        approximations=['short-window frozen point/drop scattering strengths and DSD weights',
            'gas/cloud attenuation fixed at window epoch; individual rain transfers and angular gains updated',
            'shared oscillator phase noise across local excess delays',
            'declared per-frame TX phase jumps, not measured jump statistics',
            'fixed finite anti-imaging response; edge-pilot gain checked but near-Nyquist tones roll off',
            'identical antenna patterns; no coupling, cable dispersion or measured front-end filtering'],
        scope='continuous synthetic IQ and actual pilot receiver, not rainfall estimator performance',
        elapsed_s=time.monotonic()-start,software=software_provenance(ROOT))
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:summary[k] for k in ['duration_s','receive_channels','samples_per_antenna','raw_bytes','receiver','elapsed_s']},indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/continuous_4rx_era5.yaml')
    parser.add_argument('--geometry-dir',type=Path,default=ROOT/'outputs/bom_20261012/geometry')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/bom_20261012/continuous_100ms_4rx')
    parser.add_argument('--norad-id',type=int,default=51737)
    parser.add_argument('--start-time',help='Exact UTC geometry sample for capture start')
    parser.add_argument('--duration-ms',type=float,default=100.)
    parser.add_argument('--channels',type=int,choices=[1,2,3,4])
    parser.add_argument('--case',choices=['rain','reference'],default='rain')
    parser.add_argument('--generate-only',action='store_true')
    parser.add_argument('--process-only',action='store_true')
    args=parser.parse_args()
    if args.process_only:
        print(json.dumps(process_record(args.output),indent=2))
    else:
        config=yaml.safe_load(args.config.read_text())
        if args.channels:
            config['acquisition']['receive_channels']=args.channels
            config['receiver']['rf_chain_phase_offsets_rad']=config['receiver']['rf_chain_phase_offsets_rad'][:args.channels]
            config['receiver']['array']['positions_enu_m']=config['receiver']['array']['positions_enu_m'][:args.channels]
        run(config,args.geometry_dir,args.output,args.norad_id,args.case,args.duration_ms/1000,not args.generate_only,args.start_time)
