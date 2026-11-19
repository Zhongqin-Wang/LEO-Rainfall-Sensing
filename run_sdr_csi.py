"""Extract one frame's relative pilot CSI from a selected real SDR IQ burst.

No simulation truth, synthetic no-rain reference or unvalidated absolute power
calibration is used. A capture plan repeats this operation at recorded times;
this script does not prescribe an experimentally optimal sampling cadence.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.sdr_capture import read_sc16,convert_rate
from src.starlink_waveform import FS,NFFT,FRAME_SAMPLES,public_sequences
from src.starlink_sync import receive_frame,search_coarse_cfo


def process_sdr_burst(iq_path,metadata_path,output,start_s=0.,duration_s=2*FRAME_SAMPLES/FS):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"csi_sdr.npz").unlink(missing_ok=True)
    (output/"sdr_result.json").unlink(missing_ok=True)
    metadata=json.loads(Path(metadata_path).read_text())
    if metadata.get("otwfmt")!="sc16" or metadata.get("type")!="short":
        raise ValueError("This importer requires interleaved little-endian int16 IQ (sc16/short)")
    rate=float(metadata["rate"]);frequency=float(metadata["freq"])
    if not np.isfinite([rate,frequency]).all() or rate<FS or frequency<=0:
        raise ValueError("Full-channel receiver requires >=240 MSps and a valid RF center frequency")
    raw,recording=read_sc16(iq_path,start_s,duration_s,rate)
    if recording["clipped_sample_fraction"]>0:
        raise ValueError("Selected IQ contains clipped ADC samples; amplitude CSI is not trustworthy")
    iq=convert_rate(raw,rate,FS)
    acquisition=search_coarse_cfo(iq)
    _,sss,edge,_=public_sequences()
    known=edge.copy();known[1]=sss
    result=receive_frame(iq,known,acquisition["coarse_cfo_hz"],1.,noise_source="frame_guard")
    noise_counts_squared=float(result.pop("estimated_noise_power_w"))
    time_s=recording["actual_start_s"]+float(result["frame_start_samples"])/FS
    mask=result["pilot_mask"][2:]
    coefficients=result["pilot_csi"][2:]
    signal_power=float(np.mean((abs(coefficients)**2-result["csi_noise_power"][2:])[mask]))
    # Unknown nominal transmitted amplitude and front-end gain remain in CSI.
    summary=dict(input_file=str(Path(iq_path).resolve()),capture_sample_rate_hz=rate,
        processing_sample_rate_hz=FS,rf_center_hz=frequency,receive_channels=1,
        frame_start_s_in_recording=time_s,frame_complete=True,**recording,**acquisition,
        capture_score=float(result["capture_score"]),
        total_estimated_cfo_hz=acquisition["coarse_cfo_hz"]+float(result["estimated_residual_cfo_hz"]),
        estimated_residual_sfo_ppm=float(result["estimated_residual_sfo_ppm"]),
        synchronization_coherence=float(result["synchronization_coherence"]),
        noise_corrected_pilot_power_counts_squared=signal_power,
        estimated_noise_power_counts_squared=noise_counts_squared,
        noise_model=str(result["noise_model"]),
        csi_units="ADC_counts_per_nominal_unit_reference_symbol_not_absolute_propagation_gain",
        time_origin="file-relative; no UTC inferred from unverified metadata",
        rain_estimation_performed=False,
        limitations="No satellite identity inference, absolute RF calibration, multi-beam separation or measured rain truth")
    np.savez_compressed(output/"csi_sdr.npz",**result,frequency_offset_hz=np.fft.fftfreq(NFFT,1/FS),
                        estimated_noise_power_counts_squared=noise_counts_squared,
                        capture_sample_rate_hz=rate,processing_sample_rate_hz=FS,
                        frame_start_s_in_recording=time_s,csi_units=summary["csi_units"])
    (output/"sdr_result.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--iq",type=Path,required=True)
    parser.add_argument("--metadata",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--start-s",type=float,default=0.)
    parser.add_argument("--duration-s",type=float,default=2*FRAME_SAMPLES/FS)
    args=parser.parse_args()
    print(json.dumps(process_sdr_burst(args.iq,args.metadata,args.output,args.start_s,args.duration_s),indent=2))
