"""Optional integration validation against authors' recorded IQ, not synthetic rain.

External annotations select bursts and evaluate frame start time. Neither their
TOAs nor their decoded payload/clock estimates are given to synchronization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from run_sdr_csi import process_sdr_burst
from src.starlink_waveform import FRAME_SAMPLES,FS

ROOT=Path(__file__).resolve().parent
ARCHIVE_SHA256="e3d32aa07944bed69762db4a968c4ea1e045abc9c0a9e8dd65659ef2c8bcaf2f"
SOURCE="https://rnl-data.ae.utexas.edu/datastore/supplementaryMaterial/qin-starlink-pilots/raw-iq-data/exemplar250-257.zip"


def plot_results(report,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout="constrained")
    for row in report["records"]:
        success=row["status"]=="ok"
        start=row["annotated_frame_start_s"]*1000
        axes[0].broken_barh([(start,min(1000*FRAME_SAMPLES/FS,50-start))],(.5,1),
                            facecolors="tab:blue" if success else "gray")
        axes[0].text(start,.35,str(row["exemplar_index"]),fontsize=8,rotation=45)
        if success:
            with np.load(output/f"frame_{row['exemplar_index']}"/"csi_sdr.npz") as data:
                if "estimated_noise_power_w" in data:
                    raise ValueError("Uncalibrated recorded IQ must not claim noise power in watts")
                frequency=data["frequency_offset_hz"]
                values=data["pilot_csi"][1]
                indices=np.argsort(frequency)
                axes[1].plot(frequency[indices]/1e6,20*np.log10(abs(values[indices])),
                             alpha=.55,lw=.7,label=str(row["exemplar_index"]))
    axes[0].set(xlim=(0,50),ylim=(0,2),yticks=[],xlabel="Time within raw IQ recording (ms)",
                title="Annotated target-frame occupancy: blue = processed; gray = truncated")
    axes[1].set(xlabel="Frequency offset (MHz)",ylabel="Effective SSS CSI amplitude\n(dB ADC counts / nominal symbol)",
                title="Measured signal + propagation + receiver response; no rain inference")
    axes[1].legend(title="Frame",ncol=7,fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("Public recorded Starlink IQ | 250 MSps input, 240 MSps processing")
    fig.savefig(output/"recorded_iq_validation.png",dpi=170)
    fig.savefig(output/"recorded_iq_validation.svg")
    plt.close(fig)


def validate(data_dir,output):
    data_dir,output=Path(data_dir),Path(output)
    archive=data_dir/"exemplar250-257.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_SHA256:
        raise ValueError("Public experimental IQ archive checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        for name in ("exemplar250-257.bin","exemplar250-257.json","sliceInfo.json","README.txt"):
            expected=z.read("exemplar250-257/"+name)
            path=data_dir/name
            if not path.exists(): path.write_bytes(expected)
            elif path.read_bytes()!=expected: raise ValueError(f"Extracted data changed: {name}")
    output.mkdir(parents=True,exist_ok=True)
    annotations=json.loads((data_dir/"sliceInfo.json").read_text())
    meta=json.loads((data_dir/"exemplar250-257.json").read_text())
    duration=(data_dir/"exemplar250-257.bin").stat().st_size/4/float(meta["rate"])
    rows=[]
    for index,toa in zip(annotations["frameSet"]["index"],annotations["frameSet"]["toa_s"]):
        row=dict(exemplar_index=index,annotated_frame_start_s=toa)
        if toa+FRAME_SAMPLES/FS+50e-6>duration:
            row["status"]="incomplete_at_file_boundary"
        else:
            try:
                result=process_sdr_burst(data_dir/"exemplar250-257.bin",data_dir/"exemplar250-257.json",
                    output/f"frame_{index}",max(0.,toa-50e-6),FRAME_SAMPLES/FS+100e-6)
                row.update(status="ok",frame_start_error_s=result["frame_start_s_in_recording"]-toa,
                    total_estimated_cfo_hz=result["total_estimated_cfo_hz"],
                    estimated_residual_sfo_ppm=result["estimated_residual_sfo_ppm"],
                    capture_score=result["capture_score"],synchronization_coherence=result["synchronization_coherence"])
            except (ValueError,RuntimeError,OSError) as error:
                row.update(status="failed",error=str(error))
        rows.append(row)
        print(index,row["status"],flush=True)
    successes=[row for row in rows if row["status"]=="ok"]
    report=dict(source_url=SOURCE,archive_sha256=ARCHIVE_SHA256,input_sample_rate_hz=float(meta["rate"]),
        processing_sample_rate_hz=FS,recording_duration_s=duration,records=rows,
        expected_complete_frames=int(annotations["frameSet"]["nWhole"]),successful_frames=len(successes),
        max_abs_frame_start_error_s=max((abs(r["frame_start_error_s"]) for r in successes),default=None),
        burst_selection="external TOA annotations select windows; blind CFO/time search within each window",
        receiver_aiding="public PSS/SSS/edge pilots only; no decoded payload, TOA or carrier/clock truth passed to receiver",
        amplitude_calibration="unknown; CSI in ADC counts per nominal reference symbol",
        validates="Recorded-IQ import, 250-to-240 conversion, frame acquisition and relative known-pilot CSI extraction",
        does_not_validate="Rainfall accuracy, optimal rain sampling cadence, absolute channel gain, satellite identity or all hardware scenarios")
    (output/"public_iq_validation.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    plot_results(report,output)
    if len(successes)!=report["expected_complete_frames"]:
        raise RuntimeError("Not all complete public frames succeeded; inspect public_iq_validation.json")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/starlink_public_reference/raw_iq")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/sdr_validation/public_iq")
    args=parser.parse_args()
    result=validate(args.data_dir,args.output)
    print(f"Recorded IQ: {result['successful_frames']}/{result['expected_complete_frames']} complete frames succeeded")
