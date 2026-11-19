"""Repeatable robustness study. Every attempt, including outages, is retained."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.starlink_waveform import FS, NFFT, EDGE_PILOTS, make_starlink_frame, channel_center_hz
from src.starlink_link import simulate_capture
from src.starlink_sync import receive_frame
from src.provenance import software_provenance

ROOT = Path(__file__).resolve().parent
ZERO = dict(lo_offset_hz=0., sfo_ppm=0., timing_offset_samples=0., phase_noise_linewidth_hz=0.)


def receiver_matrix(output):
    """Sensitivity grid, not a statistical model of deployed hardware/weather.

    Clear-sky input SNR and excess attenuation are varied independently, so
    receiver failure during large fades is counted. The separate pass study
    uses physical atmospheric brightness and frequency-dependent P.838 loss.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    carrier=channel_center_hz(6)
    records=[]
    for snr in (20.,10.,0.,-8.,-15.):
        for attenuation in (0.,3.,10.,20.):
            for seed in range(6):
                doppler=(-1 if seed%2 else 1)*300000.
                tx,mask,known=make_starlink_frame(100+seed,'random_qpsk')
                tx_clock=dict(ZERO,lo_offset_hz=300.,sfo_ppm=.7)
                # Different rainy/reference clocks; no perfect cancellation.
                noise=10**(-100/10)/10**(snr/10)
                settings=dict(temperature_k=noise/(1.380649e-23*FS),noise_figure_db=0.,agc='off')
                row=dict(clear_snr_db=snr,excess_attenuation_db=attenuation,rainy_snr_db=snr-attenuation,
                    seed=seed,doppler_hz=doppler,status='pending',estimated_db=np.nan,error_db=np.nan,error='')
                powers=[]
                for case in range(2):
                    # Physical RX LO sign is opposite its baseband CFO. Keep
                    # the same injected CFO as the pre-audit sensitivity grid.
                    clock=dict(ZERO,lo_offset_hz=(1 if case else -1)*4500.,sfo_ppm=8.-3*case,
                               timing_offset_samples=.37,phase_noise_linewidth_hz=50.+100*case)
                    loss=100+case*attenuation
                    iq,_=simulate_capture(tx,loss,carrier,doppler,3000.,tx_clock,clock,settings,
                        1000+seed+case*131,3000+seed+case*137)
                    try:
                        # Imperfect assistance: deliberately 200 Hz wrong.
                        result=receive_frame(iq,known,doppler+200.,1+(doppler+200.)/carrier)
                        power=float(np.mean(result['signal_power_by_frequency'][EDGE_PILOTS]))
                        if power<=0:
                            raise ValueError('Signal power unresolved after noise subtraction')
                        powers.append(power)
                        row['reference_score' if case==0 else 'rain_score']=result['capture_score']
                    except (ValueError,RuntimeError,FloatingPointError) as error:
                        powers.append(np.nan)
                        row['status']='receiver_or_power_failed'
                        row['error']+=('reference: ' if case==0 else 'rain: ')+str(error)+'; '
                if np.isfinite(powers).all():
                    row['estimated_db']=float(10*np.log10(powers[0]/powers[1]))
                    row['error_db']=row['estimated_db']-attenuation
                    row['status']='ok'
                records.append(row)
            pd.DataFrame(records).to_csv(output/'receiver_matrix.csv',index=False)
            print(f'SNR {snr:g}, excess loss {attenuation:g}: {len(records)}/120 attempted pairs',flush=True)
    data=pd.DataFrame(records)
    summaries=[]
    for (snr,attenuation),group in data.groupby(['clear_snr_db','excess_attenuation_db'],sort=False):
        success=group.status=='ok'
        errors=group.loc[success,'error_db'].to_numpy()
        summaries.append(dict(clear_snr_db=snr,excess_attenuation_db=attenuation,attempts=len(group),
            successes=int(success.sum()),success_fraction=float(success.mean()),
            conditional_bias_db=float(errors.mean()) if errors.size else None,
            conditional_rmse_db=float(np.sqrt(np.mean(errors**2))) if errors.size else None,
            conditional_max_abs_error_db=float(np.max(abs(errors))) if errors.size else None))
    summary=dict(attempted_pairs=len(data),successful_pairs=int((data.status=='ok').sum()),
        failed_pairs=int((data.status!='ok').sum()),seeds_per_condition=6,conditions=summaries,
        interpretation='Small deterministic sensitivity grid, not confidence bounds or field accuracy. Conditional errors exclude failures, success counts include every attempt.',
        calibration='Simultaneous ideal clear control, calibrated gain, scalar excess loss, white noise. No rainfall forecast algorithm is trained.',
        software=software_provenance(ROOT))
    (output/'receiver_matrix_summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    table=pd.DataFrame(summaries)
    for attenuation,group in table.groupby('excess_attenuation_db'):
        group=group.sort_values('clear_snr_db')
        axes[0].plot(group.clear_snr_db,group.success_fraction,'o-',label=f'{attenuation:g} dB fade')
        axes[1].plot(group.clear_snr_db,group.conditional_rmse_db,'o-',label=f'{attenuation:g} dB fade')
    axes[0].set(ylabel='Successful paired retrieval fraction',ylim=(-.05,1.05))
    axes[1].set(ylabel='Attenuation RMSE, successful pairs only (dB)')
    for ax in axes:
        ax.set_xlabel('Clear-control input SNR (dB)');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Receiver sensitivity: 6 seeds / condition; outages retained; synthetic ideal control')
    fig.tight_layout();fig.savefig(output/'receiver_sensitivity.png',dpi=170);fig.savefig(output/'receiver_sensitivity.svg');plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs/research_audit')
    args=p.parse_args()
    summary=receiver_matrix(args.output)
    print(json.dumps({k:summary[k] for k in ('attempted_pairs','successful_pairs','failed_pairs')},indent=2))


if __name__=='__main__':
    main()
