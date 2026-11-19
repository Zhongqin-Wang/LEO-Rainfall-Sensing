# Rainfall Sensing with LEO Satellite Signals

An open-source Python simulator for studying how rainfall affects signals from moving low Earth orbit (LEO) satellites. It couples satellite-pass geometry to rain propagation, synthesises paired rain and clear-sky orthogonal frequency-division multiplexing (OFDM) captures, and estimates rain attenuation from receiver-side measurements. The bundled example is synthetic and runs without external orbit or weather downloads.

## Quick start

Use Python 3.12 on macOS or Linux.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_example.py --output outputs/example
```

The example uses a fictional orbit, a single receiver channel and uniform rain. The script runs satellite-pass geometry, generates matched rain and clear-sky captures, estimates channel state information (CSI), and writes the retrieval results.

- `outputs/example/geometry.parquet` contains the simulated line-of-sight geometry.
- `outputs/example/weather_truth.parquet` contains the synthetic weather used by the signal generator.
- `outputs/example/rain/` and `outputs/example/reference/` contain the paired captures and CSI.
- `outputs/example/passes/satellite_0000/rain_timeseries.csv` contains per-capture estimates, truth and processing status. The same folder contains a plot of simulated and estimated attenuation.

## Configure an experiment

Use `configs/example.yaml` as a starting point. Change the receiver site, orbit, rain scenario or waveform, then regenerate geometry before generating captures. Geometry and signal generation check that the orbit and weather configuration match. Use separate output folders for independent runs.

```sh
python run_geometry.py --config configs/example.yaml --output outputs/custom
python run_link.py --config configs/example.yaml --geometry-dir outputs/custom --output outputs/custom --norad-id 99999
python run_pass.py --root outputs/custom --satellite satellite_0000
```

The tool accepts named two-line elements (TLEs) and CelesTrak GP JSON. For a current Starlink orbit snapshot, use `python update_orbits.py --config configs/base.yaml`. For a historical run, supply a dated snapshot and an observation time the snapshot covers. The simulator uses the IERS-A table bundled with Astropy for the offline example. Other scenarios can use an explicit Earth-orientation file.

## Additional workflows

- **Public Starlink waveform profile.** Run `python fetch_reference.py` to install the authors' published waveform references. The downloader verifies the expected hashes. Select a `starlink_public_v1` configuration for frame synchronisation and CSI estimation from documented sequences and pilots.
- **Continuous and multichannel capture.** `run_continuous.py` supports one to four synchronous receiver channels with independent satellite and receiver clocks. Supply a matching geometry directory and the chosen observation time. These configurations can produce large IQ files.
- **Rain scenarios.** Choose uniform rain, time-varying rain, moving rain cells or gridded Australian Bureau of Meteorology (BOM) precipitation. Historical presets require matching local inputs. For BOM and ERA5 workflows, provide the relevant data files and set their paths in the scenario configuration. `run_bom_query.py --help` describes station-grid lookup.
- **Cloud and CSI experiments.** The optional scripts accept user-supplied Himawari, rainfall labels and CSI. Install `requirements-fusion.txt` for the PyTorch models. The scripts' `--help` output describes their input layouts.

See [Model and data notes](MODEL.md) for assumptions, input provenance and interpretation of the outputs.

## Scope and attribution

This is a research simulator. The rainfall estimate is the uniform-equivalent rate along a link under an assumed rain-layer model. The synthetic example checks the processing chain. Rainfall accuracy on measured satellite links requires independent field validation.

The simulator source is provided under the [MIT License](LICENSE). The optional waveform references are distributed by their authors under separate MIT and CC BY 4.0 terms. The downloader places their licensing notice beside the downloaded assets. See the [reference manifest](reference_sources.json) for provenance.
