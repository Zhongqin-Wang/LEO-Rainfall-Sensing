"""Run a simulator command in the selected Python environment."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "geometry": [str(ROOT/"run_geometry.py")],
    "link": [str(ROOT/"run_link.py")],
    "pass": [str(ROOT/"run_pass.py")],
    "csi": [str(ROOT/"run_csi.py")],
    "rain": [str(ROOT/"run_rain.py")],
    "update-orbits": [str(ROOT/"update_orbits.py")],
    "update-eop": [str(ROOT/"update_eop.py")],
    "example": [str(ROOT/"run_example.py")],
    "summarize": [str(ROOT/"summarize_validation.py")],
    "sdr-csi": [str(ROOT/"run_sdr_csi.py")],
    "validate-public-iq": [str(ROOT/"validate_public_iq.py")],
    "export-dataset": [str(ROOT/"export_dataset.py")],
    "validate-research": [str(ROOT/"validate_research.py")],
    "summarize-rain-phase": [str(ROOT/"summarize_rain_phase.py")],
    "continuous": [str(ROOT/"run_continuous.py")],
    "validate-paper-model": [str(ROOT/"validate_paper_model.py")],
}


def main():
    if len(sys.argv)<2 or sys.argv[1] in ("-h","--help"):
        print("Usage: python3 simulate.py COMMAND [arguments]\nCommands: "+", ".join(COMMANDS))
        return
    command=sys.argv[1]
    if command not in COMMANDS:
        raise SystemExit("Unknown command: "+command)
    override=os.environ.get("LEO_SIM_PYTHON")
    candidates=[Path(override)] if override else [ROOT/".venv/bin/python"]
    runtime=next((p for p in candidates if p.is_file() and os.access(p,os.X_OK)),None)
    if runtime is None and (3,10)<=sys.version_info[:2]<=(3,12):
        if all(importlib.util.find_spec(m) is not None for m in ("numpy","scipy","astropy","sgp4","itur","yaml","pyarrow","miepython")):
            runtime=Path(sys.executable)
    if runtime is None:
        raise SystemExit("Use Python 3.10–3.12 to install requirements.txt in .venv, or set LEO_SIM_PYTHON to that environment's Python.")
    os.chdir(ROOT)
    os.execv(str(runtime),[str(runtime),*COMMANDS[command],*sys.argv[2:]])


if __name__=="__main__":
    main()
