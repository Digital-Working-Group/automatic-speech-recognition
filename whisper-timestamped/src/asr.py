"""
asr.py
scripts for automatic speech recognition
"""
import json
import importlib
from datetime import datetime
from pathlib import Path
from asr_output import write_asr

def run_asr(input_fp, **kwargs):
    """
    Run automatic speech recognition
    """
    input_fp_path = Path(input_fp)
    output_fname = kwargs.get('output_fname', Path(input_fp_path.name).stem)
    output_parent = kwargs.get('output_parent', input_fp_path.parent / "output")
    model_id = kwargs.get("model_id")
    device = kwargs.get("device", "cpu")
    output_types = kwargs.get("output_types", ["json"])
    transcribe_kwargs = kwargs.get("transcribe_kwargs", {})

    iso_now = datetime.now().isoformat().replace(':', '-').replace('.', '-')
    output_dir = Path(output_parent) / model_id.replace("/", "_") / iso_now
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"LOADING MODEL={model_id}")
    model_loader = importlib.import_module("load_scripts.models.linto-ai.whisper-timestamped.model_loader")
    model = model_loader.load(model_id=model_id, device=device, **transcribe_kwargs)
    print(f"LOADED MODEL={model_id}")

    print("RUNNING ASR")
    prediction = model(input_fp)
    with open(output_dir / "metadata.json", "w") as out_file:
        json.dump({
            "model_id": model_id,
            "input_path": input_fp,
            "transcribe_kwargs": transcribe_kwargs
        }, out_file)
    write_asr(output_types, prediction, output_dir / output_fname)
