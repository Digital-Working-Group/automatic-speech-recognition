"""
asr.py
"""

from collections.abc import Iterable
from pathlib import Path
import json
import importlib
from datetime import datetime
from asr_ouput import write_asr_json, write_asr_csv, write_asr_txt

def predict_asr(**kwargs):

    model_id = kwargs.get("model_id")
    device = kwargs.get("device", "cpu")
    input_path = kwargs.get("input_path", "../sample_files")
    output_base_dir = kwargs.get("output_base_dir", "../output")
    output_types = kwargs.get("output_types", ["json"])
    transcribe_kwargs = kwargs.get("transcribe_kwargs", {})

    output_dir = Path(output_base_dir) / (model_id.replace("/", "_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    ## Prepare files
    input_files_path = Path(input_path)
    files = [item.name for item in input_files_path.iterdir() if item.is_file()]

    print("LOADING MODEL")
    model_loader = importlib.import_module(f"load_scripts.models.linto-ai.whisper-timestamped.model_loader")
    model = model_loader.load(model_id=model_id, device=device, **transcribe_kwargs)
    print("MODEL LOADED")

    # Make predictions
    print("MAKING PREDICTIONS")
    predictions = []
    for file in files:
        filepath = f"{input_files_path}/{file}"
        pred = model(filepath)
        predictions.append(pred)

    with open(output_dir / "metadata.json", "w") as out_file:
        json.dump({
            "model_id": model_id,
            "input_path": input_path,
            "transcribe_kwargs": transcribe_kwargs
        }, out_file)

    for prediction in predictions:
        if "json" in output_types:
            write_asr_json(prediction, output_dir)
        if "csv" in output_types:
            write_asr_csv(prediction, output_dir)
        if "txt" in output_types:
            write_asr_txt(prediction, output_dir)

if __name__ == "__main__":
    pass