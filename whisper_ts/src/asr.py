"""
asr.py
scripts for automatic speech recognition
"""
import gc
import json
import torch
from tqdm import tqdm
import importlib
from datetime import datetime
from pathlib import Path
from automatic_speech_recognition.whisper_ts.src.asr_output import write_asr

def run_asr(input_fp_list, **kwargs):
    """
    Run automatic speech recognition
    """
    model_id = kwargs.get("model_id")
    device = kwargs.get("device", "cpu")
    output_types = kwargs.get("output_types", ["json"])
    transcribe_kwargs = kwargs.get("transcribe_kwargs", {})
    reload_every_n = kwargs.get("reload_every_n", 10)
    make_output_path = kwargs.get("make_output_path", None)

    print(f"LOADING MODEL={model_id}")
    model_loader = importlib.import_module("automatic_speech_recognition.whisper_ts.src.load_scripts.models.linto-ai.whisper-timestamped.model_loader")
    model = model_loader.load(model_id=model_id, device=device, **transcribe_kwargs)
    print(f"LOADED MODEL={model_id}")

    print("RUNNING ASR")
    for i, input_fp in tqdm(enumerate(input_fp_list)):
        if i % reload_every_n == 0:
            if i > 0:
                del model
                gc.collect()
                torch.cuda.empty_cache()
            print(f"LOADING MODEL={model_id}")
            model_loader = importlib.import_module("automatic_speech_recognition.whisper_ts.src.load_scripts.models.linto-ai.whisper-timestamped.model_loader")
            model = model_loader.load(model_id=model_id, device=device, **transcribe_kwargs)
            print(f"LOADED MODEL={model_id}")
        input_fp_path = Path(input_fp)
        output_fname = kwargs.get('output_fname', Path(input_fp_path.name).stem)
        output_parent = kwargs.get('output_parent', input_fp_path.parent / "output")
        if not make_output_path:
            iso_now = datetime.now().isoformat().replace(':', '-').replace('.', '-')
            output_dir = Path(output_parent) / model_id.replace("/", "_") / iso_now
        else:
            output_dir = make_output_path(output_parent, output_fname)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")        
        prediction = model(input_fp)
        with open(output_dir / "metadata.json", "w") as out_file:
            json.dump({
                "model_id": model_id,
                "input_path": input_fp,
                "transcribe_kwargs": transcribe_kwargs
            }, out_file)
        write_asr(output_types, prediction, output_dir / output_fname)
        print(f"After transcribe - allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Peak memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")    	
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

	
