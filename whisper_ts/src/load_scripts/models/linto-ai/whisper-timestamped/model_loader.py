"""
model_loader.py
"""
import whisper_timestamped as whisper
import torch

def load(**kwargs):
    """
    load a model
    """
    model_id = kwargs.get("model_id")
    device = kwargs.get("device", "cpu")
    wt_model = whisper.load_model(model_id, device=device)
    print(f"model id loading: {model_id}")

    transcribe_kwargs = {
        "beam_size": kwargs.get("beam_size", 5), 
        "best_of": kwargs.get("best_of", 5),
        "temperature": kwargs.get("temperature", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
        "language": kwargs.get("language", "en"),
        "fp16": kwargs.get("fp16", False),
        "vad": kwargs.get("vad", True) }

    if transcribe_kwargs.get("vad"):
        torch.hub.load(
        'snakers4/silero-vad',
        'silero_vad',
        trust_repo=True,
        force_reload=False)

    def transcribe_function(file, **transcribe_kwargs):
        """
        transcribe function helper
        """
        audio = whisper.load_audio(str(file))
        return whisper.transcribe(wt_model, audio, **transcribe_kwargs)
    return lambda file: transcribe_function(file, **transcribe_kwargs)
