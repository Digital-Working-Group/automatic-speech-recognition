import transformers

def load_dataset():
    transcriber = transformers.pipeline("automatic-speech-recognition", model="openai/whisper-base", trust_remote_code=True)
    return lambda dataset: transcriber(dataset["audio"])

def load_file():
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.config.forced_decoder_ids = None