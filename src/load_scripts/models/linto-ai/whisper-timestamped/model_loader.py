import whisper_timestamped as whisper
import transformers
import os


def load_old():
    wt_model = whisper.load_model("tiny", device="cpu")
    transcriber = transformers.pipeline("automatic-speech-recognition", model=wt_model, trust_remote_code=True)
    return lambda dataset: transcriber(dataset["audio"], generate_kwargs={'task': 'transcribe', 'language': 'en'})

def load(**kwargs):
    model = kwargs.get("model", "tiny")
    device = kwargs.get("device", "cpu")
    wt_model = whisper.load_model(model, device=device)

    def transcribe_function(file):
        print("Current working directory:", os.getcwd())
        print(file)
        input()
        audio = whisper.load_audio(str(file))
        return whisper.transcribe(wt_model, audio, language="en")
    return transcribe_function



# def load():
#     transcriber = transformers.pipeline("automatic-speech-recognition", model="NbAiLabBeta/nb-whisper-medium-verbatim", trust_remote_code=True)

#     def transcribe_function(dataset):
#         return transcriber(dataset["audio"], generate_kwargs={'task': 'transcribe', 'language': 'en'})
    
#     return transcribe_function

# def load():
#     transcriber = transformers.pipeline("automatic-speech-recognition", model="openai/whisper-base", trust_remote_code=True)
#     return lambda dataset: transcriber(dataset["audio"])