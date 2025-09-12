from asr import predict_asr

def main():
    """
    Runs ASR 
    """
    kwargs = {"model_id": "tiny",
              "output_types": ["json", "csv", "txt"]}
    
    predict_asr(**kwargs)

    kwargs = {"model_id": "NbAiLabBeta/nb-whisper-base-verbatim",
              "output_types": ["json", "csv", "txt"]}
    
    predict_asr(**kwargs)
    
if __name__ == "__main__":
    main()