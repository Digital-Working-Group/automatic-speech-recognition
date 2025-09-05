from asr import predict_asr,predict_asr_dataset

def main():
    """
    Runs ASR 
    """
    kwargs = {"model_id": "linto-ai/whisper-timestamped",
              "input_path": "sample_files"}
    
    old_kwargs = {"model_id": "linto-ai/whisper-timestamped",
              "dataset_id": "amaai-lab/DisfluencySpeech",
              "dataset_config_name": "default",
              "dataset_split": "train[:5]"}
    
    predict_asr(**kwargs)
    # predict_asr_dataset(**old_kwargs)


if __name__ == "__main__":
    main()