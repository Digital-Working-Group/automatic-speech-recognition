"""
main.py
main entrypoint to run ASR
"""
from asr import run_asr

def main_cpu():
    """
    Runs ASR with CPU device(s)
    """
    input_fp = '../sample_files/first_ten_Sample_HV_Clip.wav'

    kwargs = {'model_id': 'tiny', 'device': 'cpu', 'output_types': ['json', 'csv', 'txt']}
    run_asr(input_fp, **kwargs)

    kwargs = {'model_id': 'NbAiLabBeta/nb-whisper-base-verbatim', 'device': 'cpu',
              'output_types': ['json', 'csv', 'txt']}
    run_asr(input_fp, **kwargs)

def main_gpu():
    """
    Runs ASR with GPU device(s)
    """
    input_fp = '../sample_files/first_ten_Sample_HV_Clip.wav'

    kwargs = {'model_id': 'tiny', 'device': 'cuda:0', 'output_types': ['json', 'csv', 'txt']}
    run_asr(input_fp, **kwargs)

    kwargs = {'model_id': 'NbAiLabBeta/nb-whisper-base-verbatim', 'device': 'cuda:0',
              'output_types': ['json', 'csv', 'txt']}
    run_asr(input_fp, **kwargs)

if __name__ == '__main__':
    main_cpu()
    # main_gpu()
