# Automatic Speech Recognition

This repository runs several automatic speech recognition (ASR) models on different datasets.

| Table of Contents |
|---|
| [Installation and Setup](#installation-and-setup)|
| [Usage Example](#usage-example) |
| [Models](#models) |
| [Citations](#citations) |

## Installation and Setup

### Python Requirements
```
whisper-timestamped
   |-- requirements
   |   |-- py3-10-11
   |   |   |-- Dockerfile
   |   |   |-- build_docker.sh
   |   |   |-- pip-licenses.md
   |   |   |-- pip-licenses_cpu.md
   |   |   |-- requirements.txt
   |   |   |-- requirements_cpu.txt
   |   |   |-- run_docker.sh
```

Check your Python version:
```sh
python --version
```
See [Anaconda](https://www.anaconda.com/download/success) as an option to switch between Python versions. This repository has been tested with Python 3.10.11.

Install requirements for Python 3.10.11:
```sh
pip install -r requirements/py-3-10-11/requirements.txt ## Python 3.10.11 requirements
```

There are specific requirements for Python 3.10.11 when utilizing an environment without CUDA/GPU capabilities, due to an issue with nvidia-cufile-cu12.

```sh
pip install -r requirements/py-3-10-11/requirements_cpu.txt ## Python 3.10.11 CPU-only requirements
```

Note: you may use the pip install command described above even if you are working with a different Python version, but you may need to adjust the requirements.txt file to fit any dependencies specific to that Python version.

FFmpeg is required, please see [FFmpeg Setup](#ffmpeg-setup) for instructions. 

### Requirements.txt License Information
License information for each set of requirements.txt can be found in their respective `pip-licenses.md` file within the requirements/python[version] folders.

### Docker Support
[Docker](https://docs.docker.com/engine/install/) support can be found via the `Dockerfile` and `build_docker.sh` and `run_docker.sh` files within the requirements/python[version] folders.

Please see Docker's documentation for more information ([docker build](https://docs.docker.com/build/), [Dockerfile](https://docs.docker.com/build/concepts/dockerfile/), [docker run](https://docs.docker.com/reference/cli/docker/container/run/)).

## FFmpeg Setup
The following instructions have been taken from [pydub's documentation](http://github.com/jiaaro/pydub?tab=readme-ov-file#getting-ffmpeg-set-up)

> ### Mac (using homebrew):
> 
> #### libav
> brew install libav
> 
> ####    OR    #####
> 
> #### ffmpeg
> brew install ffmpeg
>
> ### Linux (using aptitude):
> 
> #### libav
> apt-get install libav-tools libavcodec-extra
> 
> ####    OR    #####
> 
> #### ffmpeg
> apt-get install ffmpeg libavcodec-extra
>

Windows FFmpeg installation:
1. Download [FFmpeg's package for Windows](https://www.ffmpeg.org/download.html#build-windows).
2. Locate the download location of the ZIP file and extract to your desired destination.
3. Edit your Environment Variables (Windows Key -> Edit environment variables for your account). Edit Path -> New -> Enter the path to \bin in the extracted FFmpeg folder.
4. Verify your installation with the command
   ```sh
   ffmpeg -version
   ```
Please see a few other tutorials on this process for Windows. Note that you may prefer to edit the environment variables for your account and not for the whole system, as shown in some tutorials.
1. [Audacity](https://support.audacityteam.org/basics/installing-ffmpeg).
2. [GeeksforGeeks](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/).
3. [Transloadit](https://transloadit.com/devtips/how-to-install-ffmpeg-on-windows-a-complete-guide/).

## Usage Example
See [src/main.py](src/main.py) for usage examples. See `main_cpu()` and `main_gpu()` for examples of utilizing CPU and GPU devices respectively.

```python
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
```

### Arguments
The `asr.run_asr()` function takes in an audio input filepath (`input_fp`) and a set of keyword arguments to define an output path, the desired model, device, output types, and additional keyword arguments for the specific transcribe function.

The model loader can be found in [load_scripts](src/load_scripts/linto-ai/whisper-timestamped/model_loader.py). For more information on `whisper.load_model`, please see the [linto-ai/whisper-timestamped documentation](https://github.com/linto-ai/whisper-timestamped).

`model_id` is the id of the ASR model to use during the predictions task.
See [Models](#models) for a list of suggested and compatible models.

#### run_asr: kwargs
| Keyword Argument | Type | Description | Default Value |
|---|---|---|---|
| output_fname | str | The desired base filename of the output files. | Basename of input_fp |
| output_parent | str | The desired root folder to place output files. | "output/" in the base directory of input_fp. |
| model_id | str | The id of the desired model. | None |
| device | str | Where operations are run. | "cpu" |
| output_types | list | List of desired output filestypes. Choices include: json, csv, txt. | ["json"] |
| transcribe_kwargs | dict | KWARGS to be passed to whisper.transcribe. See the whisper.transcribe KWARGS table for further details. | See following table. |

#### whisper.transcribe: transcribe_kwargs
| Keyword Argument | Type | Description | Default Value |
|---|---|---|---|
| beam_size | int | Number of paths explored at each step. | 5 |
| best_of | int | Used in conjunction with beam_size, takes the highest-scoring. | 5 |
| temperature | tuple | Controls randomness of the model. | (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) |
| language | str | Language of audio. | "en" |
| vad | bool | Perform voice activity detection or remove silences before transcribing. | True |

### Sample Input and Output Files

```
├───sample_files
│   │   first_ten_Sample_HV_Clip.wav
│   │
│   └───output
│       ├───NbAiLabBeta_nb-whisper-base-verbatim
│       │   └───2025-09-12T18-31-44-013031
│       │           first_ten_Sample_HV_Clip.csv
│       │           first_ten_Sample_HV_Clip.json
│       │           first_ten_Sample_HV_Clip.txt
│       │           metadata.json
│       │
│       └───tiny
│           └───2025-09-12T18-31-32-139162
│                   first_ten_Sample_HV_Clip.csv
│                   first_ten_Sample_HV_Clip.json
│                   first_ten_Sample_HV_Clip.txt
│                   metadata.json
```

## Models
See HuggingFace or [linto-ai/whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) for more information about each model.

### General ASR
For general ASR, Open-AI-whisper identifiers can be passed in without using the full HuggingFace identifier. 
- tiny 
    - See [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) for details.
- base
    - See [openai/whisper-base](https://huggingface.co/openai/whisper-base) for details.
- small 
    - See [openai/whisper-small](https://huggingface.co/openai/whisper-small) for details.
- medium 
    - See [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) for details.
- large-v2
    - See [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) for details.

### Verbatim
To use NbAiLabBeta, please pass in the full HuggingFace identifier.
- [NbAiLabBeta/nb-whisper-tiny-verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-tiny-verbatim)
- [NbAiLabBeta/nb-whisper-base-verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-base-verbatim)
- [NbAiLabBeta/nb-whisper-small-verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-small-verbatim)
- [NbAiLabBeta/nb-whisper-medium-verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-medium-verbatim)
- [NbAiLabBeta/nb-whisper-large-verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-large-verbatim)


## Citations

```bibtex
@misc{lintoai2023whispertimestamped,
  title={whisper-timestamped},
  author={Louradour, J{\'e}r{\^o}me},
  journal={GitHub repository},
  year={2023},
  publisher={GitHub},
  howpublished = {\url{https://github.com/linto-ai/whisper-timestamped}}
}
@article{radford2022robust,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
@article{JSSv031i07,
  title={Computing and Visualizing Dynamic Time Warping Alignments in R: The dtw Package},
  author={Giorgino, Toni},
  journal={Journal of Statistical Software},
  year={2009},
  volume={31},
  number={7},
  doi={10.18637/jss.v031.i07}
}
```