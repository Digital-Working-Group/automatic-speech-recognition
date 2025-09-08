# Automatic Speech Recognition

This repository runs several automaic speech recognition (ASR) models on different datasets.

| Table of Contents |
|---|
| [Installation and Setup](#installation-and-setup)|
| [Usage Example](#usage-example) |
| [Models](#models) |
| [Citations](#citations) |

## Installation and Setup

### Without Docker
Check your Python version:
```sh
python --version
```
See [Anaconda](https://www.anaconda.com/download/success) as an option to switch between Python versions. This repository has been tested with Python 3.12.11

Install requirements for Python 3.12.11:
```sh
pip install -r requirements/py-3-12-11/requirements.txt
```

Note: you may use the pip install command described above even if you are working with a different Python version, but you may need to adjust the requirements.txt file to fit any dependencies specific to that Python version.

You will also need to install FFmpeg. See [FFmpeg Setup](#ffmpeg-setup) for instructions. 

## With Docker
[Docker](https://docs.docker.com/engine/install/) is required for building and running the docker container. Docker version 24.0.6, build ed223bc was used to develop and test these scripts.

Run the necessary docker build and run commands provided in the `build_docker.sh` and `run_docker.sh` scripts. These .sh scripts were tested on Linux (CentOS 7).

```sh
./build_docker.sh
./run_docker.sh
```

The Docker commands included in the .sh scripts are:
```sh
docker build -t $docker_name .
## build the container image under the name 'docker_name' based on the Dockerfile specifications
docker run -v $(pwd):/scripts -it --rm --gpus all --name $container_name $docker_name bash
## run the built container image ('docker_name') under the container name ('container_name')
## mounts the current working directory $(pwd) as a volume to /scripts within the container
```

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
See [main.main()](src/main.py) for usage examples. The `predict_asr()` KWARGS that define the model and data to be passed into the model. For each run of `predict_asr()`, there will be one metadata JSON file that captures the KWARGs provided as well as prediction files for each of the input files provided. The prediction files produced depend on the files requested using the kwarg `output_types`, but the default will be a `prediction.json` file for each input file. 

```python
from asr import predict_asr
kwargs = {  "model_id": "YOUR MODEL",
            "input_path": "YOUR FOLDER PATH",
            "output_types", "YOUR DESIRED OUTPUT EXTS"}
predict_asr(**kwargs)
```

Or to run our preset example, you could run:
```python
from main import main
main()
```

This would output a metadata.json file, a prediction.json, prediction.csv, and prediction.txt for each file provided. We only provide one input file, so there will be one of each.

### Arguments
The `asr.predict_asr()` function takes a set of keyword arguments to define input and output paths and to load the model. The model loader can be found in [load_scripts](src/load_scripts/linto-ai/whisper-timestamped/model_loader.py). For more infomration on `whipser.load_model`, please see the [linto-ai/whipser-timestamped documentation](https://github.com/linto-ai/whisper-timestamped).

`model_id` is the id of the ASR model to use during the predictions task.
See [Models](#models) for a list of suggested and compatible models. See HuggingFace for more information about each model.

#### predict_asr: kwargs
| Keyword Argument | Type | Description | Default Value |
|---|---|---|---|
| model_id | str | The id of the desired model. | None |
| device | str | Where operations are run. | "cpu" |
| input_path | str | Path to folder containing input data. | "sample_files" |
| output_base_dir | str | Path to desired output folder. | "/scripts/output" |
| output_types | list | List of desired output filestypes. Choices include: json, csv, txt. | ["json"] |
| transcribe_kwargs | dict | KWARGS to be passed to whisper.transcribe. See the following table for further details. | See following table. |

#### whisper.transcribe: transcribe_kwargs
| Keyword Argument | Type | Description | Default Value |
|---|---|---|---|
| beam_size | int | Number of paths explored at each step. | 5 |
| best_of | int | Used in conjunction with beam_size, takes the highest-scoring. | 5 |
| temperature | tuple | Controls randomness of the model. | (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) |
| language | str | Language of audio. | "en" |
| vad | bool | Perform voice activity detection or remove silences before transcribing. | True |

### Sample Input and Output Files Output Files
Input audio files should be placed in a together in a folder to target. For our example, our sample input is set up as:

```
sample_files
└── first_minute_Sample_HV_Clip.wav
```

The sample hierarchy below shows files created by running `main()` using Docker. `predict_asr()` creates a `metadata.json` containing the KWARGS used to produce the output. It will also produce any output files specified by the `output_types` KWARG containing the predictions calculated by the model. By default, only `predictions.json` will be created, but if you pass in all three file types, the output would be the following:

```
output
├── NbAiLabBeta_nb-whisper-base-verbatim
    ├── metadata.json
    └── first_minute_Sample_HV_Clip
            ├── predictions.csv
            ├── predictions.json
            └── predictions.txt
└── tiny
    ├── metadata.json
    └── first_minute_Sample_HV_Clip
        ├── predictions.csv
        ├── predictions.json
        └── predictions.txt
```
## Models
See HuggingFace or [linto-ai/whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) for more information about each model.

### General ASR
For general ASR, Open-AI-whipser idenitifiers can be passed in without using the full HuggingFace identifier. 
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