# Automatic Speech Recognition

This repository runs several automaic speech recognition (ASR) models on different datasets.

| Table of Contents |
|---|
| [Installation and Setup](#installation-and-setup)|
| [Speaker Diarization: Usage Example](#usage-example) |
| [Calculate Performance Metrics](#calculate-performance-metrics) |
| [Calculate Performance Metrics: Installation and Setup](#installation-and-setup-1) |
| [Calculate Performance Metrics: Usage Example](#usage-example-1) |
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

## Usage Example
See [main.main()](src/main.py) for usage examples. The `predict_asr()` KWARGS that define the model and data to be passed into the model. For run of `predict_asr()`, there will be two resulting JSON files describing the metadata and the predicted speech.

```python
from asr import predict_asr
kwargs = {"model_id": "YOUR MODEL",
            "dataset_id": "YOUR DATASET",
            "dataset_config_name": "YOUR CONFIG NAME",
            "dataset_split": "YOUR SPLIT"}
predict_asr(**kwargs)
```

Or to run our preset example, you could run:
```python
from main import main
main()
```

This would output two JSON files: one containing metadata including the KWARGS provided to predict_asr() and one containing the written speech predictions.

### Arguments
The `asr.predict_asr()`  a set of keyword arguments that are passed to the dataset and model loaders. The dataset and model loaders can be found in [load_scripts](src/load_scripts/). Each `dataset_load.load()` method makes use of the [HuggingFace datasets library](https://huggingface.co/docs/datasets/en/index). Each `model_load.load()` method makes use of the [HuggingFace transformers libary](https://huggingface.co/docs/transformers/en/index). Please review the documentation for each library if you wish to modify the data or model loaders.

`model_id` and `dataset_id` are the ids of the ASR model and dataset, respectively, to evaluate.
See [Available Models](#available-models) and [Available Datasets](#available-datasets) for a list of currently implemented models and datasets. See HuggingFace for more information about each model and dataset.

`dataset_config_name` and `dataset_split` are dataset-specific names for different subsets of a dataset.
See the dataset card on HuggingFace for the config name and splits defined in a particular dataset.
Generally, for spoken language datasets, the config name is a language subset and the split is one of train, dev, test, and val.

| Keyword Argument | Type | Description | Default Value |
|---|---|---|---|
| model_id | str | The id of the desired model. | None |
| dataset_id | str | The id of the desired dataset. | None |
| dataset_config_name | str | Config name. Dataset specific. | None |
| dataset_split | str | Maps the specific split of data to load. Dataset specific. | None |
| output_base_dir | str | Filepath to the base of the desired output directory. | "/scripts/output" |

### Output Files
The sample hierarchy shows files created by running `main()` using Docker. `predict_asr()` creates two JSON files: one `metadata.json` containing the KWARGS used to produce the output and one `predictions.json` which records the predicted speech in a list.

```
output
└── NbAiLabBeta_nb-whisper-medium-verbatim_on_amaai-lab_DisfluencySpeech
    ├── metadata.json
    └── predictions.json
```
### Models and Datasets
See HuggingFace for more information about each model and dataset.

#### Available Models
##### General ASR
- [openai/whisper-base](https://huggingface.co/openai/whisper-base)
##### Verbatim
- [openai/whisper-base/prompting](https://huggingface.co/openai/whisper-base) (openai/whisper-base with prompting)
- [NbAiLabBeta/nb-whisper-medium-verbatim](https://huggingface.co/NbAiLabBeta/nb-whisper-medium-verbatim)

#### Available Datasets
##### General ASR
- [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14)
##### Verbatim
- [amaai-lab/DisfluencySpeech](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech)

## Citations

need python 3 and ffmpeg

