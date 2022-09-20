# Multitask 

Optimizing reactions using multitask bayesian optimization. Case study with alcohol amination via borrowing hydrogen  plus large-scale in-silico comparison.

## Setup

Clone the project:

    git clone https://github.com/sustainable-processes/multitask.git

Install [poetry](https://python-poetry.org/docs/) and then run the following command to install dependencies:

    poetry install

Additionally, until this [issue](https://github.com/open-reaction-database/ord-schema/issues/600) is fixed, you will need to manually install the ord-schema dependencies:
    
    poetry run pip install -r ord-requirements.txt

You can run commands from inside the poetry virtual environment in one of two ways:

  - **Option 1**: Put `poetry run` in front of every command (e.g., `poetry run dvc repro`) as we did above to install extra requirements.
  - **Option 2**: Activate the virtual environement and run everything as normal:
      - Run `poetry show -v`. Copy the path specified after "Using virtualenv"
      - Activate the virtual environemnt at that path. So, if the path is `/Users/Kobi/Library/Caches/pypoetry/virtualenvs/multitask-z7ErTcQa-py3.7`, you would run `source /Users/Kobi/Library/Caches/pypoetry/virtualenvs/multitask-z7ErTcQa-py3.7/bin/activate`
  
[DVC](https://dvc.org/doc) is used for tracking data and keeping it in sync with git. When you first clone the project, you will need to download the existing data:

    dvc pull

This will ask you to authenticate with Google Drive. Make sure to use your Cambridge account (or an account you shared the [Multitask Bayesian Optimization shared drive](https://drive.google.com/drive/u/2/folders/0AGWGXkw78NfUUk9PVA) with).

### Apple M1

You might run into some issues when installing scientific python packages such as Summit on Apple M1. Follow the steps below to install via pip:

```bash
    arch -arm64 brew install llvm@11 
    brew install hdf5
    HDF5_DIR=/opt/homebrew/opt/hdf5 PIP_NO_BINARY="h5py" LLVM_CONFIG="/opt/homebrew/Cellar/llvm@11/11.1.0_3/bin/llvm-config" arch -arm64 poetry install
```
Replace the llvm path with the version of llvm installed by brew.


### Lightning

```
lightning run app lightning_study.py \
 --name multitask \
 --open-ui False \
--without-server \
--env WANDB_API_KEY= \
```

Add you wandb api key.

## Coding Guidelines

Please format your code using [black](https://github.com/psf/black). Jupyter notebooks, which are for exploration, don't need to be formatted.

**Directory Structure**
```
├── data/  # Store all data and results here (tracked mainly by DVC)
├── dvc.lock # Lock file for DVC
├── dvc.yaml # DVC pipelines configuration
├── figures/  # Store figures in this directory and track using dvc
├── multitask # python package with key functionality
├── nbs # Exploratory Jupyter notebooks
├── ord-requirements.txt # Dependencies for ORD schema
├── params.yaml # Parameters for DVC pipelines
├── poetry.lock  # Poetry lock file
└── pyproject.toml # Poetry configuration file
```
* Put final code for pipelines in `multitask`, and add the code to pipelines as described in the previous section.
* [Typer](https://typer.tiangolo.com/tutorial/first-steps/) is great for turning a python function into a command line script. It's already installed, so you might as well use it.



