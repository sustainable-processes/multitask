from multitask.suzuki_emulator import SuzukiEmulator
from multitask.suzuki_data_utils import get_suzuki_dataset
from multitask.mt import NewSTBO, NewMTBO
from summit import *
import gpytorch
import torch
import typer
from numpy.random import default_rng
from tqdm.auto import tqdm, trange
from pathlib import Path
from typing import Iterable, Tuple, Dict, Union, List, Optional
import pandas as pd
import logging
import json
import warnings


app = typer.Typer()
N_RETRIES = 5

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler("kinetic_optimization.log")
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(handler)


@app.command()
def stbo(
    model_name: str,
    benchmark_path: str,
    output_path: str,
    max_experiments: Optional[int] = 20,
    batch_size: Optional[int] = 1,
    brute_force_categorical: Optional[bool] = False,
    repeats: Optional[int] = 20,
):
    """Optimization of a Suzuki benchmark with Single-Task Bayesian Optimziation

    Parameters
    ---------
    model_name : str
        Name of the model
    benchmark_path : str
        Path to the benchmark model files
    output_path : str
        Path where the results will be saved
    max_experiments : int
        The maximum number of experiments. Will be used with batch_size
        to determine the number of iterations. Defaults to 20.
    batch_size : int
        The size of experiment batches. Defaults to 1 (i.e., sequential optimization)
    repeats : int
        The number of repeats of the optimization. Defaults to 20.

    """
    # Ouptut path
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    # Load benchmark
    exp = SuzukiEmulator.load(model_name=model_name, save_dir=benchmark_path)

    # Single-Task Bayesian Optimization
    max_iterations = max_experiments // batch_size
    max_iterations += 1 if max_experiments % batch_size != 0 else 0
    if brute_force_categorical:
        categorical_method = None
    else:
        categorical_method = "one-hot"
    for i in trange(repeats):
        done = False
        retries = 0
        while not done:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = run_stbo(
                        exp,
                        max_iterations=max_iterations,
                        batch_size=batch_size,
                        brute_force_categorical=brute_force_categorical,
                        categorical_method=categorical_method,
                    )
                    result.save(output_path / f"repeat_{i}.json")
                    torch.save(
                        result.strategy.model.state_dict(),
                        output_path / f"repeat_{i}_model.pth",
                    )
                    done = True
                except gpytorch.utils.errors.NotPSDError:
                    retries += 1
            if retries >= N_RETRIES:
                logger.info(
                    f"Not able to find semi-positive definite matrix at {retries} tries. Skipping repeat {i}"
                )
                done = True


@app.command()
def mtbo(
    model_name: str,
    benchmark_path: str,
    ct_data_paths: List[str],
    output_path: str,
    max_experiments: Optional[int] = 20,
    batch_size: Optional[int] = 1,
    repeats: Optional[int] = 20,
    print_warnings: Optional[bool] = True,
    brute_force_categorical: bool = False,
):
    """Optimization of a Suzuki benchmark with Multitask Bayesian Optimziation"""
    # Load benchmark
    exp = SuzukiEmulator.load(model_name=model_name, save_dir=benchmark_path)

    # Load suzuki dataset
    ds_list = [
        get_suzuki_dataset(
            ct_data_path,
            split_catalyst=exp.split_catalyst,
            print_warnings=print_warnings,
        )
        for ct_data_path in ct_data_paths
    ]
    for i, ds in enumerate(ds_list):
        ds["task", "METADATA"] = i
    big_ds = pd.concat(ds_list)

    # Multi-Task Bayesian Optimization
    max_iterations = max_experiments // batch_size
    max_iterations += 1 if max_experiments % batch_size != 0 else 0
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    opt_task = len(ds_list)
    if brute_force_categorical:
        categorical_method = None
    else:
        categorical_method = "one-hot"
    for i in trange(repeats):
        done = False
        retries = 0
        while not done:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = run_mtbo(
                        exp,
                        ct_data=big_ds,
                        max_iterations=max_iterations,
                        batch_size=batch_size,
                        task=opt_task,
                        brute_force_categorical=brute_force_categorical,
                        categorical_method=categorical_method,
                    )
                    result.save(output_path / f"repeat_{i}.json")
                    torch.save(
                        result.strategy.model.state_dict(),
                        output_path / f"repeat_{i}_model.pth",
                    )
                    done = True
                except (RuntimeError, gpytorch.utils.errors.NotPSDError):
                    retries += 1
            if retries >= N_RETRIES:
                logger.info(
                    f"Not able to find semi-positive definite matrix at {retries} tries. Skipping repeat {i}"
                )
                done = True


def run_stbo(
    exp: Experiment,
    max_iterations: int = 10,
    batch_size: int = 1,
    brute_force_categorical: bool = False,
    categorical_method: str = "one-hot",
    acquisition_function: str = "EI",
):
    """Run Single Task Bayesian Optimization (AKA normal BO)"""
    exp.reset()
    assert exp.data.shape[0] == 0
    strategy = NewSTBO(
        exp.domain,
        brute_force_categorical=brute_force_categorical,
        categorical_method=categorical_method,
        acquisition_function=acquisition_function
    )
    r = Runner(
        strategy=strategy,
        experiment=exp,
        max_iterations=max_iterations,
        batch_size=batch_size,
    )
    r.run()
    return r


def run_mtbo(
    exp: Experiment,
    ct_data: DataSet,
    max_iterations: int = 10,
    batch_size=1,
    task: int = 1,
    brute_force_categorical: bool = False,
    categorical_method: str = "one-hot",
    acquisition_function: str = "EI",
):
    """Run Multitask Bayesian optimization"""
    exp.reset()
    assert exp.data.shape[0] == 0
    strategy = NewMTBO(
        exp.domain,
        pretraining_data=ct_data,
        task=task,
        brute_force_categorical=brute_force_categorical,
        categorical_method=categorical_method,
        acquisition_function=acquisition_function,
    )
    r = Runner(
        strategy=strategy,
        experiment=exp,
        max_iterations=max_iterations,
        batch_size=batch_size,
    )
    r.run()
    return r


if __name__ == "__main__":
    app()
