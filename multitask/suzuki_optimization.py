from multitask.suzuki_emulator import SuzukiEmulator
from multitask.suzuki_data_utils import get_suzuki_dataset
from summit import *

import typer
from tqdm.auto import tqdm, trange
from pathlib import Path
from typing import Iterable, Tuple, Dict, Union, List, Optional
import pandas as pd
import logging
import json


app = typer.Typer()


@app.command()
def stbo(
    model_name: str,
    benchmark_path: str,
    output_path: str,
    max_experiments: Optional[int] = 20,
    batch_size: Optional[int] = 1,
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
    for i in trange(repeats):
        result = run_stbo(exp, max_iterations=max_iterations, batch_size=batch_size)
        result.save(output_path / f"repeat_{i}.json")


@app.command()
def mtbo(
    benchmark_path: str,
    ct_data_paths: List[str],
    output_path: str,
    max_experiments: Optional[int] = 20,
    batch_size: Optional[int] = 1,
    repeats: Optional[int] = 20,
    print_warnings: Optional[bool] = True,
):
    """Optimization of a Suzuki benchmark with Multitask Bayesian Optimziation"""
    # Load benchmark
    exp = SuzukiEmulator.load(benchmark_path)

    # Load suzuki dataset
    ds = get_suzuki_dataset(ct_data_path, split_catalyst=exp.split_catalyst)

    # Single-Task Bayesian Optimization
    max_iterations = max_experiments // batch_size
    max_iterations += 1 if max_experiments % batch_size != 0 else 0
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    for i in trange(repeats):
        result = run_mtbo(exp, max_iterations=max_iterations, batch_size=batch_size)
        result.save(output_path / f"repeat_{i}.json")


def run_stbo(
    exp: Experiment,
    max_iterations: int = 10,
    batch_size=1,
    categorical_method: str = "one-hot",
):
    """Run Single Task Bayesian Optimization (AKA normal BO)"""
    exp.reset()
    strategy = STBO(exp.domain, categorical_method=categorical_method)
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
):
    """Run Multitask Bayesian optimization"""
    strategy = MTBO(
        exp.domain, pretraining_data=ct_data, categorical_method="one-hot", task=task
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
