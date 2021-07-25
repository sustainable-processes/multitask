from summit.strategies import STBO, MTBO, Transform, LHS, Chimera
from summit.benchmarks import (
    MIT_case1,
    MIT_case2,
    MIT_case3,
    MIT_case4,
    MIT_case5,
    BaumgartnerCrossCouplingEmulator,
    get_pretrained_baumgartner_cc_emulator,
    ExperimentalEmulator,
)
from summit.utils.dataset import DataSet
from summit.domain import *
import summit
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from IPython.display import clear_output
from copy import deepcopy
import pathlib


from multitask.suzuki_emulator import SuzukiEmulator
from multitask.suzuki_data_utils import get_suzuki_dataset
from multitask.mt import NewSTBO, NewMTBO
from summit import *
import gpytorch

import typer
from numpy.random import default_rng
from tqdm.auto import tqdm, trange
from pathlib import Path
from typing import Iterable, Tuple, Dict, Union, List, Optional
import pandas as pd
import logging
import json
import warnings


def generate_mit_case_1_data(n_points):
    exp_pt = MIT_case1(noise_level=1)
    rs = np.random.RandomState(100)
    lhs = LHS(exp_pt.domain, random_state=rs)
    conditions = lhs.suggest_experiments(n_points)
    exp_pt.run_experiments(conditions)
    pt_data = exp_pt.data
    pt_data["task", "METADATA"] = 0
    return pt_data


n_aux = [5, 10, 50]
aux_datasets = [generate_mit_case_1_data(n) for n in n_aux]


app = typer.Typer()
N_REPEATS = 3


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
        for j in range(N_REPEATS):
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
                    break
                except gpytorch.utils.errors.NotPSDError:
                    continue
            if j == N_REPEATS - 1:
                print(
                    f"Not able to find semi-positive definite matrix after {j} tries. Skipping repeat {i}"
                )


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
        for j in range(N_REPEATS):
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
                    break
                except (RuntimeError, gpytorch.utils.errors.NotPSDError):
                    continue
            if j == N_REPEATS - 1:
                print(
                    f"Not able to find semi-positive definite matrix at {j} tries. Skipping repeat {i}"
                )


def run_stbo(
    exp: Experiment,
    max_iterations: int = 10,
    batch_size: int = 1,
    brute_force_categorical: bool = False,
    categorical_method: str = "one-hot",
):
    """Run Single Task Bayesian Optimization (AKA normal BO)"""
    exp.reset()
    assert exp.data.shape[0] == 0
    strategy = NewSTBO(
        exp.domain,
        brute_force_categorical=brute_force_categorical,
        categorical_method=categorical_method,
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
    )
    r = Runner(
        strategy=strategy,
        experiment=exp,
        max_iterations=max_iterations,
        batch_size=batch_size,
    )
    r.run()
    return r