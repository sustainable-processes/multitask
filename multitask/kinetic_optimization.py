from multitask.mt import NewSTBO, NewMTBO
from summit import *
import gpytorch

import typer
from numpy.random import default_rng
from tqdm.auto import tqdm, trange
from pathlib import Path
from typing import List, Optional
import pandas as pd
import warnings


app = typer.Typer()
N_RETRIES = 3


@app.command()
def stbo(
    case: int,
    output_path: str,
    noise_level: Optional[float] = 0.0,
    max_experiments: Optional[int] = 20,
    batch_size: Optional[int] = 1,
    brute_force_categorical: Optional[bool] = False,
    repeats: Optional[int] = 20,
):
    """Optimization of a Suzuki benchmark with Single-Task Bayesian Optimziation

    Parameters
    ---------
    case : int
        Number of the MIT case
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
    exp = get_mit_case(case=case, noise_level=noise_level)

    # Single-Task Bayesian Optimization
    max_iterations = max_experiments // batch_size
    max_iterations += 1 if max_experiments % batch_size != 0 else 0
    if brute_force_categorical:
        categorical_method = None
    else:
        categorical_method = "one-hot"
    for i in trange(repeats):
        for j in range(N_RETRIES):
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
            if j == N_RETRIES - 1:
                print(
                    f"Not able to find semi-positive definite matrix after {j} tries. Skipping repeat {i}"
                )


@app.command()
def mtbo(
    case: int,
    ct_cases: List[int],
    output_path: str,
    ct_strategy: Optional[str] = "STBO",
    noise_level: Optional[float] = 0.0,
    ct_noise_level: Optional[float] = 0.0,
    max_experiments: Optional[int] = 20,
    max_ct_experiments: Optional[int] = 20,
    batch_size: Optional[int] = 1,
    ct_batch_size: Optional[int] = 1,
    brute_force_categorical: bool = False,
    ct_brute_force_categorical: bool = False,
    repeats: Optional[int] = 20,
):
    """Optimization of a kinetic model benchmark with Multitask Bayesian Optimziation"""
    # Load benchmark
    exp = get_mit_case(case=case, noise_level=noise_level)

    # Load cotraining cases
    ct_exps = [
        get_mit_case(case=ct_case, noise_level=ct_noise_level) for ct_case in ct_cases
    ]

    # Multi-Task Bayesian Optimization
    max_iterations = max_experiments // batch_size
    max_iterations += 1 if max_experiments % batch_size != 0 else 0
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    opt_task = len(ct_exps)
    if brute_force_categorical:
        categorical_method = None
    else:
        categorical_method = "one-hot"
    if ct_brute_force_categorical:
        ct_categorical_method = None
    else:
        ct_categorical_method = "one-hot"
    for i in trange(repeats):
        for j in range(N_RETRIES):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    big_ds = run_cotraining(
                        ct_exps,
                        ct_strategy=ct_strategy,
                        max_iterations=max_ct_experiments,
                        batch_size=ct_batch_size,
                        categorical_method=ct_categorical_method,
                    )
                    big_ds.to_csv(output_path / f"big_ds_repeat_{i}.csv")
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
            if j == N_RETRIES - 1:
                print(
                    f"Not able to find semi-positive definite matrix at {j} tries. Skipping repeat {i}"
                )


def get_mit_case(case: int, noise_level: float = 0.0) -> Experiment:
    if case == 1:
        return MIT_case1(noise_level=noise_level)
    elif case == 2:
        return MIT_case2(noise_level=noise_level)
    elif case == 3:
        return MIT_case3(noise_level=noise_level)
    elif case == 4:
        return MIT_case4(noise_level=noise_level)
    elif case == 5:
        return MIT_case5(noise_level=noise_level)


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


def run_cotraining(
    ct_exps, ct_strategy, max_iterations, batch_size, categorical_method
) -> DataSet:
    if ct_strategy == "LHS":
        strategy = LHS
    elif ct_strategy == "SOBO":
        strategy = SOBO
    elif ct_strategy == "STBO":
        strategy = NewSTBO

    big_data = []
    for task, ct_exp in tqdm(enumerate(ct_exps)):
        s = strategy(ct_exp.domain)
        r = Runner(
            strategy=s,
            experiment=ct_exp,
            max_iterations=max_iterations,
            batch_size=batch_size,
        )
        r.run()
        ct_data = r.experiment.data
        ct_data["task", "METADATA"] = task
        big_data.append(ct_data)
    return pd.concat(big_data)


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


if __name__ == "__main__":
    app()
