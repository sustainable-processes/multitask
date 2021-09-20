from multitask.mt import NewSTBO, NewMTBO
from summit import *
import gpytorch
import torch
from uuid import uuid4
import typer
import json
from tqdm.auto import tqdm, trange
from pathlib import Path
from typing import List, Optional
import pandas as pd
import warnings


app = typer.Typer()
N_RETRIES = 3


@app.command()
def stbo(
    case: Optional[int] = 1,
    output_path: Optional[str] = "data/kinetic_models",
    noise_level: Optional[float] = 0.0,
    max_experiments: Optional[int] = 20,
    num_initial_experiments: Optional[int] = 0,
    batch_size: Optional[int] = 1,
    brute_force_categorical: Optional[bool] = False,
    acquisition_function: str = "EI",
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
    args = dict(locals())
    args["strategy"] = "STBO"

    # Ouptut path
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Save command args
    with open(output_path / "args.json", "w") as f:
        json.dump(args, f)

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
        done = False
        retries = 0
        # Retries in case of a cholesky decomposition error
        while not done:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = run_stbo(
                        exp,
                        max_iterations=max_iterations,
                        num_initial_experiments=num_initial_experiments,
                        batch_size=batch_size,
                        brute_force_categorical=brute_force_categorical,
                        acquisition_function=acquisition_function,
                        categorical_method=categorical_method,
                    )
                    result.save(output_path / f"repeat_{i}.json")
                    done = True
                except (RuntimeError, gpytorch.utils.errors.NotPSDError):
                    retries += 1
            if retries >= N_RETRIES:
                print(
                    f"Not able to find semi-positive definite matrix at {retries} tries. Skipping repeat {i}"
                )


@app.command()
def mtbo(
    case: int = 1,
    ct_cases: List[int] = [2],
    output_path: Optional[str] = "data/kinetic_models",
    acquisition_function: str = "EI",
    ct_strategy: Optional[str] = "STBO",
    ct_acquisition_function: str = "EI",
    noise_level: Optional[float] = 0.0,
    ct_noise_level: Optional[float] = 0.0,
    num_initial_experiments: Optional[int] = 0,
    ct_num_initial_experiments: Optional[int] = 0,
    max_experiments: Optional[int] = 20,
    ct_max_experiments: Optional[int] = 20,
    batch_size: Optional[int] = 1,
    ct_batch_size: Optional[int] = 1,
    brute_force_categorical: bool = False,
    ct_brute_force_categorical: bool = False,
    repeats: Optional[int] = 20,
):
    """Optimization of a kinetic model benchmark with Multitask Bayesian Optimziation"""
    args = dict(locals())
    args["strategy"] = "MTBO"
    print("Torch number of threads: ", torch.get_num_threads())

    # Load benchmark
    exp = get_mit_case(case=case, noise_level=noise_level)

    # Load cotraining cases
    ct_exps = [
        get_mit_case(case=ct_case, noise_level=ct_noise_level) for ct_case in ct_cases
    ]

    # Save command args
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    print("Dumping args to:", output_path.resolve())
    with open(output_path / "args.json", "w") as f:
        json.dump(args, f)

    # Multi-Task Bayesian Optimization
    max_iterations = max_experiments // batch_size
    max_iterations += 1 if max_experiments % batch_size != 0 else 0

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
        done = False
        retries = 0
        while not done:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    big_ds = run_cotraining(
                        ct_exps,
                        ct_strategy=ct_strategy,
                        max_iterations=ct_max_experiments - ct_num_initial_experiments,
                        batch_size=ct_batch_size,
                        categorical_method=ct_categorical_method,
                        num_initial_experiments=ct_num_initial_experiments,
                        acquisition_function=ct_acquisition_function,
                    )
                    big_ds.to_csv(output_path / f"big_ds_repeat_{i}.csv")
                    result = run_mtbo(
                        exp,
                        ct_data=big_ds,
                        max_iterations=max_iterations,
                        batch_size=batch_size,
                        task=opt_task,
                        brute_force_categorical=brute_force_categorical,
                        acquisition_function=acquisition_function,
                        num_initial_experiments=num_initial_experiments,
                        categorical_method=categorical_method,
                    )
                    result.save(output_path / f"repeat_{i}.json")
                    done = True
                except (RuntimeError, gpytorch.utils.errors.NotPSDError):
                    retries += 1
            if retries >= N_RETRIES:
                print(
                    f"Not able to find semi-positive definite matrix at {retries} tries. Skipping repeat {i}"
                )


@app.command()
def mtbo_tune(
    case: int,
    ct_cases: List[int],
    output_path: Optional[str] = "data/kinetic_models/mtbo",
    acquisition_function: Optional[List[str]] = ["EI"],
    ct_strategy: Optional[List[str]] = ["STBO"],
    ct_acquisition_function: Optional[List[str]] = ["qNEI"],
    noise_level: Optional[List[float]] = [0.0],
    ct_noise_level: Optional[List[float]] = [0.0],
    num_initial_experiments: Optional[List[int]] = [0],
    ct_num_initial_experiments: Optional[List[int]] = [0],
    max_experiments: Optional[List[int]] = [20],
    ct_max_experiments: Optional[List[int]] = [20],
    batch_size: Optional[List[int]] = [1],
    ct_batch_size: Optional[List[int]] = [1],
    brute_force_categorical: Optional[List[bool]] = [False],
    ct_brute_force_categorical: Optional[List[bool]] = [False],
    repeats: Optional[int] = 20,
    cpus_per_trial: Optional[int] = 4,
):
    from ray import tune
    import ray

    # if ray_head_node_ip is not None:
    #     # Connect to existing ray cluster
    #     import ray

    #     ray.init(f"ray://{ray_head_node_ip}:10001")

    output_path = Path(output_path)

    def trainable(config):
        import torch
        config["output_path"] = str(output_path / str(uuid4()))
        print("Torch number of threads before setting: ", torch.get_num_threads())
        num_threads = config.pop("num_threads")
        #torch.set_num_threads(config.pop("num_threads"))
        print("Torch number of threads: ", torch.get_num_threads())
        mtbo(**config)

    def convert_grid(values):
        if len(values) > 1:
            return tune.grid_search(list(values))
        else:
            return values[0]

    tune_config = {
        "case": case,
        "ct_cases": ct_cases,
        "acquisition_function": convert_grid(acquisition_function),
        "ct_strategy": convert_grid(ct_strategy),
        "ct_acquisition_function": convert_grid(ct_acquisition_function),
        "noise_level": convert_grid(noise_level),
        "ct_noise_level": convert_grid(ct_noise_level),
        "num_initial_experiments": convert_grid(num_initial_experiments),
        "ct_num_initial_experiments": convert_grid(ct_num_initial_experiments),
        "max_experiments": convert_grid(max_experiments),
        "ct_max_experiments": convert_grid(ct_max_experiments),
        "batch_size": convert_grid(batch_size),
        "ct_batch_size": convert_grid(ct_batch_size),
        "brute_force_categorical": convert_grid(brute_force_categorical),
        "ct_brute_force_categorical": convert_grid(ct_brute_force_categorical),
        "repeats": 1,
        "num_threads": cpus_per_trial
    }
    # Run grid search
    tune.run(trainable, num_samples=repeats, config=tune_config, resources_per_trial={"cpu": cpus_per_trial}) 


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
    num_initial_experiments: int = 0,
    max_iterations: int = 10,
    batch_size: int = 1,
    brute_force_categorical: bool = False,
    categorical_method: str = "one-hot",
    acquisition_function: str = "EI",
):
    """Run Single Task Bayesian Optimization (AKA normal BO)"""
    exp.reset()
    assert exp.data.shape[0] == 0
    if num_initial_experiments > 0:
        strategy = LHS(exp.domain)
        suggestions = strategy.suggest_experiments(
            num_experiments=num_initial_experiments
        )
        prev_res = exp.run_experiments(suggestions)
    else:
        prev_res = None
    strategy = NewSTBO(
        exp.domain,
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
    r.run(prev_res=prev_res)
    return r


def run_cotraining(
    ct_exps,
    ct_strategy,
    max_iterations,
    batch_size,
    num_initial_experiments,
    categorical_method,
    acquisition_function,
) -> DataSet:

    big_data = []
    for task, ct_exp in tqdm(enumerate(ct_exps)):
        if num_initial_experiments > 0:
            strategy = LHS(ct_exp.domain)
            suggestions = strategy.suggest_experiments(
                num_experiments=num_initial_experiments
            )
            prev_res = ct_exp.run_experiments(suggestions)
        else:
            prev_res = None

        if ct_strategy == "LHS":
            strategy = LHS(ct_exp.domain)
        elif ct_strategy == "SOBO":
            strategy = SOBO(ct_exp.domain)
        elif ct_strategy == "STBO":
            strategy = NewSTBO(
                ct_exp.domain,
                categorical_method=categorical_method,
                acquisition_function=acquisition_function,
            )

        r = Runner(
            strategy=strategy,
            experiment=ct_exp,
            max_iterations=max_iterations,
            batch_size=batch_size,
        )
        r.run(prev_res=prev_res)
        ct_data = r.experiment.data
        ct_data["task", "METADATA"] = task
        big_data.append(ct_data)
    return pd.concat(big_data)


def run_mtbo(
    exp: Experiment,
    ct_data: DataSet,
    num_initial_experiments: int = 0,
    max_iterations: int = 10,
    batch_size=1,
    task: int = 1,
    brute_force_categorical: bool = False,
    acquisition_function: str = "EI",
    categorical_method: str = "one-hot",
):
    """Run Multitask Bayesian optimization"""
    exp.reset()
    assert exp.data.shape[0] == 0
    if num_initial_experiments > 0:
        strategy = LHS(exp.domain)
        suggestions = strategy.suggest_experiments(
            num_experiments=num_initial_experiments
        )
        prev_res = exp.run_experiments(suggestions)
        prev_res["task", "METADATA"] = task
    else:
        prev_res = None
    strategy = NewMTBO(
        exp.domain,
        pretraining_data=ct_data,
        task=task,
        brute_force_categorical=brute_force_categorical,
        acquisition_function=acquisition_function,
        categorical_method=categorical_method,
    )
    r = Runner(
        strategy=strategy,
        experiment=exp,
        max_iterations=max_iterations,
        batch_size=batch_size,
    )
    r.run(prev_res=prev_res)
    return r


if __name__ == "__main__":
    app()
