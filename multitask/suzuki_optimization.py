from multitask.suzuki_emulator import SuzukiEmulator
from multitask.suzuki_data_utils import get_suzuki_dataset
from multitask.mt import NewSTBO, NewMTBO
from multitask.utils import WandbRunner
import lightning as L
from summit import *
import gpytorch
import torch
import wandb
from numpy.random import default_rng
from tqdm.auto import tqdm, trange
from pathlib import Path
from typing import Iterable, Tuple, Dict, Union, List, Optional
import pandas as pd
import logging
import json
import warnings
import os

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

WANDB_SETTINGS = {"wandb_entity": "ceb-sre", "wandb_project": "multitask"}


class SuzukiWork(L.LightningWork):
    def __init__(
        self,
        strategy: str,
        model_name: str,
        benchmark_path: str,
        output_path: str,
        ct_data_paths: Optional[List[str]] = None,
        max_experiments: Optional[int] = 20,
        batch_size: Optional[int] = 1,
        brute_force_categorical: Optional[bool] = False,
        acquisition_function: Optional[str] = "EI",
        repeats: Optional[int] = 20,
        wandb_artifact_name: Optional[str] = None,
        print_warnings: Optional[bool] = True,
        parallel: bool = True,
    ):
        super().__init__(parallel=parallel)
        self.strategy = strategy
        self.model_name = model_name
        self.benchmark_path = benchmark_path
        self.output_path = output_path
        self.max_experiments = max_experiments
        self.batch_size = batch_size
        self.brute_force_categorical = brute_force_categorical
        self.acquisition_function = acquisition_function
        self.repeats = repeats
        self.wandb_artificant_name = wandb_artifact_name
        self.print_warnings = print_warnings
        self.ct_data_paths = ct_data_paths

        # login to wandb
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        self.wandb_run_id = None

    def run(self):
        # Ouptut path
        output_path = Path(self.output_path)
        output_path.mkdir(exist_ok=True)

        # Load benchmark
        exp = SuzukiEmulator.load(
            model_name=self.model_name, save_dir=self.benchmark_path
        )

        # Load auxliary datasets
        if self.strategy == "MTBO":
            ds_list = [
                get_suzuki_dataset(
                    ct_data_path,
                    split_catalyst=exp.split_catalyst,
                    print_warnings=self.print_warnings,
                )
                for ct_data_path in self.ct_data_paths
            ]
            for i, ds in enumerate(ds_list):
                ds["task", "METADATA"] = i
            big_ds = pd.concat(ds_list)
            opt_task = len(ds_list)

        # Saving
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)

        # Bayesian Optimization
        max_iterations = self.max_experiments // self.batch_size
        max_iterations += 1 if self.max_experiments % self.batch_size != 0 else 0
        if self.brute_force_categorical:
            categorical_method = None
        else:
            categorical_method = "one-hot"

        for i in trange(self.repeats):
            done = False
            retries = 0
            while not done:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        run = wandb.init(
                            entity=os.environ["WANDB_ENTITY"],
                            project=os.environ["WANDB_PROJECT"],
                            # config=args,
                            tags=[self.strategy],
                        )
                        self.wandb_run_id = run.id
                        self.wandb_run_name = run.name
                        self.wandb_project = run.project
                        self.wandb_entity = run.entity
                        if self.strategy == "STBO":
                            result = run_stbo(
                                exp,
                                max_iterations=max_iterations,
                                batch_size=self.batch_size,
                                brute_force_categorical=self.brute_force_categorical,
                                categorical_method=categorical_method,
                                acquisition_function=self.acquisition_function,
                                wandb_runner_kwargs=WANDB_SETTINGS,
                            )
                        elif self.strategy == "MTBO":
                            result = run_mtbo(
                                exp,
                                ct_data=big_ds,
                                max_iterations=max_iterations,
                                batch_size=self.batch_size,
                                task=opt_task,
                                brute_force_categorical=self.brute_force_categorical,
                                categorical_method=categorical_method,
                                acquisition_function=self.cquisition_function,
                            )
                        result.save(output_path / f"repeat_{i}.json")
                        torch.save(
                            result.strategy.model.state_dict(),
                            output_path / f"repeat_{i}_model.pth",
                        )
                        if self.wandb_artifact_name:
                            artifact = wandb.Artifact(
                                self.wandb_artifact_name, type="optimization_result"
                            )
                            artifact.add_file(output_path / f"repeat_{i}.json")
                            artifact.add_file(output_path / f"repeat_{i}_model.pth")
                            run.log_artifact(artifact)
                        done = True
                    except gpytorch.utils.errors.NotPSDError:
                        retries += 1
                if retries >= N_RETRIES:
                    logger.info(
                        f"Not able to find semi-positive definite matrix at {retries} tries. Skipping repeat {i}"
                    )
                    done = True


def stbo(
    model_name: str,
    benchmark_path: str,
    output_path: str,
    max_experiments: Optional[int] = 20,
    batch_size: Optional[int] = 1,
    brute_force_categorical: Optional[bool] = False,
    acquisition_function: Optional[str] = "EI",
    repeats: Optional[int] = 20,
    wandb_artifact_name: Optional[str] = None,
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
    args = dict(locals())
    args["strategy"] = "STBO"

    # Ouptut path
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    # Load benchmark
    exp = SuzukiEmulator.load(model_name=model_name, save_dir=benchmark_path)

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

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
                    run = wandb.init(
                        entity=WANDB_SETTINGS["wandb_entity"],
                        project=WANDB_SETTINGS["wandb_project"],
                        config=args,
                        tags=["STBO"],
                    )
                    result = run_stbo(
                        exp,
                        max_iterations=max_iterations,
                        batch_size=batch_size,
                        brute_force_categorical=brute_force_categorical,
                        categorical_method=categorical_method,
                        acquisition_function=acquisition_function,
                        wandb_runner_kwargs=WANDB_SETTINGS,
                    )
                    result.save(output_path / f"repeat_{i}.json")
                    torch.save(
                        result.strategy.model.state_dict(),
                        output_path / f"repeat_{i}_model.pth",
                    )
                    if wandb_artifact_name:
                        artifact = wandb.Artifact(
                            wandb_artifact_name, type="optimization_result"
                        )
                        artifact.add_file(output_path / f"repeat_{i}.json")
                        artifact.add_file(output_path / f"repeat_{i}_model.pth")
                        run.log_artifact(artifact)
                    done = True
                except gpytorch.utils.errors.NotPSDError:
                    retries += 1
            if retries >= N_RETRIES:
                logger.info(
                    f"Not able to find semi-positive definite matrix at {retries} tries. Skipping repeat {i}"
                )
                done = True


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
    acquisition_function: Optional[str] = "EI",
    wandb_artifact_name: Optional[str] = None,
):
    """Optimization of a Suzuki benchmark with Multitask Bayesian Optimziation"""
    args = dict(locals())
    args["strategy"] = "MTBO"
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

    # Saving
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    with open(output_path / "args.json", "w") as f:
        json.dump(args, f)

    # Multi-Task Bayesian Optimization
    max_iterations = max_experiments // batch_size
    max_iterations += 1 if max_experiments % batch_size != 0 else 0
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
                    run = wandb.init(
                        entity=WANDB_SETTINGS["wandb_entity"],
                        project=WANDB_SETTINGS["wandb_project"],
                        config=args,
                        tags=["MTBO"],
                    )
                    result = run_mtbo(
                        exp,
                        ct_data=big_ds,
                        max_iterations=max_iterations,
                        batch_size=batch_size,
                        task=opt_task,
                        brute_force_categorical=brute_force_categorical,
                        categorical_method=categorical_method,
                        acquisition_function=acquisition_function,
                    )
                    result.save(output_path / f"repeat_{i}.json")
                    torch.save(
                        result.strategy.model.state_dict(),
                        output_path / f"repeat_{i}_model.pth",
                    )
                    if wandb_artifact_name:
                        artifact = wandb.Artifact(
                            wandb_artifact_name, type="optimization_result"
                        )
                        artifact.add_file(output_path / f"repeat_{i}.json")
                        artifact.add_file(output_path / f"repeat_{i}_model.pth")
                        run.log_artifact(artifact)
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
    wandb_runner_kwargs: Optional[dict] = {},
):
    """Run Single Task Bayesian Optimization (AKA normal BO)"""
    exp.reset()
    assert exp.data.shape[0] == 0
    strategy = NewSTBO(
        exp.domain,
        brute_force_categorical=brute_force_categorical,
        categorical_method=categorical_method,
        acquisition_function=acquisition_function,
    )
    r = WandbRunner(
        strategy=strategy,
        experiment=exp,
        max_iterations=max_iterations,
        batch_size=batch_size,
        **wandb_runner_kwargs,
    )
    r.run(skip_wandb_intialization=True)
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
    wandb_runner_kwargs: Optional[dict] = {},
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
    r = WandbRunner(
        strategy=strategy,
        experiment=exp,
        max_iterations=max_iterations,
        batch_size=batch_size,
        **wandb_runner_kwargs,
    )
    r.run(skip_wandb_intialization=True)
    return r


if __name__ == "__main__":
    app()
