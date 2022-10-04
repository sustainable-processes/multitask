from multitask.benchmarks.suzuki_emulator import SuzukiEmulator
from multitask.etl.suzuki_data_utils import get_suzuki_dataset
from multitask.etl.cn_data_utils import get_cn_dataset
from multitask.strategies import NewSTBO, NewMTBO
from multitask.utils import WandbRunner, BenchmarkType
from botorch.models import SingleTaskGP, MultiTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from summit import *
import gpytorch
import torch
import numpy as np
from scipy.stats import spearmanr
from scipy.sparse import issparse
import wandb
from tqdm.auto import trange
from pathlib import Path
from typing import Dict, List, Literal, Optional
import pandas as pd
import logging
import json
import warnings
import os

N_RETRIES = 5

logger = logging.getLogger(__name__)


def stbo(
    model_name: str,
    wandb_benchmark_artifact_name: str,
    benchmark_type: BenchmarkType,
    output_path: str,
    wandb_main_dataset_artifact_name: Optional[str] = None,
    max_experiments: Optional[int] = 20,
    batch_size: Optional[int] = 1,
    brute_force_categorical: Optional[bool] = False,
    print_warnings: Optional[bool] = True,
    acquisition_function: Optional[str] = "EI",
    repeats: Optional[int] = 20,
    wandb_artifact_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = "multitask",
):
    """Optimization of a Suzuki or C-N benchmark with Single-Task Bayesian Optimziation

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
    args["torch_num_threads"] = torch.get_num_threads()

    # Ouptut path
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Download main dataset
    if wandb_main_dataset_artifact_name:
        api = wandb.Api()
        main_dataset_path = api.artifact(
            f"{wandb_entity}/{wandb_project}/{wandb_main_dataset_artifact_name}"
        ).download()
        main_dataset_path = Path(main_dataset_path)
        if benchmark_type == BenchmarkType.suzuki:
            main_ds = get_suzuki_dataset(
                data_path=main_dataset_path / f"{model_name}.pb",
                split_catalyst=exp.split_catalyst,
                print_warnings=print_warnings,
            )
        elif benchmark_type == BenchmarkType.cn:
            main_ds = get_cn_dataset(
                data_path=main_dataset_path / f"{model_name}.pb",
                print_warnings=print_warnings,
            )
    else:
        main_ds = None

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
                    tags = ["STBO", benchmark_type.value]
                    if os.environ.get("lightning_cloud"):
                        tags.append("lightning_cloud")
                    run = wandb.init(
                        entity=wandb_entity,
                        project=wandb_project,
                        config=args,
                        tags=tags,
                    )
                    # Download benchmark weights from wandb and load
                    benchmark_artifact = run.use_artifact(wandb_benchmark_artifact_name)
                    benchmark_path = benchmark_artifact.download()
                    if benchmark_type == BenchmarkType.suzuki:
                        exp = SuzukiEmulator.load(
                            model_name=model_name, save_dir=benchmark_path
                        )
                    elif benchmark_type == BenchmarkType.cn:
                        exp = ExperimentalEmulator.load(
                            model_name=model_name, save_dir=benchmark_path
                        )
                    else:
                        raise ValueError(f"Unknown benchmark type {benchmark_type}")
                    # Run optimization
                    result = run_stbo(
                        exp,
                        max_iterations=max_iterations,
                        batch_size=batch_size,
                        brute_force_categorical=brute_force_categorical,
                        categorical_method=categorical_method,
                        acquisition_function=acquisition_function,
                        main_ds=main_ds,
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
                finally:
                    wandb.finish()
            if retries >= N_RETRIES:
                logger.info(
                    f"Not able to find semi-positive definite matrix at {retries} tries. Skipping repeat {i}"
                )
                done = True


def mtbo(
    model_name: str,
    wandb_benchmark_artifact_name: str,
    benchmark_type: BenchmarkType,
    wandb_ct_dataset_artifact_name: str,
    ct_dataset_names: List[str],
    output_path: str,
    wandb_main_dataset_artifact_name: Optional[str] = None,
    max_experiments: Optional[int] = 20,
    batch_size: Optional[int] = 1,
    repeats: Optional[int] = 20,
    print_warnings: Optional[bool] = True,
    brute_force_categorical: bool = False,
    acquisition_function: Optional[str] = "EI",
    wandb_entity: Optional[str] = "ceb-sre",
    wandb_project: Optional[str] = "multitask",
    wandb_artifact_name: Optional[str] = None,
):
    """Optimization of a Suzuki or C-N benchmark with Multitask Bayesian Optimziation"""
    args = dict(locals())
    args["strategy"] = "MTBO"
    args["torch_num_threads"] = torch.get_num_threads()

    # Saving
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Multi-Task Bayesian Optimization
    max_iterations = max_experiments // batch_size
    max_iterations += 1 if max_experiments % batch_size != 0 else 0

    # Download main dataset
    if wandb_main_dataset_artifact_name:
        api = wandb.Api()
        main_dataset_path = api.artifact(
            f"{wandb_entity}/{wandb_project}/{wandb_main_dataset_artifact_name}"
        ).download()
        main_dataset_path = Path(main_dataset_path)
        if benchmark_type == BenchmarkType.suzuki:
            main_ds = get_suzuki_dataset(
                data_path=main_dataset_path / f"{model_name}.pb",
                split_catalyst=True,
                print_warnings=print_warnings,
            )
        elif benchmark_type == BenchmarkType.cn:
            main_ds = get_cn_dataset(
                data_path=main_dataset_path / f"{model_name}.pb",
                print_warnings=print_warnings,
            )
    else:
        main_ds = None

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
                    tags = ["MTBO", benchmark_type.value]
                    if os.environ.get("lightning_cloud"):
                        tags.append("lightning_cloud")
                    # Initialize wandb
                    run = wandb.init(
                        entity=wandb_entity,
                        project=wandb_project,
                        config=args,
                        tags=tags,
                    )

                    # Download benchmark weights from wandb and load
                    benchmark_artifact = run.use_artifact(wandb_benchmark_artifact_name)
                    benchmark_path = benchmark_artifact.download()
                    if benchmark_type == BenchmarkType.suzuki:
                        exp = SuzukiEmulator.load(
                            model_name=model_name, save_dir=benchmark_path
                        )
                    elif benchmark_type == BenchmarkType.cn:
                        exp = ExperimentalEmulator.load(
                            model_name=model_name, save_dir=benchmark_path
                        )
                    else:
                        raise ValueError(f"Unknown benchmark type {benchmark_type}")

                    # Load suzuki dataset
                    dataset_artifact = run.use_artifact(wandb_ct_dataset_artifact_name)
                    datasets_path = Path(dataset_artifact.download())
                    if benchmark_type == BenchmarkType.suzuki:
                        ds_list = [
                            get_suzuki_dataset(
                                data_path=datasets_path / f"{ct_dataset_name}.pb",
                                split_catalyst=exp.split_catalyst,
                                print_warnings=print_warnings,
                            )
                            for ct_dataset_name in ct_dataset_names
                        ]
                    elif benchmark_type == BenchmarkType.cn:
                        ds_list = [
                            get_cn_dataset(
                                data_path=datasets_path / f"{ct_dataset_name}.pb",
                                print_warnings=print_warnings,
                            )
                            for ct_dataset_name in ct_dataset_names
                        ]
                    for i, ds in enumerate(ds_list):
                        ds["task", "METADATA"] = i
                    big_ds = pd.concat(ds_list)
                    opt_task = len(ds_list)

                    # Run optimization
                    result = run_mtbo(
                        exp,
                        ct_data=big_ds,
                        max_iterations=max_iterations,
                        batch_size=batch_size,
                        task=opt_task,
                        brute_force_categorical=brute_force_categorical,
                        categorical_method=categorical_method,
                        acquisition_function=acquisition_function,
                        main_ds=main_ds,
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
                finally:
                    wandb.finish()
            if retries >= N_RETRIES:
                logger.info(
                    f"Not able to find semi-positive definite matrix at {retries} tries. Skipping repeat {i}"
                )
                done = True


class STBOCallback:
    def __init__(
        self,
        max_iterations: int,
        main_ds: Optional[DataSet] = None,
    ):
        self.max_iterations = max_iterations
        self.main_ds = main_ds

    def get_kernel_lengthscales(self, model: SingleTaskGP, domain: Domain):
        wandb_dict = {}
        # Kernel lengthscales
        k = 0

        lengthscale = model.covar_module.base_kernel.lengthscale.detach().numpy()[0]
        for v in domain.input_variables:
            if isinstance(v, ContinuousVariable):
                wandb_dict.update(
                    {f"kernel/kernel_lengthscale_{v.name}": lengthscale[k]}
                )
                k += 1
            elif isinstance(v, CategoricalVariable):
                for level in v.levels:
                    wandb_dict.update(
                        {
                            f"""kernel/kernel_lengthscale_{v.name}_{level.replace(' ', '_')}""": lengthscale[
                                k
                            ]
                        }
                    )
                    k += 1
        return wandb_dict

    def get_marginal_likelihood(
        self, model: SingleTaskGP, mll: ExactMarginalLogLikelihood, inputs, output
    ):
        with torch.no_grad():
            model_output = model(torch.tensor(inputs).double())
            train_y = torch.tensor(output).double()
            log_likelihoods = mll(model_output, train_y).numpy()
            sum_likelihood = np.exp(np.sum(log_likelihoods) / len(log_likelihoods))
        return {"marginal_likelihood": sum_likelihood}

    def get_spearmans_coefficient(
        self, model, inputs, output, include_table: bool = False
    ):
        wandb_dict = {}
        with torch.no_grad():
            model_output = model(torch.tensor(inputs).double())
        train_y = torch.tensor(output).double()
        abs_residuals = (model_output.mean - train_y[:, 0]).abs().numpy()
        uncertainties = model_output.variance.sqrt().numpy()
        if include_table:
            wandb_dict.update(
                {
                    "uncertainty_residuals": wandb.Table(
                        columns=["standard_deviation", "absolute_residual"],
                        data=np.vstack([uncertainties, abs_residuals]).T.tolist(),
                    )
                }
            )
        coeff = spearmanr(a=uncertainties, b=abs_residuals)
        wandb_dict.update(
            {"spearmans_rank_coefficient_uncertainty_residual": coeff.correlation}
        )
        return wandb_dict

    def __call__(self, cls: WandbRunner, prev_res, iteration):
        strategy: NewSTBO = cls.strategy
        domain: Domain = strategy.domain
        wandb_dict = {"iteration": iteration}

        try:
            model: SingleTaskGP = strategy.model
            mll: ExactMarginalLogLikelihood = strategy.mll
        except AttributeError:
            model = None
            mll = None

        if model is not None:
            wandb_dict.update(self.get_kernel_lengthscales(model, domain))

        if model is not None and mll is not None and self.main_ds is not None:
            inputs, output = transform(
                self.main_ds,
                domain,
                output_means=strategy.transform.output_means,
                output_stds=strategy.transform.output_stds,
                encoders=strategy.transform.encoders,
            )
            inputs = inputs.data_to_numpy().astype(float)
            output = output.data_to_numpy().astype(float)
            wandb_dict.update(self.get_marginal_likelihood(model, mll, inputs, output))

            # Calculate spearmans' rank coefficient on errors
            wandb_dict.update(
                self.get_spearmans_coefficient(
                    model,
                    inputs,
                    output,
                    include_table=True
                    if iteration == self.max_iterations - 1
                    else False,
                )
            )

        wandb.log(wandb_dict)


def run_stbo(
    exp: Experiment,
    max_iterations: int = 10,
    batch_size: int = 1,
    brute_force_categorical: bool = False,
    categorical_method: str = "one-hot",
    acquisition_function: str = "EI",
    wandb_runner_kwargs: Optional[dict] = {},
    main_ds: Optional[DataSet] = None,
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
    stbo_callback = STBOCallback(max_iterations=max_iterations, main_ds=main_ds)
    r = WandbRunner(
        strategy=strategy,
        experiment=exp,
        max_iterations=max_iterations,
        batch_size=batch_size,
        **wandb_runner_kwargs,
    )
    r.run(
        skip_wandb_intialization=True,
        callback=stbo_callback,
    )
    return r


class MTBOCallback(STBOCallback):
    def __init__(
        self,
        max_iterations: int,
        opt_task: int,
        main_ds: Optional[DataSet] = None,
    ):
        self.max_iterations = max_iterations
        self.opt_task = opt_task
        self.main_ds = main_ds

    def get_kernel_lengthscales(self, model: SingleTaskGP, domain: Domain):
        wandb_dict = super().get_kernel_lengthscales(model, domain)
        with torch.no_grad():
            task_covar_matrix = model.task_covar_module._eval_covar_matrix().numpy()
        for i in range(task_covar_matrix.shape[0]):
            for j in range(task_covar_matrix.shape[1]):
                wandb_dict.update(
                    {f"kernel/task_covar_{i}_{j}": task_covar_matrix[i, j]}
                )
        return wandb_dict

    def __call__(self, cls: WandbRunner, prev_res, iteration):
        strategy: NewMTBO = cls.strategy
        domain: Domain = strategy.domain
        wandb_dict = {"iteration": iteration}

        try:
            model: MultiTaskGP = strategy.model
            mll: ExactMarginalLogLikelihood = strategy.mll
        except AttributeError:
            model = None
            mll = None

        if model is not None:
            wandb_dict.update(self.get_kernel_lengthscales(model, domain))

        if model is not None and mll is not None and self.main_ds is not None:
            inputs, output = transform(
                self.main_ds,
                domain,
                output_means=strategy.transform.output_means,
                output_stds=strategy.transform.output_stds,
                encoders=strategy.transform.encoders,
            )
            inputs = inputs.data_to_numpy().astype(float)
            inputs_task = np.append(
                inputs,
                self.opt_task * np.ones((inputs.shape[0], 1)),
                axis=1,
            ).astype(np.float)
            output = output.data_to_numpy().astype(float)

            # Marginal likelihood
            wandb_dict.update(
                self.get_marginal_likelihood(model, mll, inputs_task, output)
            )

            # Calculate spearmans' rank coefficient on errors
            wandb_dict.update(
                self.get_spearmans_coefficient(
                    model,
                    inputs_task,
                    output,
                    include_table=True
                    if iteration == self.max_iterations - 1
                    else False,
                )
            )

        wandb.log(wandb_dict)


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
    main_ds: Optional[DataSet] = None,
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
    mtbo_callback = MTBOCallback(
        max_iterations=max_iterations, opt_task=task, main_ds=main_ds
    )
    r.run(
        skip_wandb_intialization=True,
        callback=mtbo_callback,
    )
    return r


def transform(
    ds: DataSet,
    domain: Domain,
    output_means: Dict,
    output_stds: Dict,
    encoders: Optional[Dict] = None,
):
    """Transform a dataset using existing parameters"""
    ds = ds.copy()
    input_columns = []
    for v in domain.input_variables:
        if isinstance(v, ContinuousVariable):
            bounds = domain[v.name].bounds
            ds[v.name, "DATA"] = (ds[v.name] - bounds[0]) / (bounds[1] - bounds[0])
            input_columns.append(v.name)
        elif isinstance(v, CategoricalVariable):
            ohe = encoders[v.name]
            values = np.atleast_2d(ds[v.name].to_numpy()).T
            one_hot_values = ohe.transform(values)
            if issparse(one_hot_values):
                one_hot_values = one_hot_values.todense()
            for loc, l in enumerate(v.levels):
                column_name = f"{v.name}_{l}"
                ds[column_name, "DATA"] = one_hot_values[:, loc]
                input_columns.append(column_name)
            ds = ds.drop(columns=[v.name], axis=1)

    output_columns = []
    for v in domain.output_variables:
        ds[v.name] = (ds[v.name] - output_means[v.name]) / output_stds[v.name]
        output_columns.append(v.name)
    return ds[input_columns], ds[output_columns]
