import subprocess
import time
from typing import Dict, List, Optional
import wandb
from multitask.benchmarks.suzuki_benchmark_training import (
    train_benchmark as train_suzuki_benchmark,
)
from multitask.benchmarks.cn_benchmark_training import (
    train_benchmark as train_cn_benchmark,
)
from multitask.utils import BenchmarkType
import lightning as L
from lightning_app.structures.dict import Dict
from pathlib import Path
import subprocess
from typing import List, Optional, Literal
import logging
import wandb

logger = logging.getLogger(__name__)


WANDB_SETTINGS = {"wandb_entity": "ceb-sre", "wandb_project": "multitask"}


class BenchmarkWork(L.LightningWork):
    """Train a benchmark"""

    def __init__(
        self,
        benchmark_type: BenchmarkType,
        wandb_dataset_artifact_name: str,
        dataset_name: str,
        save_path: str,
        figure_path: str,
        max_epochs: Optional[int] = 1000,
        cv_folds: Optional[int] = 5,
        verbose: Optional[int] = 0,
        parallel: Optional[bool] = False,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = "multitask",
        **kwargs,
    ):
        super().__init__(parallel=parallel, **kwargs)
        self.benchmark_type = benchmark_type
        self.wandb_dataset_artifact_name = wandb_dataset_artifact_name
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.figure_path = figure_path
        self.verbose = verbose
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.max_epochs = max_epochs
        self.cv_folds = cv_folds
        self.finished = False

    def run(self, **kwargs):
        cmd = [
            "multitask",
            "benchmarks",
            "train-cn" if self.benchmark_type == BenchmarkType.cn else "train-suzuki",
            self.dataset_name,
            self.save_path,
            self.figure_path,
            "--wandb-dataset-artifact-name",
            self.wandb_dataset_artifact_name,
            "--wandb-project",
            self.wandb_project,
            "--max-epochs",
            str(self.max_epochs),
            "--cv-folds",
            str(self.cv_folds),
        ]
        if self.wandb_entity:
            cmd += ["--wandb-entity", self.wandb_entity]
        if self.verbose:
            cmd += ["--verbose 1"]
        print(cmd)
        subprocess.run(cmd, shell=False)


class OptimizationWork(L.LightningWork):
    """Optimization of the Suzuki or C-N benchmark"""

    def __init__(
        self,
        strategy: str,
        model_name: str,
        wandb_benchmark_artifact_name: str,
        benchmark_type: BenchmarkType,
        output_path: str,
        wandb_dataset_artifact_name: Optional[str] = None,
        ct_dataset_names: Optional[List[str]] = None,
        max_experiments: Optional[int] = 20,
        batch_size: Optional[int] = 1,
        brute_force_categorical: Optional[bool] = False,
        acquisition_function: Optional[str] = "EI",
        repeats: Optional[int] = 20,
        wandb_artifact_name: Optional[str] = None,
        print_warnings: Optional[bool] = True,
        parallel: bool = True,
        cloud_compute=None,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = "multitask",
        **kwargs,
    ):
        super().__init__(
            parallel=parallel,
            cloud_compute=cloud_compute,
            cloud_build_config=L.BuildConfig(
                [
                    "pandas>=1.2.4",
                    "matplotlib>=3.4.1",
                    "numpy>=1.20.2",
                    "scikit-learn>=1.0.2",
                    "summit>=0.8.6",
                    "pint>=0.17",
                    "tqdm>=4.61.0",
                    "rdkit-pypi>=2021.9.4",
                    "wandb>=0.13.12",
                    "typer>=0.6.1",
                    ".",
                ]
            ),
            **kwargs,
        )
        self.strategy = strategy
        self.model_name = model_name
        self.wandb_benchmark_artifact_name = wandb_benchmark_artifact_name
        self.wandb_dataset_artifact_name = wandb_dataset_artifact_name
        self.benchmark_type = benchmark_type.value
        self.ct_dataset_names = ct_dataset_names
        self.output_path = output_path
        self.max_experiments = max_experiments
        self.batch_size = batch_size
        self.brute_force_categorical = brute_force_categorical
        self.acquisition_function = acquisition_function
        self.repeats = repeats
        self.wandb_artifact_name = wandb_artifact_name
        self.print_warnings = print_warnings
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.finished = False

    def run(self):
        cmd = [
            "multitask",
            "optimization",
            self.strategy.lower(),
            self.model_name,
            self.wandb_benchmark_artifact_name,
            self.benchmark_type,
        ]
        if self.strategy.lower() == "mtbo":
            cmd += [self.wandb_dataset_artifact_name]
            cmd += self.ct_dataset_names
        cmd += [self.output_path]
        options = [
            "--max-experiments",
            str(self.max_experiments),
            "--batch-size",
            str(self.batch_size),
            "--repeats",
            str(self.repeats),
            "--wandb-artifact-name",
            str(self.wandb_artifact_name),
            "--acquisition-function",
            str(self.acquisition_function),
        ]
        if not self.print_warnings:
            options += ["--no-print-warning"]
        if self.brute_force_categorical:
            options += ["--brute-force-categorical"]
        if self.wandb_entity:
            options += ["--wandb-entity", self.wandb_entity]
        if self.wandb_project:
            options += ["--wandb-project", self.wandb_project]
        # print(cmd + options)
        subprocess.run(cmd + options, shell=False)
        self.finished = True


class MultitaskBenchmarkStudy(L.LightningFlow):
    """
    Benchmarking study of single task vs multitask optimization
    """

    def __init__(
        self,
        run_benchmark_training: bool,
        run_single_task: bool,
        run_multi_task: bool,
        compute_type: str = "cpu-medium",
        parallel: bool = True,
        max_workers: int = 10,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = "multitask",
    ):
        super().__init__()

        self.max_experiments = 20
        self.batch_size = 1
        self.repeats = 20
        self.run_benchmark_training = run_benchmark_training
        self.run_single_task = run_single_task
        self.run_multi_task = run_multi_task
        self.compute_type = compute_type
        self.parallel = parallel
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        # Workers
        self.max_workers = max_workers
        self.workers = Dict()
        self.total_jobs = 0
        self.current_workers: List[int] = []
        self.succeded: List[int] = []
        self.all_initialized = False

    def run(self):
        # Benchmark training
        if self.run_benchmark_training and self.all_initialized:
            # Train Baumgartner CN benchmark
            baumgartner_cn_runs = [
                BenchmarkWork(
                    benchmark_type="cn",
                    dataset_name=f"baumgartner_cn_case_{case}",
                    wandb_dataset_artifact_name="baumgartner_cn:latest",
                    save_path=f"data/baumgartner_cn/emulator_case_{case}/",
                    figure_path="figures/",
                    parallel=self.parallel,
                    cloud_compute=L.CloudCompute(name=self.compute_type),
                    wandb_entity=self.wandb_entity,
                    wandb_project=self.wandb_project,
                )
                for case in range(1, 5)
            ]
            for r in baumgartner_cn_runs:
                r.run(
                    max_epochs=1000,
                    cv_folds=5,
                    verbose=1,
                )

            # Train Baumgartner Suzuki benchmark
            baumgartner_suzuki_runs = [
                BenchmarkWork(
                    benchmark_type="suzuki",
                    dataset_name=f"baumgartner_suzuki",
                    wandb_dataset_artifact_name="baumgartner_suzuki:latest",
                    save_path="data/baumgartner_suzuki/emulator",
                    figure_path="figures/",
                    parallel=self.parallel,
                    cloud_compute=L.CloudCompute(self.compute_type),
                    wandb_entity=self.wandb_entity,
                    wandb_project=self.wandb_project,
                )
            ]

            # Train Reizman Suzuki benchmarks
            reizman_suzuki_runs = [
                BenchmarkWork(
                    benchmark_type="suzuki",
                    dataset_name=f"reizman_suzuki_case_{case}",
                    wandb_dataset_artifact_name="reizman_suzuki:latest",
                    save_path=f"data/reizman_suzuki/emulator_case_{case}/",
                    figure_path="figures/",
                    parallel=self.parallel,
                    cloud_compute=L.CloudCompute(name=self.compute_type),
                    wandb_entity=self.wandb_entity,
                    wandb_project=self.wandb_project,
                )
                for case in range(1, 5)
            ]
            for r in reizman_suzuki_runs + baumgartner_suzuki_runs:
                r.run(
                    split_catalyst=False,
                    max_epochs=1000,
                    cv_folds=5,
                    verbose=1,
                    print_warnings=False,
                )

        # Multi task benchmarking
        if self.run_multi_task and not self.all_initialized:
            configs = []
            # configs += self.generate_suzuki_configs_multitask(
            #     max_experiments=self.max_experiments,
            #     batch_size=self.batch_size,
            #     repeats=self.repeats,
            #     parallel=self.parallel,
            # )
            configs += self.generate_cn_configs_multitask(
                max_experiments=self.max_experiments,
                batch_size=self.batch_size,
                repeats=self.repeats,
                parallel=self.parallel,
            )
            for i, config in enumerate(configs):
                self.workers[str(self.total_jobs + i)] = OptimizationWork(
                    **config,
                    cloud_compute=L.CloudCompute(self.compute_type, idle_timeout=60),
                    wandb_entity=self.wandb_entity,
                    wandb_project=self.wandb_project,
                )
            self.total_jobs += len(configs)

        # Single task benchmarking
        if self.run_single_task and not self.all_initialized:
            configs = []
            # configs += self.generate_suzuki_configs_single_task(
            #     max_experiments=self.max_experiments,
            #     batch_size=self.batch_size,
            #     repeats=self.repeats,
            #     parallel=self.parallel,
            # )
            configs += self.generate_cn_configs_single_task(
                max_experiments=self.max_experiments,
                batch_size=self.batch_size,
                repeats=self.repeats,
                parallel=self.parallel,
            )
            for i, config in enumerate(configs):
                self.workers[str(self.total_jobs + i)] = OptimizationWork(
                    **config,
                    cloud_compute=L.CloudCompute(self.compute_type, idle_timeout=60),
                    wandb_entity=self.wandb_entity,
                )
            self.total_jobs += len(configs)

        self.all_initialized = True

        if self.run_single_task or self.run_multi_task:
            # Check for finished jobs
            for i in self.current_workers:
                if self.workers[str(i)].finished:
                    self.current_workers.remove(i)
                    self.succeded.append(i)
                    self.workers[str(i)].stop()

            # Queue new jobs
            i = 0
            while len(self.current_workers) < self.max_workers and i < self.total_jobs:
                if i not in self.succeded and i not in self.current_workers:
                    self.workers[str(i)].run()
                    self.current_workers.append(i)
                    print(f"Job {i+1} of {self.total_jobs} queued")
                i += 1

    @staticmethod
    def generate_suzuki_configs_single_task(
        max_experiments: int, batch_size: int, repeats: int, parallel: bool
    ):
        # STBO Reizman
        reizman_stbo_configs = [
            {
                "strategy": "STBO",
                "benchmark_type": BenchmarkType.suzuki,
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case}:latest",
                "model_name": f"reizman_suzuki_case_{case}",
                "output_path": f"data/reizman_suzuki/results_stbo_case_{case}/",
                "wandb_artifact_name": "stbo_reizman_suzuki",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
            }
            for case in range(1, 5)
        ]

        # STBO Baumgartner
        baumgartner_stbo_configs = [
            {
                "strategy": "STBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"baumgartner_suzuki",
                "wandb_benchmark_artifact_name": "benchmark_baumgartner_suzuki:latest",
                "output_path": f"data/baumgarnter_suzuki/results_stbo/",
                "wandb_artifact_name": "stbo_baumgartner_suzuki",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
            }
        ]

        return reizman_stbo_configs + baumgartner_stbo_configs

    @staticmethod
    def generate_suzuki_configs_multitask(
        max_experiments: int, batch_size: int, repeats: int, parallel: bool
    ):
        # MTBO Reizman one cotraining with Baumgartner
        reizman_mtbo_configs_baugmartner_one = [
            {
                "strategy": "MTBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"reizman_suzuki_case_{case}",
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case}:latest",
                "output_path": f"data/multitask_results/results_reizman_suzuki_{case}_cotrain_baumgartner_suzuki",
                "wandb_dataset_artifact_name": "baumgartner_suzuki:latest",
                "ct_dataset_names": [f"baumgartner_suzuki"],
                "wandb_artifact_name": "mtbo_reizman_suzuki_one_cotraining_baumgartner_suzuki",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
            }
            for case in range(1, 5)
        ]

        # MTBO Reizman one cotraining with reizman
        reizman_mtbo_configs_reizman_one = [
            {
                "strategy": "MTBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"reizman_suzuki_case_{case_main}",
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case_main}:latest",
                "output_path": f"data/multitask_results/results_reizman_suzuki_case_{case_main}_cotrain_reizman_suzuki_case_{case_aux}",
                "wandb_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [f"reizman_suzuki_case_{case_aux}"],
                "wandb_artifact_name": f"mtbo_reizman_suzuki_{case_main}_one_cotraining_reizman_suzuki_case_{case_aux}",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
            }
            for case_main in range(1, 5)
            for case_aux in range(1, 5)
        ]

        # MTBO Baumgartner one cotraining with Reizman
        baumgartner_mtbo_configs_reizman_one = [
            {
                "strategy": "MTBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"baumgartner_suzuki",
                "wandb_benchmark_artifact_name": "benchmark_baumgartner_suzuki:latest",
                "output_path": f"data/multitask_results/results_baumgartner_suzuki_cotrain_reizman_suzuki_case_{case}",
                "wandb_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [
                    f"reizman_suzuki_case_{case}",
                ],
                "wandb_artifact_name": f"mtbo_baumgartner_suzuki_one_cotraining_reizman_suzuki_case_{case}",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
            }
            for case in range(1, 5)
        ]
        return (
            reizman_mtbo_configs_baugmartner_one
            + reizman_mtbo_configs_reizman_one
            + baumgartner_mtbo_configs_reizman_one
        )

    @staticmethod
    def generate_cn_configs_single_task(
        max_experiments: int, batch_size: int, repeats: int, parallel: bool
    ):
        baumgartner_stbo_configs = [
            {
                "strategy": "STBO",
                "benchmark_type": BenchmarkType.cn,
                "wandb_benchmark_artifact_name": f"benchmark_baumgartner_cn_case_{case}:latest",
                "model_name": f"baumgartner_cn_case_{case}",
                "output_path": f"data/baumgartner_cn/results_stbo_case_{case}/",
                "wandb_artifact_name": "stbo_baumgartner_cn",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
            }
            for case in range(1, 5)
        ]

        return baumgartner_stbo_configs

    @staticmethod
    def generate_cn_configs_multitask(
        max_experiments: int, batch_size: int, repeats: int, parallel: bool
    ):
        # MTBO Reizman one cotraining with Baumgartner
        baumgartner_mtbo_configs_reizman_one = [
            {
                "strategy": "MTBO",
                "benchmark_type": BenchmarkType.cn,
                "model_name": f"baumgartner_cn_case_{case_main}",
                "wandb_benchmark_artifact_name": f"benchmark_baumgartner_cn_case_{case_main}:latest",
                "output_path": f"data/multitask_results/results_baumgartner_cn_case_{case_main}_cotrain_baumgartner_cn_case_{case_aux}",
                "wandb_dataset_artifact_name": f"baumgartner_cn:latest",
                "ct_dataset_names": [f"baumgartner_cn_case_{case_aux}"],
                "wandb_artifact_name": f"mtbo_baumgartner_cn_{case_main}_one_cotraining_baumgartner_cn_case_{case_aux}",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
            }
            for case_main in range(1, 5)
            for case_aux in range(1, 5)
        ]
        return baumgartner_mtbo_configs_reizman_one


if __name__ == "__main__":
    app = L.LightningApp(
        MultitaskBenchmarkStudy(
            run_benchmark_training=False,
            run_single_task=False,
            run_multi_task=True,
            compute_type="cpu-medium",
            parallel=True,
            max_workers=20,
            wandb_entity="ceb-sre",
        )
    )
