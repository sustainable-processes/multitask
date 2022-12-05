from datetime import datetime
import subprocess
import time
from typing import Dict, List, Optional
from multitask.benchmarks.suzuki_benchmark_training import (
    train_benchmark as train_suzuki_benchmark,
)
from multitask.benchmarks.cn_benchmark_training import (
    train_benchmark as train_cn_benchmark,
)
from multitask.utils import BenchmarkType
import lightning as L
from lightning.app.structures import Dict
from lai_jupyter import JupyterLab
from pathlib import Path
import subprocess
from typing import List, Optional, Literal
import logging
from time import sleep
import warnings

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
        split_catalyst: Optional[bool] = False,
        max_epochs: Optional[int] = 1000,
        cv_folds: Optional[int] = 5,
        verbose: Optional[int] = 0,
        parallel: Optional[bool] = False,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = "multitask",
        **kwargs,
    ):
        super().__init__(parallel=parallel, **kwargs)
        self.benchmark_type = benchmark_type.value
        self.wandb_dataset_artifact_name = wandb_dataset_artifact_name
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.figure_path = figure_path
        self.split_catalyst = split_catalyst
        if self.benchmark_type == BenchmarkType.cn.value and self.split_catalyst:
            warnings.warn("Split catalyst not supported for CN benchmark")
        self.verbose = verbose
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.max_epochs = max_epochs
        self.cv_folds = cv_folds
        self.finished = False

    def run(self, **kwargs):
        if self.benchmark_type == BenchmarkType.cn.value:
            subcmd = "train-cn"
        elif self.benchmark_type == BenchmarkType.suzuki.value:
            subcmd = "train-suzuki"
        else:
            raise ValueError(f"Invalid benchmark type: {self.benchmark_type}")
        cmd = [
            "multitask",
            "benchmarks",
            subcmd,
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
        if self.split_catalyst and self.benchmark_type == BenchmarkType.suzuki.value:
            cmd += ["--split-catalyst"]
        elif (
            not self.split_catalyst
            and self.benchmark_type == BenchmarkType.suzuki.value
        ):
            cmd += ["--no-split-catalyst"]
        if self.wandb_entity:
            cmd += ["--wandb-entity", self.wandb_entity]
        if self.verbose:
            cmd += ["--verbose", "1"]
        print(cmd)
        subprocess.run(cmd, shell=False, check=True)
        self.finished = True


class OptimizationWork(L.LightningWork):
    """Optimization of the Suzuki or C-N benchmark"""

    def __init__(
        self,
        strategy: str,
        model_name: str,
        wandb_benchmark_artifact_name: str,
        benchmark_type: BenchmarkType,
        output_path: str,
        wandb_ct_dataset_artifact_name: Optional[str] = None,
        ct_dataset_names: Optional[List[str]] = None,
        max_experiments: Optional[int] = 20,
        batch_size: Optional[int] = 1,
        brute_force_categorical: Optional[bool] = False,
        acquisition_function: Optional[str] = "EI",
        repeats: Optional[int] = 20,
        print_warnings: Optional[bool] = True,
        parallel: bool = True,
        cloud_compute: L.CloudCompute = None,
        wandb_main_dataset_artifact_name: Optional[str] = None,
        wandb_optimization_artifact_name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = "multitask",
        **kwargs,
    ):
        super().__init__(
            parallel=parallel,
            cloud_compute=cloud_compute,
            # Make sure the multitask package is installed
            # cloud_build_config=L.BuildConfig(requirements=["."]),
            **kwargs,
        )
        self.strategy = strategy
        self.model_name = model_name
        self.wandb_benchmark_artifact_name = wandb_benchmark_artifact_name
        self.wandb_ct_dataset_artifact_name = wandb_ct_dataset_artifact_name
        self.benchmark_type = benchmark_type.value
        self.ct_dataset_names = ct_dataset_names
        self.output_path = output_path
        self.max_experiments = max_experiments
        self.batch_size = batch_size
        self.brute_force_categorical = brute_force_categorical
        self.acquisition_function = acquisition_function
        self.repeats = repeats
        self.wandb_optimization_artifact_name = wandb_optimization_artifact_name
        self.wandb_main_dataset_artifact_name = wandb_main_dataset_artifact_name
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
            cmd += [self.wandb_ct_dataset_artifact_name]
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
            str(self.wandb_optimization_artifact_name),
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
        if self.wandb_main_dataset_artifact_name:
            options += [
                "--wandb-main-dataset-artifact-name",
                self.wandb_main_dataset_artifact_name,
            ]
        if (
            self.strategy.lower() == "stbo"
            and self.wandb_ct_dataset_artifact_name
            and self.ct_dataset_names
        ):
            options += [
                "--wandb-ct-dataset-artifact-name",
                self.wandb_ct_dataset_artifact_name,
            ]
            for ct_dataset_name in self.ct_dataset_names:
                options += [f"--ct-dataset-names", ct_dataset_name]
        print(" ".join(cmd + options))
        subprocess.run(cmd + options, shell=False, check=True)
        self.finished = True


class MultitaskBenchmarkStudy(L.LightningFlow):
    """
    Benchmarking study of single task vs multitask optimization
    """

    def __init__(
        self,
        run_benchmark_training: bool,
        run_single_task: bool,
        run_single_task_head_start: bool,
        run_multi_task: bool,
        run_suzuki: bool = True,
        run_cn: bool = True,
        split_catalyst_suzuki: bool = True,
        # compute_type: str = "gpu",
        run_jupyter: bool = False,
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
        self.run_single_task_head_start = run_single_task_head_start
        self.run_multi_task = run_multi_task
        self.run_suzuki = run_suzuki
        self.split_catalyst_suzuki = split_catalyst_suzuki
        self.run_cn = run_cn
        # self.compute_type = compute_type
        self.run_jupyter = run_jupyter
        self.parallel = parallel
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        # Jupyter
        self.jupyter_work = JupyterLab(
            kernel="python",
            cloud_compute=L.CloudCompute("gpu"),
        )

        # Workers
        self.max_workers = max_workers
        self.train_workers = Dict()
        self.opt_workers = Dict()
        self.total_training_jobs = 0
        self.total_opt_jobs = 0
        self.current_workers: List[int] = []
        self.succeded: List[int] = []
        self.all_benchmarks_finished = False
        self.num_optimization_started = 0

        # C-N benhcmark
        if self.run_cn and self.run_benchmark_training:
            for case in range(1, 5):
                self.train_workers[str(self.total_training_jobs)] = BenchmarkWork(
                    benchmark_type=BenchmarkType.cn,
                    dataset_name=f"baumgartner_cn_case_{case}",
                    wandb_dataset_artifact_name="baumgartner_cn:latest",
                    save_path=f"data/baumgartner_cn/emulator_case_{case}/",
                    figure_path="figures/",
                    parallel=self.parallel,
                    cloud_compute=L.CloudCompute(name="gpu"),
                    wandb_entity=self.wandb_entity,
                    wandb_project=self.wandb_project,
                    max_epochs=1000,
                    cv_folds=5,
                    verbose=1,
                )
                self.total_training_jobs += 1

        if self.run_suzuki and self.run_benchmark_training:
            # Train Baumgartner Suzuki benchmark
            self.train_workers[str(self.total_training_jobs)] = BenchmarkWork(
                benchmark_type=BenchmarkType.suzuki,
                dataset_name=f"baumgartner_suzuki",
                wandb_dataset_artifact_name="baumgartner_suzuki:latest",
                save_path="data/baumgartner_suzuki/emulator",
                figure_path="figures/",
                parallel=self.parallel,
                cloud_compute=L.CloudCompute(name="gpu"),
                wandb_entity=self.wandb_entity,
                wandb_project=self.wandb_project,
                split_catalyst=self.split_catalyst_suzuki,
                max_epochs=1000,
                cv_folds=5,
                verbose=1,
            )
            self.total_training_jobs += 1

            # Train Reizman Suzuki benchmarks
            for case in range(1, 5):
                self.train_workers[str(self.total_training_jobs)] = BenchmarkWork(
                    benchmark_type=BenchmarkType.suzuki,
                    dataset_name=f"reizman_suzuki_case_{case}",
                    wandb_dataset_artifact_name="reizman_suzuki:latest",
                    save_path=f"data/reizman_suzuki/emulator_case_{case}/",
                    figure_path="figures/",
                    parallel=self.parallel,
                    cloud_compute=L.CloudCompute(name="gpu"),
                    wandb_entity=self.wandb_entity,
                    wandb_project=self.wandb_project,
                    split_catalyst=self.split_catalyst_suzuki,
                    max_epochs=1000,
                    cv_folds=5,
                    verbose=1,
                )
                self.total_training_jobs += 1

        # Multi task benchmarking
        if self.run_multi_task:
            configs = []
            if self.run_suzuki:
                configs += self.generate_suzuki_configs_multitask(
                    max_experiments=self.max_experiments,
                    batch_size=self.batch_size,
                    repeats=self.repeats,
                    parallel=self.parallel,
                )
            if self.run_cn:
                configs += self.generate_cn_configs_multitask(
                    max_experiments=self.max_experiments,
                    batch_size=self.batch_size,
                    repeats=self.repeats,
                    parallel=self.parallel,
                )
            for i, config in enumerate(configs):
                compute_type = config.pop("compute_type")
                self.opt_workers[str(self.total_opt_jobs + i)] = OptimizationWork(
                    **config,
                    cloud_compute=L.CloudCompute(name=compute_type),
                    wandb_entity=self.wandb_entity,
                    wandb_project=self.wandb_project,
                )
            self.total_opt_jobs += len(configs)

        # Single task benchmarking
        if self.run_single_task:
            configs = []
            if self.run_suzuki:
                configs += self.generate_suzuki_configs_single_task(
                    max_experiments=self.max_experiments,
                    batch_size=self.batch_size,
                    repeats=self.repeats,
                    parallel=self.parallel,
                )
            if self.run_cn:
                configs += self.generate_cn_configs_single_task(
                    max_experiments=self.max_experiments,
                    batch_size=self.batch_size,
                    repeats=self.repeats,
                    parallel=self.parallel,
                )
            for i, config in enumerate(configs):
                compute_type = config.pop("compute_type")
                self.opt_workers[str(self.total_opt_jobs + i)] = OptimizationWork(
                    **config,
                    cloud_compute=L.CloudCompute(name=compute_type),
                    wandb_entity=self.wandb_entity,
                    wandb_project=self.wandb_project,
                )
            self.total_opt_jobs += len(configs)

        # Single task head start benchmarking
        if self.run_single_task_head_start:
            configs = []
            if self.run_suzuki:
                configs += self.generate_suzuki_configs_single_task_headstart(
                    max_experiments=self.max_experiments,
                    batch_size=self.batch_size,
                    repeats=self.repeats,
                    parallel=self.parallel,
                )
            if self.run_cn:
                configs += self.generate_cn_configs_singletask_headstart(
                    max_experiments=self.max_experiments,
                    batch_size=self.batch_size,
                    repeats=self.repeats,
                    parallel=self.parallel,
                )
            for i, config in enumerate(configs):
                compute_type = config.pop("compute_type")
                self.opt_workers[str(self.total_opt_jobs + i)] = OptimizationWork(
                    **config,
                    cloud_compute=L.CloudCompute(name=compute_type),
                    wandb_entity=self.wandb_entity,
                    wandb_project=self.wandb_project,
                )
            self.total_opt_jobs += len(configs)

    def run(self):
        start = datetime.now()
        # Jupyter
        if self.run_jupyter:
            self.jupyter_work.run()

        # Benchmark training
        if self.run_benchmark_training and not self.all_benchmarks_finished:
            # Check for finished jobs
            for i in self.current_workers:
                if self.train_workers[str(i)].finished:
                    self.current_workers.remove(i)
                    self.succeded.append(i)
                    self.train_workers[str(i)].stop()

            # Queue new jobs
            i = 0
            while (
                len(self.current_workers) < self.max_workers
                and i < self.total_training_jobs
            ):
                if i not in self.succeded and i not in self.current_workers:
                    self.train_workers[str(i)].run()
                    self.current_workers.append(i)
                    print(
                        f"Job {i+1} of {self.total_training_jobs} training jobs queued"
                    )
                i += 1

            if all([w.finished for w in self.train_workers.values()]):
                self.current_workers = []
                self.succeded = []
                self.all_benchmarks_finished = True
                for w in self.train_workers.values():
                    w.stop()
        elif not self.run_benchmark_training:
            self.all_benchmarks_finished = True

        if (
            (
                self.run_single_task
                or self.run_single_task_head_start
                or self.run_multi_task
            )
            and self.all_benchmarks_finished
            # and not self.all_optimization_started
        ):
            #     print(f"Job {i+1} of {self.total_opt_jobs} optimization jobs started")
            #     w.run()
            # self.all_optimization_started = True
            # Check for finished jobs
            for i in self.current_workers:
                if self.opt_workers[str(i)].finished and (
                    (datetime.now() - start).total_seconds() < 0.5
                ):
                    self.current_workers.remove(i)
                    self.succeded.append(i)
                    self.opt_workers[str(i)].stop()

            # # Queue new jobs
            i = 0
            while (
                len(self.current_workers) < self.max_workers
                and i < self.total_opt_jobs
                and ((datetime.now() - start).total_seconds() < 0.5)
            ):
                if i not in self.succeded and i not in self.current_workers:
                    self.opt_workers[str(i)].run()
                    self.current_workers.append(i)
                    print(
                        f"Job {i+1} of {self.total_opt_jobs} optimization jobs queued"
                    )
                i += 1
        end = datetime.now()
        print(f"Run took {(end-start).total_seconds()} seconds")

    def configure_layout(self):
        return {"name": "JupyterLab", "content": self.jupyter_work}

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
                "wandb_optimization_artifact_name": "stbo_reizman_suzuki",
                "wandb_main_dataset_artifact_name": f"reizman_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
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
                "wandb_optimization_artifact_name": "stbo_baumgartner_suzuki",
                "wandb_main_dataset_artifact_name": f"baumgartner_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
        ]

        return reizman_stbo_configs + baumgartner_stbo_configs

    @staticmethod
    def generate_suzuki_configs_single_task_headstart(
        max_experiments: int, batch_size: int, repeats: int, parallel: bool
    ):
        # MTBO Reizman one cotraining with Baumgartner
        reizman_stbo_configs_baugmartner_one = [
            {
                "strategy": "STBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"reizman_suzuki_case_{case}",
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case}:latest",
                "output_path": f"data/singletask_head_start/results_reizman_suzuki_{case}_cotrain_baumgartner_suzuki",
                "wandb_ct_dataset_artifact_name": "baumgartner_suzuki:latest",
                "ct_dataset_names": [f"baumgartner_suzuki"],
                "wandb_optimization_artifact_name": "stbo_reizman_suzuki_one_cotraining_baumgartner_suzuki",
                "wandb_main_dataset_artifact_name": f"reizman_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case in range(1, 5)
        ]

        # MTBO Reizman one cotraining with reizman
        reizman_stbo_configs_reizman_one = [
            {
                "strategy": "STBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"reizman_suzuki_case_{case_main}",
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case_main}:latest",
                "output_path": f"data/singletask_head_start/results_reizman_suzuki_case_{case_main}_cotrain_reizman_suzuki_case_{case_aux}",
                "wandb_ct_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [f"reizman_suzuki_case_{case_aux}"],
                "wandb_optimization_artifact_name": f"stbo_reizman_suzuki_{case_main}_one_cotraining_reizman_suzuki_case_{case_aux}",
                "wandb_main_dataset_artifact_name": f"reizman_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case_main in range(1, 5)
            for case_aux in range(1, 5)
            if case_main != case_aux
        ]

        # MTBO Baumgartner one cotraining with Reizman
        baumgartner_stbo_configs_reizman_one = [
            {
                "strategy": "STBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"baumgartner_suzuki",
                "wandb_benchmark_artifact_name": "benchmark_baumgartner_suzuki:latest",
                "output_path": f"data/singletask_head_start/results_baumgartner_suzuki_cotrain_reizman_suzuki_case_{case}",
                "wandb_ct_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [
                    f"reizman_suzuki_case_{case}",
                ],
                "wandb_optimization_artifact_name": f"stbo_baumgartner_suzuki_one_cotraining_reizman_suzuki_case_{case}",
                "wandb_main_dataset_artifact_name": "baumgartner_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case in range(1, 5)
        ]

        # MTBO Baumgartner cotraining with all Reizman
        baumgartner_stbo_configs_reizman_all = [
            {
                "strategy": "STBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"baumgartner_suzuki",
                "wandb_benchmark_artifact_name": "benchmark_baumgartner_suzuki:latest",
                "output_path": f"data/singletask_head_start/results_baumgartner_suzuki_cotrain_reizman_suzuki_all",
                "wandb_ct_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [
                    f"reizman_suzuki_case_{case}" for case in range(1, 5)
                ],
                "wandb_optimization_artifact_name": f"stbo_baumgartner_suzuki_one_cotraining_reizman_suzuki_all",
                "wandb_main_dataset_artifact_name": "baumgartner_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
        ]

        # MTBO Reizman cotraining with all Reizman
        reizman_stbo_configs_reizman_all = [
            {
                "strategy": "STBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"reizman_suzuki_case_{case_main}",
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case_main}:latest",
                "output_path": f"data/singletask_head_start/results_reizman_suzuki_case_{case_main}_cotrain_reizman_suzuki_case_all",
                "wandb_ct_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [
                    f"reizman_suzuki_case_{case_aux}"
                    for case_aux in range(1, 5)
                    if case_main != case_aux
                ],
                "wandb_optimization_artifact_name": f"stbo_reizman_suzuki_{case_main}_one_cotraining_reizman_suzuki_all",
                "wandb_main_dataset_artifact_name": f"reizman_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case_main in range(1, 5)
        ]

        return (
            reizman_stbo_configs_baugmartner_one
            + reizman_stbo_configs_reizman_one
            + baumgartner_stbo_configs_reizman_one
            + baumgartner_stbo_configs_reizman_all
            + reizman_stbo_configs_reizman_all
        )

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
                "wandb_ct_dataset_artifact_name": "baumgartner_suzuki:latest",
                "ct_dataset_names": [f"baumgartner_suzuki"],
                "wandb_optimization_artifact_name": "mtbo_reizman_suzuki_one_cotraining_baumgartner_suzuki",
                "wandb_main_dataset_artifact_name": f"reizman_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
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
                "wandb_ct_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [f"reizman_suzuki_case_{case_aux}"],
                "wandb_optimization_artifact_name": f"mtbo_reizman_suzuki_{case_main}_one_cotraining_reizman_suzuki_case_{case_aux}",
                "wandb_main_dataset_artifact_name": f"reizman_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case_main in range(1, 5)
            for case_aux in range(1, 5)
            if case_main != case_aux
        ]

        # MTBO Baumgartner one cotraining with Reizman
        baumgartner_mtbo_configs_reizman_one = [
            {
                "strategy": "MTBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"baumgartner_suzuki",
                "wandb_benchmark_artifact_name": "benchmark_baumgartner_suzuki:latest",
                "output_path": f"data/multitask_results/results_baumgartner_suzuki_cotrain_reizman_suzuki_case_{case}",
                "wandb_ct_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [
                    f"reizman_suzuki_case_{case}",
                ],
                "wandb_optimization_artifact_name": f"mtbo_baumgartner_suzuki_one_cotraining_reizman_suzuki_case_{case}",
                "wandb_main_dataset_artifact_name": "baumgartner_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case in range(1, 5)
        ]

        # MTBO Baumgartner cotraining with all Reizman
        baumgartner_mtbo_configs_reizman_all = [
            {
                "strategy": "MTBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"baumgartner_suzuki",
                "wandb_benchmark_artifact_name": "benchmark_baumgartner_suzuki:latest",
                "output_path": f"data/multitask_results/results_baumgartner_suzuki_cotrain_reizman_suzuki_all",
                "wandb_ct_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [
                    f"reizman_suzuki_case_{case}" for case in range(1, 5)
                ],
                "wandb_optimization_artifact_name": f"mtbo_baumgartner_suzuki_one_cotraining_reizman_suzuki_all",
                "wandb_main_dataset_artifact_name": "baumgartner_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
        ]

        # MTBO Baumgartner cotraining with all Reizman except 2
        baumgartner_mtbo_configs_reizman_no_two = [
            {
                "strategy": "MTBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"baumgartner_suzuki",
                "wandb_benchmark_artifact_name": "benchmark_baumgartner_suzuki:latest",
                "output_path": f"data/multitask_results/results_baumgartner_suzuki_cotrain_reizman_suzuki_two",
                "wandb_ct_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [
                    f"reizman_suzuki_case_{case}" for case in range(1, 5) if case != 2
                ],
                "wandb_optimization_artifact_name": f"mtbo_baumgartner_suzuki_one_cotraining_reizman_suzuki_two",
                "wandb_main_dataset_artifact_name": "baumgartner_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
        ]

        # MTBO Reizman cotraining with all Reizman
        reizman_mtbo_configs_reizman_all = [
            {
                "strategy": "MTBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"reizman_suzuki_case_{case_main}",
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case_main}:latest",
                "output_path": f"data/multitask_results/results_reizman_suzuki_case_{case_main}_cotrain_reizman_suzuki_case_all",
                "wandb_ct_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [
                    f"reizman_suzuki_case_{case_aux}"
                    for case_aux in range(1, 5)
                    if case_main != case_aux
                ],
                "wandb_optimization_artifact_name": f"mtbo_reizman_suzuki_{case_main}_one_cotraining_reizman_suzuki_all",
                "wandb_main_dataset_artifact_name": f"reizman_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case_main in range(1, 5)
        ]

        # MTBO Reizman cotraining with all Reizman
        reizman_mtbo_configs_reizman_no_two = [
            {
                "strategy": "MTBO",
                "benchmark_type": BenchmarkType.suzuki,
                "model_name": f"reizman_suzuki_case_{case_main}",
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case_main}:latest",
                "output_path": f"data/multitask_results/results_reizman_suzuki_case_{case_main}_cotrain_reizman_suzuki_case_all",
                "wandb_ct_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [
                    f"reizman_suzuki_case_{case_aux}"
                    for case_aux in range(1, 5)
                    if (case_main != case_aux) and (case_aux != 2)
                ],
                "wandb_optimization_artifact_name": f"mtbo_reizman_suzuki_{case_main}_one_cotraining_reizman_suzuki_all",
                "wandb_main_dataset_artifact_name": f"reizman_suzuki:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case_main in range(1, 5)
            if case_main != 2
        ]

        return (
            reizman_mtbo_configs_baugmartner_one
            + reizman_mtbo_configs_reizman_one
            + baumgartner_mtbo_configs_reizman_one
            + baumgartner_mtbo_configs_reizman_all
            + baumgartner_mtbo_configs_reizman_no_two
            + reizman_mtbo_configs_reizman_no_two
            + reizman_mtbo_configs_reizman_all
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
                "wandb_optimization_artifact_name": "stbo_baumgartner_cn",
                "wandb_main_dataset_artifact_name": f"baumgartner_cn:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case in range(1, 5)
        ]

        return baumgartner_stbo_configs

    @staticmethod
    def generate_cn_configs_singletask_headstart(
        max_experiments: int, batch_size: int, repeats: int, parallel: bool
    ):
        # MTBO Baumgartner cotraining with one Baumgartner
        baumgartner_stbo_configs_reizman_one = [
            {
                "strategy": "STBO",
                "benchmark_type": BenchmarkType.cn,
                "model_name": f"baumgartner_cn_case_{case_main}",
                "wandb_benchmark_artifact_name": f"benchmark_baumgartner_cn_case_{case_main}:latest",
                "output_path": f"data/multitask_results/results_baumgartner_cn_case_{case_main}_cotrain_baumgartner_cn_case_{case_aux}",
                "wandb_ct_dataset_artifact_name": f"baumgartner_cn:latest",
                "ct_dataset_names": [f"baumgartner_cn_case_{case_aux}"],
                "wandb_optimization_artifact_name": f"stbo_baumgartner_cn_{case_main}_one_cotraining_baumgartner_cn_case_{case_aux}",
                "wandb_main_dataset_artifact_name": "baumgartner_cn:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case_main in range(1, 5)
            for case_aux in range(1, 5)
            if case_main != case_aux
        ]

        # MTBO Reizman cotraining with all Baumgartner
        baumgartner_stbo_configs_reizman_all = [
            {
                "strategy": "STBO",
                "benchmark_type": BenchmarkType.cn,
                "model_name": f"baumgartner_cn_case_{case_main}",
                "wandb_benchmark_artifact_name": f"benchmark_baumgartner_cn_case_{case_main}:latest",
                "output_path": f"data/singletask_head_start/results_baumgartner_cn_case_{case_main}_cotrain_baumgartner_cn_case_all",
                "wandb_ct_dataset_artifact_name": f"baumgartner_cn:latest",
                "ct_dataset_names": [
                    f"baumgartner_cn_case_{case_aux}"
                    for case_aux in range(1, 5)
                    if case_main != case_aux
                ],
                "wandb_optimization_artifact_name": f"stbo_baumgartner_cn_{case_main}_one_cotraining_baumgartner_cn_case_all",
                "wandb_main_dataset_artifact_name": "baumgartner_cn:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case_main in range(1, 5)
        ]
        return (
            baumgartner_stbo_configs_reizman_one + baumgartner_stbo_configs_reizman_all
        )

    @staticmethod
    def generate_cn_configs_multitask(
        max_experiments: int, batch_size: int, repeats: int, parallel: bool
    ):
        # MTBO Baumgartner cotraining with one Baumgartner
        baumgartner_mtbo_configs_reizman_one = [
            {
                "strategy": "MTBO",
                "benchmark_type": BenchmarkType.cn,
                "model_name": f"baumgartner_cn_case_{case_main}",
                "wandb_benchmark_artifact_name": f"benchmark_baumgartner_cn_case_{case_main}:latest",
                "output_path": f"data/multitask_results/results_baumgartner_cn_case_{case_main}_cotrain_baumgartner_cn_case_{case_aux}",
                "wandb_ct_dataset_artifact_name": f"baumgartner_cn:latest",
                "ct_dataset_names": [f"baumgartner_cn_case_{case_aux}"],
                "wandb_optimization_artifact_name": f"mtbo_baumgartner_cn_{case_main}_one_cotraining_baumgartner_cn_case_{case_aux}",
                "wandb_main_dataset_artifact_name": "baumgartner_cn:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case_main in range(1, 5)
            for case_aux in range(1, 5)
            if case_main != case_aux
        ]

        # MTBO Reizman cotraining with all Baumgartner
        baumgartner_mtbo_configs_reizman_all = [
            {
                "strategy": "MTBO",
                "benchmark_type": BenchmarkType.cn,
                "model_name": f"baumgartner_cn_case_{case_main}",
                "wandb_benchmark_artifact_name": f"benchmark_baumgartner_cn_case_{case_main}:latest",
                "output_path": f"data/multitask_results/results_baumgartner_cn_case_{case_main}_cotrain_baumgartner_cn_case_all",
                "wandb_ct_dataset_artifact_name": f"baumgartner_cn:latest",
                "ct_dataset_names": [
                    f"baumgartner_cn_case_{case_aux}"
                    for case_aux in range(1, 5)
                    if case_main != case_aux
                ],
                "wandb_optimization_artifact_name": f"mtbo_baumgartner_cn_{case_main}_one_cotraining_baumgartner_cn_case_all",
                "wandb_main_dataset_artifact_name": "baumgartner_cn:latest",
                "brute_force_categorical": True,
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "EI",
                "parallel": parallel,
                "compute_type": "gpu",
            }
            for case_main in range(1, 5)
        ]
        return (
            baumgartner_mtbo_configs_reizman_one + baumgartner_mtbo_configs_reizman_all
        )


app = L.LightningApp(
    MultitaskBenchmarkStudy(
        run_jupyter=True,
        run_benchmark_training=False,
        run_single_task=True,
        run_single_task_head_start=True,
        run_multi_task=True,
        run_suzuki=False,
        split_catalyst_suzuki=False,
        run_cn=False,
        parallel=True,
        max_workers=30,
        wandb_entity="ceb-sre",
        wandb_project="multitask",
    ),
)
