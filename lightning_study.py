import subprocess
from typing import List, Optional
from multitask.suzuki_benchmark_training import BenchmarkTraining
import lightning as L
from lightning_app.structures.dict import Dict
import logging
import wandb
import os


logger = logging.getLogger(__name__)


class SummitBuildConfig(L.BuildConfig):
    def build_commands(self) -> List[str]:
        return [
            "pip install .",
        ]


class SuzukiWork(L.LightningWork):
    def __init__(
        self,
        strategy: str,
        model_name: str,
        wandb_benchmark_artifact_name: str,
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
        **kwargs,
    ):
        super().__init__(
            parallel=parallel,
            cloud_build_config=SummitBuildConfig(),
            cloud_compute=cloud_compute,
        )
        self.strategy = strategy
        self.model_name = model_name
        self.wandb_benchmark_artifact_name = wandb_benchmark_artifact_name
        self.wandb_dataset_artifact_name = wandb_dataset_artifact_name
        self.ct_dataset_names = ct_dataset_names
        self.output_path = output_path
        self.max_experiments = max_experiments
        self.batch_size = batch_size
        self.brute_force_categorical = brute_force_categorical
        self.acquisition_function = acquisition_function
        self.repeats = repeats
        self.wandb_artifact_name = wandb_artifact_name
        self.print_warnings = print_warnings

    def run(self):
        wandb.login(key=os.environ.get("WANDB_API_KEY"))

        cmd = f"""
        python multitask/suzuki_optimization.py \
        {self.strategy.lower()} \
        {self.model_name} \
        {self.wandb_benchmark_artifact_name} \
        """
        if self.strategy == "MTBO":
            cmd += f" {self.wandb_dataset_artifact_name} "
            cmd += " ".join([c + " " for c in self.ct_dataset_names])
        cmd += f" {self.output_path} \ "
        options = f"""
        --max-experiments {self.max_experiments} \
        --batch-size {self.batch_size} \
        --repeats {self.repeats} \
        --wandb-artifact-name {self.wandb_artifact_name}  \
        --acquisition-function {self.acquisition_function} \
        --print-warnings {self.print_warnings}
        """
        if self.brute_force_categorical:
            options += "\n--brute-force-categorical"
        options = ""
        try:
            subprocess.run(cmd + options, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise (e)


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

        # Workers
        self.all_running = False
        self.workers = {}
        self.count = 0

        # Benchmark training
        if self.run_benchmark_training and not self.all_running:
            # Train Baumgartner benchmark
            baumgartner_runs = [
                BenchmarkTraining(
                    data_path="data/baumgartner_suzuki/ord/baumgartner_suzuki.pb",
                    save_path="data/baumgartner_suzuki/emulator",
                    figure_path="figures/",
                    parallel=True,
                    cloud_compute=L.CloudCompute(self.compute_type),
                )
            ]

            # Train Reizman benchmarks
            reizman_runs = [
                BenchmarkTraining(
                    data_path=f"data/reizman_suzuki/ord/reizman_suzuki_case_{case}.pb",
                    save_path=f"data/reizman_suzuki/emulator_case_{case}/",
                    figure_path="figures/",
                    parallel=True,
                    cloud_compute=L.CloudCompute(name=self.compute_type),
                )
                for case in range(1, 5)
            ]
            for r in reizman_runs + baumgartner_runs:
                r.run(
                    split_catalyst=False,
                    max_epochs=1000,
                    cv_folds=5,
                    verbose=1,
                    print_warnings=False,
                )

        # Single task benchmarking
        if self.run_single_task and not self.all_running:
            configs = self.generate_suzuki_configs_single_task(
                max_experiments=self.max_experiments,
                batch_size=self.batch_size,
                repeats=self.repeats,
                parallel=self.parallel,
            )
            self.runner = SuzukiWork(
                **configs[0],
                cloud_compute=L.CloudCompute(self.compute_type, shm_size=4096),
            )
        #     for i, config in enumerate(configs):
        #         self.workers[str(self.count + i)] = SuzukiWork(
        #             **config,
        #             cloud_compute=L.CloudCompute(self.compute_type, shm_size=4096),
        #         )
        #     self.count += len(configs)

        # # Multi task benchmarking
        # if self.run_multi_task and not self.all_running:
        #     configs = self.generate_suzuki_configs_multitask(
        #         max_experiments=self.max_experiments,
        #         batch_size=self.batch_size,
        #         repeats=self.repeats,
        #         parallel=self.parallel,
        #     )
        #     for i, config in enumerate(configs):
        #         self.workers[str(self.count + i)] = SuzukiWork(
        #             **config,
        #             cloud_compute=L.CloudCompute(self.compute_type, shm_size=4096),
        #         )
        #     self.count += len(configs)

    def run(self):
        if not self.all_running:
            self.runner.run()
            # for i in range(self.count):
            #     self.workers[str(i)].run()
            self.all_running = True

    @staticmethod
    def generate_suzuki_configs_single_task(
        max_experiments: int, batch_size: int, repeats: int, parallel: bool
    ):
        # STBO Reizman
        reizman_stbo_configs = [
            {
                "strategy": "STBO",
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case}:latest",
                "model_name": f"reizman_suzuki_case_{case}",
                "output_path": f"data/reizman_suzuki/results_stbo_case_{case}/",
                "wandb_artifact_name": "stbo_reizman_suzuki",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "qNEI",
                "parallel": parallel,
            }
            for case in range(1, 5)
        ]

        # STBO Baumgartner
        baumgartner_stbo_configs = [
            {
                "strategy": "STBO",
                "model_name": f"baumgartner_suzuki",
                "wandb_benchmark_artifact_name": "benchmark_baumgartner_suzuki:latest",
                "output_path": f"data/baumgarnter_suzuki/results_stbo/",
                "wandb_artifact_name": "stbo_baumgartner_suzuki",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "qNEI",
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
                "model_name": f"reizman_suzuki_case_{case}",
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case}:latest",
                "output_path": f"data/multitask_results/results_reizman_suzuki_{case}_cotrain_baumgartner_suzuki",
                "wandb_dataset_artifact_name": "baumgartner_suzuki:latest",
                "ct_dataset_names": [f"baumgartner_suzuki"],
                "wandb_artifact_name": "mtbo_reizman_suzuki_one_cotraining_baumgartner_suzuki",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "qNEI",
                "parallel": parallel,
            }
            for case in range(1, 5)
        ]

        # MTBO Reizman one cotraining with reizman
        reizman_mtbo_configs_reizman_one = [
            {
                "strategy": "MTBO",
                "model_name": f"reizman_suzuki_case_{case_main}",
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case_main}:latest",
                "output_path": f"data/multitask_results/results_reizman_suzuki_case_{case_main}_cotrain_reizman_suzuki_case_{case_aux}",
                "wandb_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [f"reizman_suzuki_case_{case_aux}"],
                "wandb_artifact_name": f"mtbo_reizman_suzuki_{case_main}_one_cotraining_reizman_suzuki_case_{case_aux}",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "qNEI",
                "parallel": parallel,
            }
            for case_main in range(1, 5)
            for case_aux in range(1, 5)
        ]

        # MTBO Baumgartner one cotraining with Reizman
        baumgartner_mtbo_configs_reizman_one = [
            {
                "strategy": "MTBO",
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
                "acquisition_function": "qNEI",
                "parallel": parallel,
            }
            for case in range(1, 5)
        ]
        return (
            reizman_mtbo_configs_baugmartner_one
            + reizman_mtbo_configs_reizman_one
            + baumgartner_mtbo_configs_reizman_one
        )


app = L.LightningApp(
    MultitaskBenchmarkStudy(
        run_benchmark_training=False,
        run_single_task=True,
        run_multi_task=True,
        compute_type="cpu-medium",
        parallel=False,
    )
)
