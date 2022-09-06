from pathlib import Path
import os.path as osp
from multitask.suzuki_optimization import SuzukiWork
from typing import Dict, Optional, Literal
import lightning as L
from lightning.app import structures
from lightning.app.frontend import StreamlitFrontend
from lightning_app.components.python import TracerPythonScript
import wandb
import logging
import os.path as ops


logger = logging.getLogger(__name__)


class BenchmarkTraining(TracerPythonScript):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb_run = None

    def run(self):
        # Download data
        wandb_run = wandb.init(job_type="training")

        # Train model using script
        super().run()

        # Upload to wandb
        name = Path(self.script_args[-3]).parts[-1].rstrip(".pb")
        artifact = wandb.Artifact(name, type="model")
        artifact.add_dir(self.script_args[-2])
        artifact.add_file(osp.join(self.script_args[-1], f"/{name}_parity_plot.png"))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()


class MultitaskBenchmarkStudy(L.LightningFlow):
    def __init__(
        self, run_benchmark_training: bool, run_single_task: bool, run_multi_task: bool
    ):
        super().__init__()

        self.max_experiments = 20
        self.batch_size = 1
        self.repeats = 20
        self.run_benchmark_training = run_benchmark_training
        self.run_single_task = run_single_task
        self.run_multi_task = run_multi_task

        # Download data if not already downloaded

    def run(self):
        # Benchmark training
        if self.run_benchmark_training:
            # Train Baumgartner benchmark
            baumgartner_runs = [
                BenchmarkTraining(
                    script_path=ops.join(
                        ops.dirname(__file__), "suzuki_benchmark_training.py"
                    ),
                    script_args=[
                        "--no-split-catalyst",
                        # "--max-epochs=1000",
                        # "--cv-folds=5",
                        # "-verbose=0",
                        "--no-print-warnings",
                        "data/baumgartner_suzuki/ord/baumgartner_suzuki.pb",
                        "data/baumgartner_suzuki/emulator",
                        "figures/",
                    ],
                    parallel=True,
                )
            ]

            # Train Reizman benchmarks
            reizman_runs = [
                BenchmarkTraining(
                    script_path=ops.join(
                        ops.dirname(__file__), "suzuki_benchmark_training.py"
                    ),
                    script_args=[
                        f"data/reizman_suzuki/ord/reizman_suzuki_case_{case}.pb",
                        f"data/reizman_suzuki/emulator_case_{case}/",
                        "figures/",
                        # "--no-split-catalyst",
                        # "--max-epochs 1000",
                        # "--cv-folds 5",
                        # "-verbose 0",
                        "--no-print-warnings",
                    ],
                    parallel=True,
                )
                for case in range(1, 5)
            ]
            for r in reizman_runs + baumgartner_runs:
                r.run()

        # Single task benchmarking
        if self.run_single_task:
            runs = [
                SuzukiWork(**config)
                for config in self.generate_suzuki_configs_single_task(
                    max_experiments=self.max_experiments,
                    batch_size=self.batch_size,
                    repeats=self.repeats,
                )
            ]
            for r in runs.items():
                r.run()

        # Multi task benchmarking
        if self.run_multi_task:
            runs = [
                SuzukiWork(**config)
                for config in self.generate_suzuki_configs_multitask(
                    max_experiments=self.max_experiments,
                    batch_size=self.batch_size,
                    repeats=self.repeats,
                )
            ]
            for r in runs.items():
                r.run()

    @staticmethod
    def generate_suzuki_configs_single_task(
        max_experiments: int, batch_size: int, repeats: int
    ):
        # MTBO Reizman
        reizman_stbo_configs = [
            {
                "strategy": "STBO",
                "model_name": f"reizman_suzuki_case_{case}",
                "benchmark_path": f"data/reizman_suzuki/emulator_case_{case}",
                "output_path": f"data/reizman_suzuki/results_stbo_case_{case}/",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "qNEI",
                "parallel": True,
            }
            for case in range(1, 5)
        ]

        # STBO Baumgartner
        baumgartner_stbo_configs = [
            {
                "strategy": "STBO",
                "model_name": f"baumgartner_suzuki_case_{case}",
                "benchmark_path": f"data/baumgartner_suzuki/emulator_case_{case}",
                "output_path": f"data/baumgarnter_suzuki/results_stbo_case_{case}/",
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "qNEI",
                "parallel": True,
            }
            for case in range(1, 5)
        ]

        return reizman_stbo_configs + baumgartner_stbo_configs

    @staticmethod
    def generate_suzuki_configs_multitask(
        max_experiments: int, batch_size: int, repeats: int
    ):
        # MTBO Reizman one cotraining with Baumgartner
        reizman_mtbo_configs_baugmartner_one = [
            {
                "strategy": "MTBO",
                "model_name": f"reizman_suzuki_case_{case}",
                "benchmark_path": f"data/reizman_suzuki/emulator_case_{case}",
                "output_path": f"data/multitask_results/results_reizman_suzuki_{case}_cotrain_baumgartner_suzuki",
                "ct_data_paths": [f"data/baumgartner_suzuki/ord/baumgartner_suzuki.pb"],
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "qNEI",
                "parallel": True,
            }
            for case in range(1, 5)
        ]

        # MTBO Reizman one cotraining with reizman
        reizman_mtbo_configs_reizman_one = [
            {
                "strategy": "MTBO",
                "model_name": f"reizman_suzuki_case_{case_main}",
                "benchmark_path": f"data/reizman_suzuki/emulator_case_{case_main}",
                "output_path": f"data/multitask_results/results_reizman_suzuki_case_{case_main}_cotrain_reizman_suzuki_case_{case_aux}",
                "ct_data_paths": [
                    f"data/reizman_suzuki/ord/reizman_suzuki_case_{case_aux}.pb"
                ],
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "qNEI",
                "parallel": True,
            }
            for case_main in range(1, 5)
            for case_aux in range(1, 5)
        ]

        # MTBO Baumgartner one cotraining with Reizman
        baumgartner_mtbo_configs_reizman_one = [
            {
                "strategy": "MTBO",
                "model_name": f"baumgartner_suzuki",
                "benchmark_path": f"data/baumgartner_suzuki/emulator",
                "output_path": f"data/multitask_results/results_baumgartner_suzuki_cotrain_reizman_suzuki_case_{case}",
                "ct_data_paths": [
                    f"data/reizman_suzuki/ord/reizman_suzuki_case_{case}.pb",
                ],
                "max_experiments": max_experiments,
                "batch_size": batch_size,
                "repeats": repeats,
                "acquisition_function": "qNEI",
                "parallel": True,
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
        run_benchmark_training=True, run_single_task=False, run_multi_task=False
    )
)
