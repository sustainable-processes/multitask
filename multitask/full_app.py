from multitask.suzuki_optimization import SuzukiWork
from multitask.suzuki_benchmark_training import BenchmarkTraining
import lightning as L
import logging


logger = logging.getLogger(__name__)


class MultitaskBenchmarkStudy(L.LightningFlow):
    """
    Benchmarking study of single task vs multitask optimization
    """

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

    def run(self):
        # Benchmark training
        if self.run_benchmark_training:
            # Train Baumgartner benchmark
            baumgartner_runs = [
                BenchmarkTraining(
                    data_path="data/baumgartner_suzuki/ord/baumgartner_suzuki.pb",
                    save_path="data/baumgartner_suzuki/emulator",
                    figure_path="figures/",
                    parallel=True,
                )
            ]

            # Train Reizman benchmarks
            reizman_runs = [
                BenchmarkTraining(
                    data_path=f"data/reizman_suzuki/ord/reizman_suzuki_case_{case}.pb",
                    save_path=f"data/reizman_suzuki/emulator_case_{case}/",
                    figure_path="figures/",
                    parallel=True,
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
        if self.run_single_task:
            runs = [
                SuzukiWork(**config)
                for config in self.generate_suzuki_configs_single_task(
                    max_experiments=self.max_experiments,
                    batch_size=self.batch_size,
                    repeats=self.repeats,
                )
            ]
            for r in runs:
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
            for r in runs:
                r.run()

    @staticmethod
    def generate_suzuki_configs_single_task(
        max_experiments: int, batch_size: int, repeats: int
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
                "parallel": False,
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
                "parallel": True,
            }
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
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case}:latest",
                "output_path": f"data/multitask_results/results_reizman_suzuki_{case}_cotrain_baumgartner_suzuki",
                "wandb_dataset_artifact_name": "baumgartner_suzuki:latest",
                "ct_dataset_names": [f"baumgartner_suzuki"],
                "wandb_artifact_name": "mtbo_reizman_suzuki_one_cotraining_baumgartner_suzuki",
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
                "wandb_benchmark_artifact_name": f"benchmark_reizman_suzuki_case_{case_main}:latest",
                "output_path": f"data/multitask_results/results_reizman_suzuki_case_{case_main}_cotrain_reizman_suzuki_case_{case_aux}",
                "wandb_dataset_artifact_name": f"reizman_suzuki:latest",
                "ct_dataset_names": [f"reizman_suzuki_case_{case_aux}"],
                "wandb_artifact_name": f"mtbo_reizman_suzuki_{case_main}_one_cotraining_reizman_suzuki_case_{case_aux}",
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
        run_benchmark_training=False, run_single_task=True, run_multi_task=False
    )
)
