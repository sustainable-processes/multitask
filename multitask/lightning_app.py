import logging
from multitask.suzuki_optimization import SuzukiWork
from typing import Optional
import lightning as L

N_RETRIES = 5
WANDB_SETTINGS = {"wandb_entity": "ceb-sre", "wandb_project": "multitask"}

logger = logging.getLogger(__name__)


class MultitaskBenchmarkStudy(L.LightningFlow):
    def __init__(
        self, run_suzuki: bool, max_experiments: int, batch_size: int, repeats: int
    ):
        super().__init__()
        self.max_experiments = max_experiments
        self.batch_size = batch_size
        self.repeats = repeats
        self.run_suzuki = run_suzuki

    def run(self):
        # Suzuki benchmarking studies
        if self.run_suzuki:
            suzuki_runs = [
                SuzukiWork(**config)
                for config in self.generate_suzuki_configs(
                    max_experiments=self.max_experiments,
                    batch_size=self.batch_size,
                    repeats=self.repeats,
                )
            ]
            for r in suzuki_runs:
                r.run()

    def generate_suzuki_configs(
        self, max_experiments: int, batch_size: int, repeats: int
    ):
        # Single task Reizman
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
            }
            for case in range(1, 5)
        ]

        return reizman_stbo_configs


app = L.LightningApp(
    MultitaskBenchmarkStudy(
        run_suzuki=True, max_experiments=20, batch_size=5, repeats=5
    )
)
