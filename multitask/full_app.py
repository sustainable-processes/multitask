from copy import deepcopy
import uuid
from multitask.suzuki_optimization import SuzukiWork
from typing import Dict, Optional, Literal
import lightning as L
from lightning.app import structures
from lightning.app.frontend import StreamlitFrontend
from lightning_app.components.python import TracerPythonScript
import wandb
import logging
import os.path as ops

N_RETRIES = 5

logger = logging.getLogger(__name__)

RequestMethods = Literal[
    "train_benchmarks", "run_single_task_benchmarks", "run_multi_task_benchmarks"
]


class Request:
    def __init__(self, method: RequestMethods):
        self.method = method
        self.payload = {}

    def to_dict(self):
        return {"method": self.method, "payload": self.payload}

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)


class MultitaskBenchmarkStudy(L.LightningFlow):
    def __init__(self):
        super().__init__()

        self.max_experiments = 20
        self.batch_size = 1
        self.repeats = 20
        self.benchmark_runs = structures.Dict()
        self.suzuki_single_task_runs = structures.Dict()
        self.suzuki_multi_task_runs = structures.Dict()
        self.requests = []

        # Download data if not already downloaded

    def run(self):
        # Hanndle request
        for request_dict in self.requests:
            self._handle_request(Request.from_dict(request_dict))

    def handle_request(self, request):
        # Benchmark training
        if request.method == "train_benchmarks":
            # Train Baumgartner benchmark
            baumgartner_runs = {
                uuid.uuid4(): TracerPythonScript(
                    script_path=ops.join(
                        ops.dirname(__file__), "suzuki_benchmark_training.py"
                    ),
                    script_args=[
                        "data/baumgartner_suzuki/ord/baumgartner_suzuki.pb",
                        "data/baumgartner_suzuki/emulator",
                        "figures/",
                        "--no-split-catalyst",
                        "--max-epochs 1000",
                        "--cv-folds 5",
                        "-verbose 0",
                        "--no-print-warnings",
                    ],
                    parallel=True,
                )
                for case in range(1, 5)
            }

            # Train Reizman benchmarks
            reizman_runs = {
                uuid.uuid4(): TracerPythonScript(
                    script_path=ops.join(
                        ops.dirname(__file__), "suzuki_benchmark_training.py"
                    ),
                    script_args=[
                        f"data/reizman_suzuki/ord/reizman_suzuki_case_{case}.pb",
                        f"data/reizman_suzuki/emulator_case_{case}/",
                        "figures/",
                        "--no-split-catalyst",
                        "--max-epochs 1000",
                        "--cv-folds 5",
                        "-verbose 0",
                        "--no-print-warnings",
                    ],
                    parallel=True,
                )
                for case in range(1, 5)
            }
            for id, r in reizman_runs + baumgartner_runs:
                r.run()
                self.benchmark_runs[id] = r

        # Single task benchmarking
        elif request.method == "run_single_task_benchmarks":
            runs = {
                uuid.uuid4(): SuzukiWork(**config)
                for config in self.generate_suzuki_configs_single_task(
                    max_experiments=self.max_experiments,
                    batch_size=self.batch_size,
                    repeats=self.repeats,
                )
            }
            for id, r in runs.items():
                r.run()
                self.suzuki_single_task_runs[id] = r
        # Multi task benchmarking
        elif request.method == "run_multi_task_benchmarks":
            runs = {
                uuid.uuid4(): SuzukiWork(**config)
                for config in self.generate_suzuki_configs_multitask(
                    max_experiments=self.max_experiments,
                    batch_size=self.batch_size,
                    repeats=self.repeats,
                )
            }
            for id, r in runs.items():
                r.run()
                self.suzuki_multi_task_runs[id] = r

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

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_fn)


def render_fn(state):
    import time
    import streamlit as st

    # Training
    st.markdown(
        """
        # Multitask Benchmarking

        ## 1. Benchmark Training
        """
    )
    st.write("State")
    st.write(state._state)
    training_works = state._state["structures"]["train_benchmarks"]["works"]
    if len(training_works) > 0:

        # Should tag runs and print that tag out
        st.markdown("See results on [wandb](https://wandb.ai/ceb-sre/multitask)!")

        # Display a progress bar?
    else:
        if st.button("Train benchmarks"):
            state.requests.append(Request("train_benchmarks"))

    # Benchmarking
    st.markdown(
        """
        ## 2. Suzuki Benchmarking
        """
    )
    if st.button("Run single task benchmarking"):
        # Should tag runs and print that tag out
        st.write("Running single task benchmarking")
        st.markdown("See results on [wandb](https://wandb.ai/ceb-sre/multitask)")

    if st.button("Run multitask benchmarking"):
        # Should tag runs and print that tag out
        st.write("Running multi task benchmarking")
        st.markdown("See results on [wandb](https://wandb.ai/ceb-sre/multitask)")


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.flow = MultitaskBenchmarkStudy()

    def run(self):
        self.flow.run()

    def configure_layout(self):
        # Main UI
        return {
            "name": "Multitask Benchmarking",
            "content": self.flow,
        }


app = L.LightningApp(RootFlow())
