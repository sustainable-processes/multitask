"""
Make figures for publication
"""
from .plots import make_comparison_plot, get_wandb_run_dfs
from pathlib import Path
from summit import *
from rdkit import Chem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional
import wandb
import logging
import string


logger = logging.getLogger(__name__)


def baumgartner_suzuki_auxiliary_reizman_suzuki(
    num_iterations: int = 20,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    wandb_entity: str = "ceb-sre",
    wandb_project: str = "multitask",
    figure_dir: str = "figures",
):
    # Wandb API
    api = wandb.Api()

    # STBO downloads
    logger.info("Getting Baumgartner Suzuki STBO data")
    stbo_dfs = get_wandb_run_dfs(
        api,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        model_name="baumgartner_suzuki",
        strategy="STBO",
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        num_iterations=num_iterations,
    )

    # Setup figure
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.2, hspace=0.5)
    k = 1
    axis_fontsize = 14
    heading_fontsize = 18
    logger.info(
        "Making plots for Baumgartner Suzuki optimization with auxiliary of Reizman Suzuki"
    )
    for i in range(1, 5):
        # Get MTBO data
        logger.info(
            f"Getting Baumgartner Suzuki MTBO (cotrain Reizman suzuki {i}) data"
        )
        mtbo_dfs = get_wandb_run_dfs(
            api,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            model_name="baumgartner_suzuki",
            strategy="MTBO",
            include_tags=include_tags,
            filter_tags=filter_tags,
            only_finished_runs=only_finished_runs,
            num_iterations=num_iterations,
            extra_filters={
                "config.ct_dataset_names": [f"reizman_suzuki_case_{i}"],
            },
        )

        # Make comparison subplot
        ax = fig.add_subplot(1, 4, k)
        make_comparison_plot(
            dict(results=stbo_dfs, label="STBO", color="#a50026"),
            dict(results=mtbo_dfs, label="MTBO", color="#313695"),
            output_name="yld_best",
            ax=ax,
        )

        # Format subplot
        ax.set_title(f"Reizman Case {i}", fontsize=21)
        ax.set_xlim(0, 20)
        ax.tick_params("y", labelsize=axis_fontsize)
        xlabels = np.arange(0, 21, 5)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(xlabels, fontsize=axis_fontsize)
        ax.set_ylim(0, 100)
        k += 1

    # Format and save figure
    # fig.suptitle("Baumgartner Optimization")
    fig.supxlabel("Number of experiments", fontsize=heading_fontsize)
    fig.supylabel("Yield (%)", fontsize=heading_fontsize)
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    fig.savefig(
        figure_dir / "baumgartner_reizman_one_cotraining_optimization.png",
        dpi=300,
        transparent=True,
    )
    fig.savefig(
        figure_dir / "baumgartner_reizman_one_cotraining_optimization.svg",
        transparent=True,
    )
    logger.info("Plots saved to %s", figure_dir)


def reizman_suzuki_auxiliary_baumgartner_suzuki(
    num_iterations: int = 20,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    wandb_entity: str = "ceb-sre",
    wandb_project: str = "multitask",
    figure_dir: str = "figures",
):
    # Get runs
    api = wandb.Api()
    x_axis = "iteration"
    keys = ["yld_best"]

    # Setup figure
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.2, hspace=0.5)
    k = 1
    axis_fontsize = 14

    # Make subplots for each case
    letters = ["a", "b", "c", "d"]
    logger.info(
        "Making plots for Reizman Suzuki optimization with auxiliary of Baumgartner Suzuki"
    )
    for i in range(1, 5):
        # Filter data
        logger.info(f"Getting Reizman Suzuki case {i} STBO data")
        stbo_filters = {
            "config.model_name": "baumgartner_suzuki",
            "config.strategy": "STBO",
        }
        stbo_runs = download_runs_wandb(
            api,
            wandb_entity,
            wandb_project,
            only_finished_runs=only_finished_runs,
            include_tags=include_tags,
            filter_tags=filter_tags,
            extra_filters=stbo_filters,
        )
        stbo_dfs = [run.history(x_axis=x_axis, keys=keys) for run in tqdm(stbo_runs)]
        stbo_dfs = [
            stbo_df for stbo_df in stbo_dfs if stbo_df.shape[0] == num_iterations
        ]
        if len(stbo_dfs) == 0:
            raise ValueError("No Reizman STBO runs found")
        mtbo_dfs = [
            run.history()
            for run in runs
            if run.config.get("strategy") == "MTBO"
            and run.config.get("model_name") == f"reizman_suzuki_case_{i}"
            and run.config.get("ct_dataset_names")[0] == "baumgartner_suzuki"
            and len(run.config.get("ct_dataset_names")) == 1
        ]
        stbo_dfs = [
            stbo_df for stbo_df in stbo_dfs if stbo_df.shape[0] == num_iterations
        ]
        mtbo_dfs = [
            mtbo_df for mtbo_df in mtbo_dfs if mtbo_df.shape[0] == num_iterations
        ]
        if len(stbo_dfs) == 0:
            raise ValueError("No Reizman STBO runs found")
        if len(mtbo_dfs) == 0:
            raise ValueError("No Reizman MTBO runs found")
        logger.info(
            f"Found {len(stbo_dfs)} STBO and {len(mtbo_dfs)} MTBO runs for Reizman Suzuki case {i} with auxiliary of Baumgarnter Suzuki",
        )

        # Make subplot
        ax = fig.add_subplot(1, 4, k)
        make_comparison_plot(
            dict(results=stbo_dfs, label="STBO", color="#a50026"),
            dict(results=mtbo_dfs, label="MTBO", color="#313695"),
            output_name="yld_best",
            ax=ax,
        )

        # Format subplot
        ax.set_title(f"({letters[i - 1]})", fontsize=21)
        ax.set_xlim(0, 20)
        ax.tick_params("y", labelsize=axis_fontsize)
        xlabels = np.arange(0, 21, 5)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(xlabels, fontsize=axis_fontsize)
        ax.set_ylim(0, 100)
        # remove_frame(ax, sides=["right", "top"])
        k += 1

    # Format and save figure
    # fig.suptitle("Reizman Optimization", fontsize=21)
    fig.supxlabel("Number of experiments", fontsize=21)
    fig.supylabel("Yield (%)", fontsize=21)
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    fig.savefig(
        figure_dir / "reizman_baumgartner_one_cotraining_optimization.png", dpi=300
    )
    fig.savefig(
        figure_dir / "reizman_baumgartner_one_cotraining_optimization.svg", dpi=300
    )


def reizman_suzuki_auxiliary_reizman_suzuki(
    num_iterations: int = 20,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    wandb_entity: str = "ceb-sre",
    wandb_project: str = "multitask",
    figure_dir: str = "figures",
):
    """Make plots for Reizman Suzuki optimization with auxiliary of Reizman Suzuki."""
    # Get runs
    api = wandb.Api()
    runs = download_runs_wandb(
        api,
        wandb_entity,
        wandb_project,
        only_finished_runs=only_finished_runs,
        include_tags=include_tags,
        filter_tags=filter_tags,
    )

    # Setup figure
    fig = plt.figure(figsize=(15, 15))
    fig.subplots_adjust(wspace=0.2, hspace=0.5)
    k = 1
    axis_fontsize = 14

    # Make subplots for each case
    letters = list(string.ascii_lowercase)
    logger.info(
        "Making plots for Reizman Suzuki optimization with auxiliary of Reizman Suzuki"
    )
    for i in range(1, 5):
        # Filter STBO data
        stbo_dfs = [
            run.history()
            for run in runs
            if run.config.get("model_name") == f"reizman_suzuki_case_{i}"
            and run.config.get("strategy") == "STBO"
        ]
        stbo_dfs = [
            stbo_df for stbo_df in stbo_dfs if stbo_df.shape[0] == num_iterations
        ]
        if len(stbo_dfs) == 0:
            raise ValueError("No Reizman STBO runs found")
        for j in range(1, 5):
            if i != j:
                # Filter MTBO data
                mtbo_dfs = [
                    run.history()
                    for run in runs
                    if run.config.get("strategy") == "MTBO"
                    and run.config.get("model_name") == f"reizman_suzuki_case_{i}"
                    and run.config.get("ct_dataset_names")[0]
                    == "reizman_suzuki_case_{j}"
                    and len(run.config.get("ct_dataset_names")) == 1
                ]
                mtbo_dfs = [
                    mtbo_df
                    for mtbo_df in mtbo_dfs
                    if mtbo_df.shape[0] == num_iterations
                ]
                if len(mtbo_dfs) == 0:
                    raise ValueError(
                        f"No Reizman MTBO runs found for case {i} (auxiliary reizman {j})"
                    )
                logger.info(
                    f"Found {len(stbo_dfs)} STBO and {len(mtbo_dfs)} MTBO runs for Reizman Suzuki case {i} with auxiliary of Reizman Suzuki case {j}"
                )

                # Make subplot
                ax = fig.add_subplot(4, 3, k)
                make_comparison_plot(
                    dict(results=stbo_dfs, label="STBO", color="#a50026"),
                    dict(results=mtbo_dfs, label="MTBO", color="#313695"),
                    ax=ax,
                )
                ax.set_title(f"({letters[k-1]}) Case {i} - Auxiliary {j}")
                ax.set_ylim(0, 100)
                k += 1

    # Format plot
    fig.suptitle("Reizman Optimization")
    fig.supxlabel("Number of reactions")
    fig.supylabel("Yield (%)")
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    fig.savefig(figure_dir / "reizman_reizman_one_cotraining_optimization.png", dpi=300)
