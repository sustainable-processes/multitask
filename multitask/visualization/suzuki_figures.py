"""
Make figures for publication
"""
from multitask.utils import download_runs_wandb
from .plots import make_comparison_plot, remove_frame
from pathlib import Path
from summit import *
from rdkit import Chem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
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

    # Filter data
    stbo_dfs = [
        run.history()
        for run in runs
        if run.config.get("model_name") == "baumgartner_suzuki" and "STBO" in run.tags
    ]
    mtbo_dfs = [
        run.history()
        for run in runs
        if run.config.get("model_name") == "baumgartner_suzuki" and "MTBO" in run.tags
    ]

    if len(stbo_dfs) == 0:
        raise ValueError("No Baumbartner STBO runs found")
    if len(mtbo_dfs) == 0:
        raise ValueError("No Baumgarnter MTBO runs found")

    # Setup figure
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.2, hspace=0.5)
    k = 1
    axis_fontsize = 14

    # Make figure
    for i in range(1, 5):
        ax = fig.add_subplot(1, 4, k)
        make_comparison_plot(
            dict(results=stbo_dfs, label="STBO", color="#a50026"),
            dict(results=mtbo_dfs, label="MTBO", color="#313695"),
            output_name="yld_best",
            ax=ax,
        )
        ax.set_title(f"Auxiliary task: Reizman Case {i}", fontsize=21)
        ax.set_xlim(0, 20)
        ax.tick_params("y", labelsize=axis_fontsize)
        xlabels = np.arange(0, 21, 5)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(xlabels, fontsize=axis_fontsize)
        ax.set_ylim(0, 100)
        remove_frame(ax, sides=["right", "top"])
        k += 1

    # Format and save figure
    # fig.suptitle("Baumgartner Optimization")
    fig.supxlabel("Number of experiments", fontsize=21)
    fig.supylabel("Yield (%)", fontsize=21)
    fig.tight_layout()
    fig.savefig(
        "../figures/baumgartner_reizman_one_cotraining_optimization.png",
        dpi=300,
        transparent=True,
    )
    fig.savefig(
        "../figures/baumgartner_reizman_one_cotraining_optimization.svg",
        transparent=True,
    )


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
    runs = download_runs_wandb(
        api,
        wandb_entity,
        wandb_project,
        only_finished_runs=only_finished_runs,
        include_tags=include_tags,
        filter_tags=filter_tags,
    )

    # Setup figure
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.2, hspace=0.5)
    k = 1
    axis_fontsize = 14

    # Make subplots for each case
    letters = ["a", "b", "c", "d"]
    for i in range(1, 5):
        # Filter data
        stbo_dfs = [
            run.history()
            for run in runs
            if run.config.get("model_name") == f"reizman_suzuki_case_{i}"
            and run.config.get("strategy") == "STBO"
        ]
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
        remove_frame(ax, sides=["right", "top"])
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
