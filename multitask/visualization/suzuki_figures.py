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


reizman_case_to_name = {1: "SR1", 2: "SR2", 3: "SR3", 4: "SR4"}


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
    """Make plots for Baumgartner Suzuki optimization with auxiliary of Reizman Suzuki."""
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
        ax.set_title(f"Aux. {reizman_case_to_name[i]}", fontsize=heading_fontsize)
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
    fig.supylabel("Best Yield (%)", fontsize=heading_fontsize)
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
    """Make plots for Reizman Suzuki optimization with auxiliary of Baumgartner Suzuki"""
    # Wandb API
    api = wandb.Api()

    # Setup figure
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.2, hspace=0.5)
    k = 1
    axis_fontsize = 14
    heading_fontsize = 18

    # Make subplots for each case
    # letters = ["a", "b", "c", "d"]
    logger.info(
        "Making plots for Reizman Suzuki optimization with auxiliary of Baumgartner Suzuki"
    )
    for i in range(1, 5):
        # STBO data
        logger.info(f"Getting Reizman Suzuki case {i} STBO data")
        stbo_dfs = get_wandb_run_dfs(
            api,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            model_name=f"reizman_suzuki_case_{i}",
            strategy="STBO",
            include_tags=include_tags,
            filter_tags=filter_tags,
            only_finished_runs=only_finished_runs,
            num_iterations=num_iterations,
        )

        # MTBO data
        logger.info(
            f"Getting Reizman Suzuki case {i} (auxiliary Baumgartner suzuki) MTBO data"
        )
        mtbo_dfs = get_wandb_run_dfs(
            api,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            model_name=f"reizman_suzuki_case_{i}",
            strategy="MTBO",
            include_tags=include_tags,
            filter_tags=filter_tags,
            only_finished_runs=only_finished_runs,
            num_iterations=num_iterations,
            extra_filters={
                "config.ct_dataset_names": [f"baumgartner_suzuki"],
            },
        )

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
        ax.set_title(f"{reizman_case_to_name[i]} (Aux. SB1)", fontsize=heading_fontsize)
        ax.set_xlim(0, 20)
        ax.tick_params("y", labelsize=axis_fontsize)
        xlabels = np.arange(0, 21, 5)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(xlabels, fontsize=axis_fontsize)
        ax.set_ylim(0, 100)
        k += 1

    # Format and save figure
    # fig.suptitle("Reizman Optimization", fontsize=21)
    fig.supxlabel("Number of experiments", fontsize=heading_fontsize)
    fig.supylabel("Best Yield (%)", fontsize=heading_fontsize)
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
    # Wandb API
    api = wandb.Api()

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
        # Get STBO data
        logger.info(f"Getting Reizman Suzuki case {i} STBO data")
        stbo_dfs = get_wandb_run_dfs(
            api,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            model_name=f"reizman_suzuki_case_{i}",
            strategy="STBO",
            include_tags=include_tags,
            filter_tags=filter_tags,
            only_finished_runs=only_finished_runs,
            num_iterations=num_iterations,
        )
        for j in range(1, 5):
            if i != j:
                # Get MTBO data
                logger.info(
                    f"Getting Reizman Suzuki case {i} (auxiliary Reizman suzuki case {j}) MTBO data"
                )
                mtbo_dfs = get_wandb_run_dfs(
                    api,
                    wandb_entity=wandb_entity,
                    wandb_project=wandb_project,
                    model_name=f"reizman_suzuki_case_{i}",
                    strategy="MTBO",
                    include_tags=include_tags,
                    filter_tags=filter_tags,
                    only_finished_runs=only_finished_runs,
                    num_iterations=num_iterations,
                    extra_filters={
                        "config.ct_dataset_names": [f"reizman_suzuki_case_{j}"],
                    },
                )
                logger.info(
                    f"Found {len(stbo_dfs)} STBO and {len(mtbo_dfs)} MTBO runs for Reizman Suzuki case {i} with auxiliary of Reizman Suzuki case {j}"
                )

                # Make subplot
                ax = fig.add_subplot(4, 3, k)
                make_comparison_plot(
                    dict(results=stbo_dfs, label="STBO", color="#a50026"),
                    dict(results=mtbo_dfs, label="MTBO", color="#313695"),
                    output_name="yld_best",
                    ax=ax,
                )
                ax.set_title(
                    f"{reizman_case_to_name[i]} (Aux. {reizman_case_to_name[j]})"
                )
                ax.set_ylim(0, 100)
                k += 1

    # Format plot
    fig.suptitle("Reizman Optimization")
    fig.supxlabel("Number of reactions")
    fig.supylabel("Best Yield (%)")
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    fig.savefig(figure_dir / "reizman_reizman_one_cotraining_optimization.png", dpi=300)


def all_suzuki(
    num_iterations: int = 20,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    wandb_entity: str = "ceb-sre",
    wandb_project: str = "multitask",
    figure_dir: str = "figures",
):
    """Make plots for all Suzuki optimization runs."""
    baumgartner_suzuki_auxiliary_reizman_suzuki(
        num_iterations=num_iterations,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
    reizman_suzuki_auxiliary_reizman_suzuki(
        num_iterations=num_iterations,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
    reizman_suzuki_auxiliary_baumgartner_suzuki(
        num_iterations=num_iterations,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
