"""
Make Suzuki cross coupling figures for publication
"""
from .plots import make_comparison_plot, get_wandb_run_dfs
from pathlib import Path
from summit import *
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import wandb
import logging


logger = logging.getLogger(__name__)


reizman_case_to_name = {1: "SR1", 2: "SR2", 3: "SR3", 4: "SR4"}


def baumgartner_suzuki_auxiliary_one_reizman_suzuki(
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
    axis_fontsize = 16
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
        ax.set_title(f"SB1 (Aux. {reizman_case_to_name[i]})", fontsize=heading_fontsize)
        ax.set_xlim(0, 20)
        ax.tick_params("y", labelsize=axis_fontsize)
        xlabels = np.arange(0, 21, 5)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(xlabels, fontsize=axis_fontsize)
        ax.set_ylim(0, 100)
        k += 1

    # Format and save figure
    # fig.suptitle("Baumgartner Optimization")
    fig.supxlabel("Experiment number", fontsize=heading_fontsize)
    fig.supylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    fig.savefig(
        figure_dir
        / "baumgartner_suzuki_reizman_suzuki_one_cotraining_optimization.png",
        dpi=300,
        transparent=True,
    )
    logger.info("Plots saved to %s", figure_dir)


def baumgartner_suzuki_auxiliary_all_reizman_suzuki(
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
    fig, ax = plt.subplots(1, figsize=(5, 5))
    axis_fontsize = 16
    heading_fontsize = 18
    logger.info(
        "Making plot for Baumgartner Suzuki optimization with auxiliary of Reizman Suzuki"
    )
    # Get MTBO data
    logger.info(f"Getting Baumgartner Suzuki MTBO (cotrain all Reizman Suzuki) data")
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
            "config.ct_dataset_names": [
                f"reizman_suzuki_case_{j}" for j in range(1, 5)
            ],
        },
    )

    # Make comparison subplot
    make_comparison_plot(
        dict(results=stbo_dfs, label="STBO", color="#a50026"),
        dict(results=mtbo_dfs, label="MTBO", color="#313695"),
        output_name="yld_best",
        ax=ax,
    )

    # Format subplot
    ax.set_xlim(0, 20)
    ax.tick_params("y", labelsize=axis_fontsize)
    xlabels = np.arange(0, 21, 5)
    ax.set_xticks(xlabels)
    ax.set_xticklabels(xlabels, fontsize=axis_fontsize)
    ax.set_ylim(0, 100)
    k += 1

    # Format and save figure
    # fig.suptitle("Baumgartner Optimization")
    ax.set_xlabel("Experiment number", fontsize=heading_fontsize)
    ax.set_ylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    fig.savefig(
        figure_dir
        / "baumgartner_suzuki_reizman_suzuki_all_cotraining_optimization.png",
        dpi=300,
        transparent=True,
    )
    logger.info("Plots saved to %s", figure_dir)


def reizman_suzuki_auxiliary_one_baumgartner_suzuki(
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
    fig.supxlabel("Experiment number", fontsize=heading_fontsize)
    fig.supylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    fig.savefig(
        figure_dir
        / "reizman_suzuki_baumgartner_suzuki_one_cotraining_optimization.png",
        dpi=300,
    )
    logger.info("Plots saved to %s", figure_dir)


def reizman_suzuki_auxiliary_one_reizman_suzuki(
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
    heading_fontsize = 18

    # Make subplots for each case
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

                # # Make subplot
                ax = fig.add_subplot(4, 3, k)
                make_comparison_plot(
                    dict(results=stbo_dfs, label="STBO", color="#a50026"),
                    dict(results=mtbo_dfs, label="MTBO", color="#313695"),
                    output_name="yld_best",
                    ax=ax,
                )
                ax.set_title(
                    f"{reizman_case_to_name[i]} (Aux. {reizman_case_to_name[j]})",
                    fontsize=heading_fontsize,
                )
                ax.set_xlim(0, 20)
                ax.tick_params("y", labelsize=axis_fontsize)
                xlabels = np.arange(0, 21, 5)
                ax.set_xticks(xlabels)
                ax.set_xticklabels(xlabels, fontsize=axis_fontsize)
                ax.set_ylim(0, 100)
                k += 1

    # Format plot
    # fig.suptitle("Reizman Optimization")
    fig.supxlabel("Experiment number", fontsize=heading_fontsize)
    fig.supylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    fig.savefig(
        figure_dir / "reizman_suzuki_reizman_suzuki_one_cotraining_optimization.png",
        dpi=300,
    )
    logger.info("Plots saved to %s", figure_dir)


def reizman_suzuki_auxiliary_all_baumgartner_suzuki(
    num_iterations: int = 20,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    wandb_entity: str = "ceb-sre",
    wandb_project: str = "multitask",
    figure_dir: str = "figures",
):
    """Make plots for Reizman Suzuki optimization with auxiliary of all Reizman Suzuki"""
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
        "Making plots for Reizman Suzuki optimization with auxiliary of all REizman Suzuki"
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
            f"Getting Reizman Suzuki case {i} (auxiliary all Reizman suzuki) MTBO data"
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
                "config.ct_dataset_names": [
                    f"reizman_suzuki_case_{j}" for j in range(1, 5) if j != i
                ],
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
        aux_names = ",".join(
            [f"{reizman_case_to_name[j]}" for j in range(1, 5) if j != i]
        )
        ax.set_title(
            f"{reizman_case_to_name[i]} (Aux. {aux_names})", fontsize=heading_fontsize
        )
        xlabels = np.arange(0, 21, 5)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(xlabels, fontsize=axis_fontsize)
        ax.set_yticks([0, 20, 40, 60, 80, 100, 120])
        ax.set_yticklabels([0, 20, 40, 60, 80, 100, ""], fontsize=axis_fontsize)
        ax.tick_params(direction="in")
        k += 1

    # Format and save figure
    # fig.suptitle("Reizman Optimization", fontsize=21)
    fig.supxlabel("Experiment number", fontsize=heading_fontsize)
    fig.supylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    fig.savefig(
        figure_dir / "reizman_suzuki_reizman_suzuki_all_cotraining_optimization.png",
        dpi=300,
    )
    logger.info("Plots saved to %s", figure_dir)


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
    baumgartner_suzuki_auxiliary_one_reizman_suzuki(
        num_iterations=num_iterations,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
    baumgartner_suzuki_auxiliary_all_reizman_suzuki(
        num_iterations=num_iterations,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
    reizman_suzuki_auxiliary_one_reizman_suzuki(
        num_iterations=num_iterations,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
    reizman_suzuki_auxiliary_one_baumgartner_suzuki(
        num_iterations=num_iterations,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
    reizman_suzuki_auxiliary_all_baumgartner_suzuki(
        num_iterations=num_iterations,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
