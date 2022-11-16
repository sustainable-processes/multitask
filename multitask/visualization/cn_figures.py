"""
Make figures for publication
"""
from .plots import (
    get_wandb_run_dfs,
    make_yld_comparison_plot,
    make_categorical_comparison_plot,
)
from multitask.etl.etl_baumgartner_cn import ligands, bases
from summit import *
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
import logging


logger = logging.getLogger(__name__)
categorical_map = {}


def combine_catalyst_bases(ds: DataSet):
    ds["categoricals"] = ds["base"]


def baumgartner_cn_auxiliary_one_baumgartner_cn(
    num_iterations: int = 20,
    num_repeats: int = 20,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = "multitask",
    figure_dir: Optional[str] = "figures",
):
    """Make plots for Baumgartner C-N optimization with auxiliary of Baumgartner C-N."""
    # Setup wandb api
    api = wandb.Api()

    # Setup figure
    fig_yld = plt.figure(figsize=(15, 15))
    fig_yld.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_cat_best = plt.figure(figsize=(15, 15))
    fig_cat_best.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_cat_counts = plt.figure(figsize=(15, 15))
    fig_cat_counts.subplots_adjust(wspace=0.2, hspace=0.5)
    k = 1
    axis_fontsize = 14
    heading_fontsize = 18

    # Make subplots for each case
    logger.info(
        "Making plots for Baumgartner C-N optimization with auxiliary of Baumgartner C-N"
    )
    for i in range(1, 5):
        # Filter STBO data
        stbo_dfs = get_wandb_run_dfs(
            api,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            model_name=f"baumgartner_cn_case_{i}",
            strategy="STBO",
            include_tags=include_tags,
            filter_tags=filter_tags,
            only_finished_runs=only_finished_runs,
            num_iterations=num_iterations,
        )
        stbo_dfs = stbo_dfs[:num_repeats]

        for j in range(1, 5):
            if i != j:
                # Filter MTBO data
                mtbo_dfs = get_wandb_run_dfs(
                    api,
                    wandb_entity=wandb_entity,
                    wandb_project=wandb_project,
                    model_name=f"baumgartner_cn_case_{i}",
                    strategy="MTBO",
                    include_tags=include_tags,
                    filter_tags=filter_tags,
                    only_finished_runs=only_finished_runs,
                    num_iterations=num_iterations,
                    extra_filters={
                        "config.ct_dataset_names": [f"baumgartner_cn_case_{j}"],
                    },
                )
                mtbo_dfs = mtbo_dfs[:num_repeats]
                stbo_head_start_dfs = get_wandb_run_dfs(
                    api,
                    wandb_entity=wandb_entity,
                    wandb_project=wandb_project,
                    model_name=f"baumgartner_cn_case_{i}",
                    strategy="STBO",
                    include_tags=include_tags,
                    filter_tags=filter_tags,
                    only_finished_runs=only_finished_runs,
                    num_iterations=num_iterations,
                    extra_filters={
                        "config.ct_dataset_names": [f"baumgartner_cn_case_{j}"],
                    },
                )
                stbo_head_start_dfs = stbo_head_start_dfs[:num_repeats]
                logger.info(
                    f"Found {len(stbo_dfs)} STBO and {len(mtbo_dfs)} MTBO runs for Baumgartner C-N case {i} with auxiliary of Baumgartner C-N case {j}"
                )

                # Make subplot
                ax_yld = fig_yld.add_subplot(4, 3, k)
                make_yld_comparison_plot(
                    dict(results=stbo_dfs, label="STBO", color="#a50026"),
                    dict(results=mtbo_dfs, label="MTBO", color="#313695"),
                    output_name="yld_best",
                    ax=ax_yld,
                )

                # Format subplot
                ax_yld.tick_params("both", labelsize=axis_fontsize)
                ax_yld.set_title(
                    f"C-N B{i} (Aux.C-N B{j})",
                    fontsize=heading_fontsize,
                )
                xlabels = np.arange(0, 21, 5)
                ax_yld.set_xticks(xlabels)
                ax_yld.set_xticklabels(xlabels, fontsize=axis_fontsize)
                ax_yld.set_yticks([0, 20, 40, 60, 80, 100, 120])
                ax_yld.set_yticklabels(
                    [0, 20, 40, 60, 80, 100, ""], fontsize=axis_fontsize
                )
                ax_yld.tick_params(direction="in")

                ax_cat_best = fig_cat_best.add_subplot(4, 3, k)
                make_categorical_comparison_plot(
                    dict(results=stbo_dfs, label="STBO", color="#a50026"),
                    dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
                    dict(results=mtbo_dfs, label="MTBO", color="#313695"),
                    categorical_variable="catalyst_smiles",
                    output_name="yld",
                    categorical_map=catalyst_map,
                    ax=ax_cat_best,
                    plot_type="best",
                )
                ax_cat_counts = fig_cat_counts.add_subplot(4, 3, k)
                make_categorical_comparison_plot(
                    dict(results=stbo_dfs, label="STBO", color="#a50026"),
                    dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
                    dict(results=mtbo_dfs, label="MTBO", color="#313695"),
                    categorical_variable="catalyst_smiles",
                    output_name="yld",
                    categorical_map=catalyst_map,
                    ax=ax_cat_counts,
                    plot_type="counts",
                )
                k += 1

    # Format plot
    # fig.suptitle("Baumgarnter Optimization", fontsize=heading_fontsize)
    fig_yld.supxlabel("Experiment Number", fontsize=heading_fontsize)
    fig_yld.supylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig_yld.tight_layout()
    figure_dir = Path(figure_dir)
    fig_yld.savefig(
        figure_dir / "baumgartner_cn_baumgartner_cn_one_cotraining_optimization.png",
        dpi=300,
    )


def baumgartner_cn_auxiliary_all_baumgartner_cn(
    num_iterations: int = 20,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = "multitask",
    figure_dir: Optional[str] = "figures",
):
    """Make plots for Baumgartner C-N optimization with auxiliary of Baumgartner C-N."""
    # Setup wandb api
    api = wandb.Api()

    # Setup figure
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.2, hspace=0.5)
    axis_fontsize = 14
    heading_fontsize = 18

    # Make subplots for each case
    logger.info(
        "Making plots for Baumgartner C-N optimization with auxiliary of Baumgartner C-N"
    )
    for i in range(1, 5):
        # Filter STBO data
        stbo_dfs = get_wandb_run_dfs(
            api,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            model_name=f"baumgartner_cn_case_{i}",
            strategy="STBO",
            include_tags=include_tags,
            filter_tags=filter_tags,
            only_finished_runs=only_finished_runs,
            num_iterations=num_iterations,
        )

        mtbo_dfs = get_wandb_run_dfs(
            api,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            model_name=f"baumgartner_cn_case_{i}",
            strategy="MTBO",
            include_tags=include_tags,
            filter_tags=filter_tags,
            only_finished_runs=only_finished_runs,
            num_iterations=num_iterations,
            extra_filters={
                "config.ct_dataset_names": [
                    f"baumgartner_cn_case_{j}" for j in range(1, 5) if i != j
                ],
            },
        )
        logger.info(
            f"Found {len(stbo_dfs)} STBO and {len(mtbo_dfs)} MTBO runs for Baumgartner C-N case {i} with auxiliary of all Baumgartner C-N"
        )

        # # Make subplot
        ax = fig.add_subplot(1, 4, i)
        make_yld_comparison_plot(
            dict(results=stbo_dfs, label="STBO", color="#a50026"),
            dict(results=mtbo_dfs, label="MTBO", color="#313695"),
            output_name="yld_best",
            ax=ax,
        )
        aux_names = ",".join([f"B{j}" for j in range(1, 5) if i != j])
        ax.set_title(
            f"C-N B{i} (Aux. C-N {aux_names})",
            fontsize=heading_fontsize,
        )
        xlabels = np.arange(0, 21, 5)
        ax.set_xticks(xlabels)
        ax.set_xticklabels(xlabels, fontsize=axis_fontsize)
        ax.set_yticks([0, 20, 40, 60, 80, 100, 120])
        ax.set_yticklabels([0, 20, 40, 60, 80, 100, ""], fontsize=axis_fontsize)
        ax.tick_params(direction="in")

    # Format plot
    # fig.suptitle("Baumgarnter Optimization", fontsize=heading_fontsize)
    fig.supxlabel("Experiment Number", fontsize=heading_fontsize)
    fig.supylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    fig.savefig(
        figure_dir / "baumgartner_cn_baumgartner_cn_all_cotraining_optimization.png",
        dpi=300,
    )


def all_cn(
    num_iterations: int = 20,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = "multitask",
    figure_dir: Optional[str] = "figures",
):
    baumgartner_cn_auxiliary_one_baumgartner_cn(
        num_iterations=num_iterations,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
    baumgartner_cn_auxiliary_all_baumgartner_cn(
        num_iterations=num_iterations,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
