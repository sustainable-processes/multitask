"""
Make figures for publication
"""
from .plots import (
    get_wandb_run_dfs,
    make_yld_comparison_plot,
    make_categorical_comparison_plot,
)
from summit import *
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


catalyst_map = {
    "cycloPd AlPhos 4-Chlorotoluene": "P3-L10",
    "cycloPd tBuBrettPhos 4-Chlorotoluene": "P3-L9",
    "cycloPd tBuXPhos 4-Chlorotoluene": "P3-L8",
}

bases = [
    "TEA",
    "TMG",
    "BTMG",
    "DBU",
    "MTBD",
    "BTTP",
    "P2Et",
]
categorical_map = {
    f"{cat_name} + {base}": f"{cat_ref} + {base}"
    for cat_name, cat_ref in catalyst_map.items()
    for base in bases
}


def combine_catalyst_bases(ds: DataSet):
    ds["categoricals", "DATA"] = ds["catalyst"] + " + " + ds["base"]
    return ds


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
            extra_filters={"config.ct_dataset_names": []},
            limit=num_repeats,
        )
        stbo_dfs = [combine_catalyst_bases(stbo_df) for stbo_df in stbo_dfs]

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
                    limit=num_repeats,
                )
                mtbo_dfs = [combine_catalyst_bases(mtbo_df) for mtbo_df in mtbo_dfs][
                    :num_repeats
                ]
                # stbo_head_start_dfs = get_wandb_run_dfs(
                #     api,
                #     wandb_entity=wandb_entity,
                #     wandb_project=wandb_project,
                #     model_name=f"baumgartner_cn_case_{i}",
                #     strategy="STBO",
                #     include_tags=include_tags,
                #     filter_tags=filter_tags,
                #     only_finished_runs=only_finished_runs,
                #     num_iterations=num_iterations,
                #     extra_filters={
                #         "config.ct_dataset_names": [f"baumgartner_cn_case_{j}"],
                #     },
                #     limit=num_repeats,
                # )
                # stbo_head_start_dfs = [
                #     combine_catalyst_bases(stbo_head_start_df)
                #     for stbo_head_start_df in stbo_head_start_dfs
                # ]
                logger.info(
                    f"Found {len(stbo_dfs)} STBO and {len(mtbo_dfs)} MTBO runs for Baumgartner C-N case {i} with auxiliary of Baumgartner C-N case {j}"
                )

                # Make yield subplot
                ax_yld = fig_yld.add_subplot(4, 3, k)
                make_yld_comparison_plot(
                    dict(results=stbo_dfs, label="STBO", color="#a50026"),
                    # dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
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

                # Make categorical subplot
                ax_cat_best = fig_cat_best.add_subplot(4, 3, k)
                make_categorical_comparison_plot(
                    dict(results=stbo_dfs, label="STBO", color="#a50026"),
                    # dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
                    dict(results=mtbo_dfs, label="MTBO", color="#313695"),
                    categorical_variable="categoricals",
                    output_name="yld",
                    categorical_map=categorical_map,
                    ax=ax_cat_best,
                    plot_type="best",
                )
                ax_cat_counts = fig_cat_counts.add_subplot(4, 3, k)
                make_categorical_comparison_plot(
                    dict(results=stbo_dfs, label="STBO", color="#a50026"),
                    # dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
                    dict(results=mtbo_dfs, label="MTBO", color="#313695"),
                    categorical_variable="categoricals",
                    output_name="yld",
                    categorical_map=categorical_map,
                    ax=ax_cat_counts,
                    plot_type="counts",
                )

                # Format subplot
                ax_cat_best.set_title(
                    f"C-N B{i} (Aux.C-N B{j})",
                    fontsize=heading_fontsize,
                )
                ax_cat_best.set_ylim(0.0, 1.0)
                ax_cat_best.tick_params("y", labelsize=12, direction="in")
                ax_cat_best.set_xticklabels(
                    ax_cat_best.get_xticklabels(), rotation=30, ha="right"
                )
                ax_cat_counts.set_title(
                    f"C-N B{i} (Aux.C-N B{j})",
                    fontsize=heading_fontsize,
                )
                ax_cat_counts.set_ylim(0.0, 1.0)
                ax_cat_counts.tick_params("y", labelsize=12, direction="in")
                ax_cat_counts.set_xticklabels(
                    ax_cat_counts.get_xticklabels(), rotation=30, ha="right"
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

    # Catalyst best
    fig_cat_best.supxlabel("Catalyst", y=-0.01, fontsize=heading_fontsize)
    fig_cat_best.supylabel(
        "Frequency was best catalyst", x=-0.01, fontsize=heading_fontsize
    )
    fig_cat_best.tight_layout()
    fig_cat_best.savefig(
        figure_dir
        / "baumgartner_cn_baumgartner_cn_one_cotraining_optimization_catalyst_best.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )

    # Catalyst counts
    fig_cat_counts.supxlabel("Catalyst", y=-0.01, fontsize=heading_fontsize)
    fig_cat_counts.supylabel("Frequency selected", x=-0.01, fontsize=heading_fontsize)
    fig_cat_counts.tight_layout()
    fig_cat_counts.savefig(
        figure_dir
        / "baumgartner_cn_baumgartner_cn_one_cotraining_optimization_catalyst_counts.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )


def baumgartner_cn_auxiliary_all_baumgartner_cn(
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
    fig_yld = plt.figure(figsize=(15, 5))
    fig_yld.subplots_adjust(wspace=0.2, hspace=0.6)
    fig_cat_best = plt.figure(figsize=(15, 5))
    fig_cat_best.subplots_adjust(wspace=0.2, hspace=0.6)
    fig_cat_counts = plt.figure(figsize=(15, 5))
    fig_cat_counts.subplots_adjust(wspace=0.2, hspace=0.6)
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
            extra_filters={"config.ct_dataset_names": []},
        )
        stbo_dfs = [combine_catalyst_bases(stbo_df) for stbo_df in stbo_dfs][
            :num_repeats
        ]
        # stbo_head_start_dfs = get_wandb_run_dfs(
        #     api,
        #     wandb_entity=wandb_entity,
        #     wandb_project=wandb_project,
        #     model_name=f"baumgartner_cn_case_{i}",
        #     strategy="STBO",
        #     include_tags=include_tags,
        #     filter_tags=filter_tags,
        #     only_finished_runs=only_finished_runs,
        #     num_iterations=num_iterations,
        #     extra_filters={
        #         "config.ct_dataset_names": [
        #             f"baumgartner_cn_case_{j}" for j in range(1, 5) if i != j
        #         ],
        #     },
        # )
        # stbo_head_start_dfs = [
        #     combine_catalyst_bases(stbo_head_start_df)
        #     for stbo_head_start_df in stbo_head_start_dfs
        # ][:num_repeats]
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
        mtbo_dfs = [combine_catalyst_bases(mtbo_df) for mtbo_df in mtbo_dfs][
            :num_repeats
        ]
        logger.info(
            f"Found {len(stbo_dfs)} STBO and {len(mtbo_dfs)} MTBO runs for Baumgartner C-N case {i} with auxiliary of all Baumgartner C-N"
        )

        # Make yield subplot
        ax = fig_yld.add_subplot(1, 4, i)
        make_yld_comparison_plot(
            dict(results=stbo_dfs, label="STBO", color="#a50026"),
            # dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
            dict(results=mtbo_dfs, label="MTBO", color="#313695"),
            output_name="yld_best",
            ax=ax,
        )

        # Format subplot
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

        # Make categorical subplot
        ax_cat_best = fig_cat_best.add_subplot(1, 4, i)
        make_categorical_comparison_plot(
            dict(results=stbo_dfs, label="STBO", color="#a50026"),
            # dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
            dict(results=mtbo_dfs, label="MTBO", color="#313695"),
            categorical_variable="categoricals",
            output_name="yld",
            categorical_map=categorical_map,
            ax=ax_cat_best,
            plot_type="best",
        )
        ax_cat_counts = fig_cat_counts.add_subplot(1, 4, i)
        make_categorical_comparison_plot(
            dict(results=stbo_dfs, label="STBO", color="#a50026"),
            # dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
            dict(results=mtbo_dfs, label="MTBO", color="#313695"),
            categorical_variable="categoricals",
            output_name="yld",
            categorical_map=categorical_map,
            ax=ax_cat_counts,
            plot_type="counts",
        )

        # Format categorical subplots
        ax_cat_best.set_title(
            f"C-N B{i} (Aux. C-N {aux_names})",
            fontsize=heading_fontsize,
        )
        ax_cat_best.set_ylim(0.0, 1.0)
        ax_cat_best.tick_params("y", labelsize=12, direction="in")
        ax_cat_best.set_xticklabels(
            ax_cat_best.get_xticklabels(), rotation=30, ha="right"
        )
        ax_cat_counts.set_title(
            f"C-N B{i} (Aux. C-N {aux_names})",
            fontsize=heading_fontsize,
        )
        ax_cat_counts.set_ylim(0.0, 1.0)
        ax_cat_counts.tick_params("y", labelsize=12, direction="in")
        ax_cat_counts.set_xticklabels(
            ax_cat_counts.get_xticklabels(), rotation=30, ha="right"
        )

    # Format plot
    fig_yld.supxlabel("Experiment Number", fontsize=heading_fontsize)
    fig_yld.supylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig_yld.tight_layout()
    figure_dir = Path(figure_dir)
    fig_yld.savefig(
        figure_dir / "baumgartner_cn_baumgartner_cn_all_cotraining_optimization.png",
        dpi=300,
    )

    # Catalyst best
    fig_cat_best.supxlabel("Catalyst", y=-0.01, fontsize=heading_fontsize)
    fig_cat_best.supylabel(
        "Frequency was best catalyst", x=-0.01, fontsize=heading_fontsize
    )
    fig_cat_best.tight_layout()
    fig_cat_best.savefig(
        figure_dir
        / "baumgartner_cn_baumgartner_cn_all_cotraining_optimization_catalyst_best.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )

    # Catalyst counts
    fig_cat_counts.supxlabel("Catalyst", y=-0.01, fontsize=heading_fontsize)
    fig_cat_counts.supylabel("Frequency selected", x=-0.01, fontsize=heading_fontsize)
    fig_cat_counts.tight_layout()
    fig_cat_counts.savefig(
        figure_dir
        / "baumgartner_cn_baumgartner_cn_all_cotraining_optimization_catalyst_counts.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )


def all_cn(
    num_iterations: int = 20,
    num_repeats: int = 20,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = "multitask",
    figure_dir: Optional[str] = "figures",
):
    baumgartner_cn_auxiliary_one_baumgartner_cn(
        num_iterations=num_iterations,
        num_repeats=num_repeats,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
    baumgartner_cn_auxiliary_all_baumgartner_cn(
        num_iterations=num_iterations,
        num_repeats=num_repeats,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
