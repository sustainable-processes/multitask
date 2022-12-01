"""
Make Suzuki cross coupling figures for publication
"""
from multitask.etl.etl_baumgartner_suzuki import ligands, pre_catalysts
from .plots import (
    make_yld_comparison_plot,
    make_categorical_comparison_plot,
    get_wandb_run_dfs,
)
from pathlib import Path
from summit import *
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import wandb
import logging
import os


logger = logging.getLogger(__name__)


reizman_case_to_name = {1: "SR1", 2: "SR2", 3: "SR3", 4: "SR4"}
catalyst_map = {
    Chem.CanonSmiles(
        f"""{pre_cat["SMILES"]}.{ligand["SMILES"]}"""
    ): f"{pre_cat_ref}-{ligand_ref}"
    for ligand_ref, ligand in ligands.items()
    for pre_cat_ref, pre_cat in pre_catalysts.items()
}


def standardize_catalyst_smiles(ds: DataSet):
    ds["catalyst_smiles"] = ds["catalyst_smiles"].apply(Chem.CanonSmiles)
    # catalyst_df = ds["catalyst"].str.split(".", expand=True, regex=False)
    # catalyst_df.columns = ["pre_cat", "ligand"]
    # catalyst_df["pre_cat"] = catalyst_df["pre_cat"].apply(Chem.CanonSmiles)
    # catalyst_df["ligand"] = catalyst_df["ligand"].apply(Chem.CanonSmiles)
    # ds["catalyst"] = catalyst_df["pre_cat"] + "." + catalyst_df["ligand"]
    return ds


def baumgartner_suzuki_auxiliary_one_reizman_suzuki(
    num_iterations: int = 20,
    num_repeats: int = 20,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    wandb_entity: str = "ceb-sre",
    wandb_project: str = "multitask",
    figure_dir: str = "figures",
):
    """Make plots for Baumgartner Suzuki optimization with auxiliary of Reizman Suzuki."""
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
        # num_iterations=num_iterations,
        extra_filters={"config.ct_dataset_names": []},
    )
    stbo_dfs = [standardize_catalyst_smiles(stbo_df) for stbo_df in stbo_dfs][
        :num_repeats
    ]

    # Setup figure
    fig_yld = plt.figure(figsize=(15, 5))
    fig_yld.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_cat_best = plt.figure(figsize=(15, 5))
    fig_cat_best.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_cat_counts = plt.figure(figsize=(15, 5))
    fig_cat_counts.subplots_adjust(wspace=0.2, hspace=0.5)
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
        mtbo_dfs = [standardize_catalyst_smiles(mtbo_df) for mtbo_df in mtbo_dfs][
            :num_repeats
        ]
        stbo_head_start_dfs = get_wandb_run_dfs(
            api,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            model_name="baumgartner_suzuki",
            strategy="STBO",
            include_tags=include_tags,
            filter_tags=filter_tags,
            only_finished_runs=only_finished_runs,
            num_iterations=num_iterations,
            extra_filters={
                "config.ct_dataset_names": [f"reizman_suzuki_case_{i}"],
            },
        )
        stbo_head_start_dfs = [
            standardize_catalyst_smiles(stbo_df) for stbo_df in stbo_head_start_dfs
        ][:num_repeats]

        # Make yield comparison subplot
        ax_yld = fig_yld.add_subplot(1, 4, k)
        make_yld_comparison_plot(
            dict(results=stbo_dfs, label="STBO", color="#a50026"),
            dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
            dict(results=mtbo_dfs, label="MTBO", color="#313695"),
            output_name="yld_best",
            ax=ax_yld,
        )

        # Format subplot
        ax_yld.set_title(
            f"SB1 (Aux. {reizman_case_to_name[i]})", fontsize=heading_fontsize
        )
        ax_yld.set_xlim(0, 20)
        ax_yld.tick_params("y", labelsize=axis_fontsize)
        xlabels = np.arange(0, 21, 5)
        ax_yld.set_xticks(xlabels)
        ax_yld.set_xticklabels(xlabels, fontsize=axis_fontsize)
        ax_yld.set_ylim(0, 100)

        # Make catalyst comparison plot
        ax_cat_best = fig_cat_best.add_subplot(1, 4, k)
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
        ax_cat_counts = fig_cat_counts.add_subplot(1, 4, k)
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

        # Format subplot
        ax_cat_best.set_title(
            f"SB1 (Aux. {reizman_case_to_name[i]})", fontsize=heading_fontsize
        )
        ax_cat_best.set_ylim(0.0, 1.0)
        ax_cat_best.tick_params("y", labelsize=12, direction="in")
        ax_cat_counts.set_title(
            f"SB1 (Aux. {reizman_case_to_name[i]})", fontsize=heading_fontsize
        )
        ax_cat_counts.set_ylim(0.0, 1.0)
        ax_cat_counts.tick_params("y", labelsize=12, direction="in")
        k += 1

    # Format and save figure
    figure_dir = Path(figure_dir)
    fig_yld.supxlabel("Experiment number", fontsize=heading_fontsize)
    fig_yld.supylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig_yld.tight_layout()
    fig_yld.savefig(
        figure_dir
        / "baumgartner_suzuki_reizman_suzuki_one_cotraining_optimization.png",
        dpi=300,
        transparent=True,
    )

    # Catalyst best
    fig_cat_best.supxlabel("Catalyst", y=-0.01, fontsize=heading_fontsize)
    fig_cat_best.supylabel(
        "Frequency was best catalyst", x=-0.01, fontsize=heading_fontsize
    )
    fig_cat_best.tight_layout()
    fig_cat_best.savefig(
        figure_dir
        / "baumgartner_suzuki_reizman_suzuki_one_cotraining_optimization_catalyst_best.png",
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
        / "baumgartner_suzuki_reizman_suzuki_one_cotraining_optimization_catalyst_counts.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    logger.info("Plots saved to %s", figure_dir)


def baumgartner_suzuki_auxiliary_all_reizman_suzuki(
    num_iterations: Optional[int] = 20,
    num_repeats: Optional[int] = 20,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: Optional[bool] = True,
    wandb_entity: Optional[str] = "ceb-sre",
    wandb_project: Optional[str] = "multitask",
    figure_dir: Optional[str] = "figures",
):
    """Make plots for Baumgartner Suzuki optimization with auxiliary of Reizman Suzuki."""
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
        extra_filters={"config.ct_dataset_names": []},
    )
    stbo_dfs = [standardize_catalyst_smiles(stbo_df) for stbo_df in stbo_dfs][
        :num_repeats
    ]

    # Setup figure
    fig_yld, ax_yld = plt.subplots(1, figsize=(5, 5))
    fig_cat_best, ax_cat_best = plt.subplots(1, figsize=(5, 5))
    fig_cat_counts, ax_cat_counts = plt.subplots(1, figsize=(5, 5))
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
    mtbo_dfs = [standardize_catalyst_smiles(mtbo_df) for mtbo_df in mtbo_dfs][
        :num_repeats
    ]
    stbo_head_start_dfs = get_wandb_run_dfs(
        api,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        model_name="baumgartner_suzuki",
        strategy="STBO",
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        # num_iterations=num_iterations,
        extra_filters={
            "config.ct_dataset_names": [
                f"reizman_suzuki_case_{j}" for j in range(1, 5)
            ],
        },
    )
    stbo_head_start_dfs = [
        standardize_catalyst_smiles(stbo_df) for stbo_df in stbo_head_start_dfs
    ][:num_repeats]

    # Make comparison subplot
    make_yld_comparison_plot(
        dict(results=stbo_dfs, label="STBO", color="#a50026"),
        dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
        dict(results=mtbo_dfs, label="MTBO", color="#313695"),
        output_name="yld_best",
        ax=ax_yld,
    )

    # Format subplot
    ax_yld.set_xlim(0, 20)
    ax_yld.tick_params("y", labelsize=axis_fontsize)
    xlabels = np.arange(0, 21, 5)
    ax_yld.set_xticks(xlabels)
    ax_yld.set_xticklabels(xlabels, fontsize=axis_fontsize)
    ax_yld.set_ylim(0, 100)

    # Categorical plots
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

    # Format subplot
    ax_cat_best.set_ylim(0.0, 1.0)
    ax_cat_best.tick_params("y", labelsize=12, direction="in")
    ax_cat_counts.set_ylim(0.0, 1.0)
    ax_cat_counts.tick_params("y", labelsize=12, direction="in")

    # Format and save figure
    ax_yld.set_xlabel("Experiment number", fontsize=heading_fontsize)
    ax_yld.set_ylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig_yld.tight_layout()
    figure_dir = Path(figure_dir)
    fig_yld.savefig(
        figure_dir
        / "baumgartner_suzuki_reizman_suzuki_all_cotraining_optimization.png",
        dpi=300,
        transparent=True,
    )

    # Catalyst best
    fig_cat_best.supxlabel("Catalyst", y=-0.01, fontsize=heading_fontsize)
    fig_cat_best.supylabel(
        "Frequency was best catalyst", x=-0.01, fontsize=heading_fontsize
    )
    fig_cat_best.tight_layout()
    fig_cat_best.savefig(
        figure_dir
        / "baumgartner_suzuki_reizman_suzuki_all_cotraining_optimization_catalyst_best.png",
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
        / "baumgartner_suzuki_reizman_suzuki_all_cotraining_optimization_catalyst_counts.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    logger.info("Plots saved to %s", figure_dir)


def reizman_suzuki_auxiliary_one_baumgartner_suzuki(
    num_iterations: int = 20,
    num_repeats: int = 20,
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
    fig_yld = plt.figure(figsize=(15, 5))
    fig_yld.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_cat_best = plt.figure(figsize=(15, 5))
    fig_cat_best.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_cat_counts = plt.figure(figsize=(15, 5))
    fig_cat_counts.subplots_adjust(wspace=0.2, hspace=0.5)
    k = 1
    axis_fontsize = 14
    heading_fontsize = 18

    # Make subplots for each case
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
            extra_filters={"config.ct_dataset_names": []},
        )
        stbo_dfs = [standardize_catalyst_smiles(stbo_df) for stbo_df in stbo_dfs][
            :num_repeats
        ]

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
        mtbo_dfs = [standardize_catalyst_smiles(mtbo_df) for mtbo_df in mtbo_dfs][
            :num_repeats
        ]
        stbo_head_start_dfs = get_wandb_run_dfs(
            api,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            model_name=f"reizman_suzuki_case_{i}",
            strategy="STBO",
            include_tags=include_tags,
            filter_tags=filter_tags,
            only_finished_runs=only_finished_runs,
            num_iterations=num_iterations,
            extra_filters={
                "config.ct_dataset_names": [f"baumgartner_suzuki"],
            },
        )
        stbo_head_start_dfs = [
            standardize_catalyst_smiles(stbo_df) for stbo_df in stbo_head_start_dfs
        ][:num_repeats]

        logger.info(
            f"Found {len(stbo_dfs)} STBO and {len(mtbo_dfs)} MTBO runs for Reizman Suzuki case {i} with auxiliary of Baumgarnter Suzuki",
        )

        # Make subplot
        ax_yld = fig_yld.add_subplot(1, 4, k)
        make_yld_comparison_plot(
            dict(results=stbo_dfs, label="STBO", color="#a50026"),
            dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
            dict(results=mtbo_dfs, label="MTBO", color="#313695"),
            output_name="yld_best",
            ax=ax_yld,
        )

        # Format subplot
        ax_yld.set_title(
            f"{reizman_case_to_name[i]} (Aux. SB1)", fontsize=heading_fontsize
        )
        ax_yld.set_xlim(0, 20)
        ax_yld.tick_params("y", labelsize=axis_fontsize)
        xlabels = np.arange(0, 21, 5)
        ax_yld.set_xticks(xlabels)
        ax_yld.set_xticklabels(xlabels, fontsize=axis_fontsize)
        ax_yld.set_ylim(0, 100)

        # Make catalyst comparison plot
        ax_cat_best = fig_cat_best.add_subplot(1, 4, k)
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
        ax_cat_counts = fig_cat_counts.add_subplot(1, 4, k)
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

        # Format subplot
        ax_cat_best.set_title(
            f"{reizman_case_to_name[i]} (Aux. SB1)", fontsize=heading_fontsize
        )
        ax_cat_best.set_ylim(0.0, 1.0)
        ax_cat_best.tick_params("y", labelsize=12, direction="in")
        ax_cat_counts.set_title(
            f"{reizman_case_to_name[i]} (Aux. SB1)", fontsize=heading_fontsize
        )
        ax_cat_counts.set_ylim(0.0, 1.0)
        ax_cat_counts.tick_params("y", labelsize=12, direction="in")
        k += 1

    # Format and save figure
    # fig.suptitle("Reizman Optimization", fontsize=21)
    fig_yld.supxlabel("Experiment number", fontsize=heading_fontsize)
    fig_yld.supylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig_yld.tight_layout()
    figure_dir = Path(figure_dir)
    fig_yld.savefig(
        figure_dir
        / "reizman_suzuki_baumgartner_suzuki_one_cotraining_optimization.png",
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
        / "reizman_suzuki_baumgartner_suzuki_one_cotraining_optimization_catalyst_best.png",
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
        / "reizman_suzuki_baumgartner_suzuki_one_cotraining_optimization_catalyst_counts.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    logger.info("Plots saved to %s", figure_dir)


def reizman_suzuki_auxiliary_one_reizman_suzuki(
    num_iterations: int = 20,
    num_repeats: int = 20,
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
            extra_filters={"config.ct_dataset_names": []},
        )
        stbo_dfs = [standardize_catalyst_smiles(stbo_df) for stbo_df in stbo_dfs][
            :num_repeats
        ]
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
                mtbo_dfs = [
                    standardize_catalyst_smiles(mtbo_df) for mtbo_df in mtbo_dfs
                ][:num_repeats]
                stbo_head_start_dfs = get_wandb_run_dfs(
                    api,
                    wandb_entity=wandb_entity,
                    wandb_project=wandb_project,
                    model_name=f"reizman_suzuki_case_{i}",
                    strategy="STBO",
                    include_tags=include_tags,
                    filter_tags=filter_tags,
                    only_finished_runs=only_finished_runs,
                    num_iterations=num_iterations,
                    extra_filters={
                        "config.ct_dataset_names": [f"reizman_suzuki_case_{j}"],
                    },
                )
                stbo_head_start_dfs = [
                    standardize_catalyst_smiles(stbo_df)
                    for stbo_df in stbo_head_start_dfs
                ][:num_repeats]
                logger.info(
                    f"Found {len(stbo_dfs)} STBO and {len(mtbo_dfs)} MTBO runs for Reizman Suzuki case {i} with auxiliary of Reizman Suzuki case {j}"
                )

                # Make subplot
                ax_yld = fig_yld.add_subplot(4, 3, k)
                make_yld_comparison_plot(
                    dict(results=stbo_dfs, label="STBO", color="#a50026"),
                    dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
                    dict(results=mtbo_dfs, label="MTBO", color="#313695"),
                    output_name="yld_best",
                    ax=ax_yld,
                )

                # Format plot
                ax_yld.set_title(
                    f"{reizman_case_to_name[i]} (Aux. {reizman_case_to_name[j]})",
                    fontsize=heading_fontsize,
                )
                ax_yld.set_xlim(0, 20)
                ax_yld.tick_params("y", labelsize=axis_fontsize)
                xlabels = np.arange(0, 21, 5)
                ax_yld.set_xticks(xlabels)
                ax_yld.set_xticklabels(xlabels, fontsize=axis_fontsize)
                ax_yld.set_ylim(0, 100)

                # Make catalyst comparison plot
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

                # Format subplot
                ax_cat_best.set_title(
                    f"{reizman_case_to_name[i]} (Aux. {reizman_case_to_name[j]})",
                    fontsize=heading_fontsize,
                )
                ax_cat_best.set_ylim(0.0, 1.0)
                ax_cat_best.tick_params("y", labelsize=12, direction="in")
                ax_cat_counts.set_title(
                    f"{reizman_case_to_name[i]} (Aux. {reizman_case_to_name[j]})",
                    fontsize=heading_fontsize,
                )
                ax_cat_counts.set_ylim(0.0, 1.0)
                ax_cat_counts.tick_params("y", labelsize=12, direction="in")
                k += 1

    # Format plot
    # fig.suptitle("Reizman Optimization")
    fig_yld.supxlabel("Experiment number", fontsize=heading_fontsize)
    fig_yld.supylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig_yld.tight_layout()
    figure_dir = Path(figure_dir)
    fig_yld.savefig(
        figure_dir / "reizman_suzuki_reizman_suzuki_one_cotraining_optimization.png",
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
        / "reizman_suzuki_reizman_suzuki_one_cotraining_optimization_catalyst_best.png",
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
        / "reizman_suzuki_reizman_suzuki_one_cotraining_optimization_catalyst_counts.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    logger.info("Plots saved to %s", figure_dir)


def reizman_suzuki_auxiliary_all_baumgartner_suzuki(
    num_iterations: int = 20,
    num_repeats: int = 20,
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
    fig_yld = plt.figure(figsize=(15, 5))
    fig_yld.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_cat_best = plt.figure(figsize=(15, 5))
    fig_cat_best.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_cat_counts = plt.figure(figsize=(15, 5))
    fig_cat_counts.subplots_adjust(wspace=0.2, hspace=0.5)
    k = 1
    axis_fontsize = 14
    heading_fontsize = 18

    # Make subplots for each case
    # letters = ["a", "b", "c", "d"]
    logger.info(
        "Making plots for Reizman Suzuki optimization with auxiliary of all Reizman Suzuki"
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
            extra_filters={"config.ct_dataset_names": []},
        )
        stbo_dfs = [standardize_catalyst_smiles(stbo_df) for stbo_df in stbo_dfs][
            :num_repeats
        ]

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
                "config.ct_dataset_names": [f"baumgartner_suzuki"],
            },
        )
        mtbo_dfs = [standardize_catalyst_smiles(mtbo_df) for mtbo_df in mtbo_dfs][
            :num_repeats
        ]
        stbo_head_start_dfs = get_wandb_run_dfs(
            api,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            model_name=f"reizman_suzuki_case_{i}",
            strategy="STBO",
            include_tags=include_tags,
            filter_tags=filter_tags,
            only_finished_runs=only_finished_runs,
            num_iterations=num_iterations,
            extra_filters={"config.ct_dataset_names": [f"baumgartner_suzuki"]},
        )
        stbo_head_start_dfs = [
            standardize_catalyst_smiles(stbo_df) for stbo_df in stbo_head_start_dfs
        ][:num_repeats]

        logger.info(
            f"Found {len(stbo_dfs)} STBO and {len(mtbo_dfs)} MTBO runs for Reizman Suzuki case {i} with auxiliary of Baumgarnter Suzuki",
        )

        # Make subplot
        ax_yld = fig_yld.add_subplot(1, 4, k)
        make_yld_comparison_plot(
            dict(results=stbo_dfs, label="STBO", color="#a50026"),
            dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
            dict(results=mtbo_dfs, label="MTBO", color="#313695"),
            output_name="yld_best",
            ax=ax_yld,
        )

        # Format subplot
        ax_yld.set_title(
            f"{reizman_case_to_name[i]} (Aux. SB1)", fontsize=heading_fontsize
        )
        xlabels = np.arange(0, 21, 5)
        ax_yld.set_xticks(xlabels)
        ax_yld.set_xticklabels(xlabels, fontsize=axis_fontsize)
        ax_yld.set_yticks([0, 20, 40, 60, 80, 100, 120])
        ax_yld.set_yticklabels([0, 20, 40, 60, 80, 100, ""], fontsize=axis_fontsize)
        ax_yld.tick_params(direction="in")

        # Make catalyst comparison plot
        ax_cat_best = fig_cat_best.add_subplot(1, 4, k)
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
        ax_cat_counts = fig_cat_counts.add_subplot(1, 4, k)
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

        # Format subplot
        ax_cat_best.set_title(
            f"{reizman_case_to_name[i]} (Aux. SB1)", fontsize=heading_fontsize
        )
        ax_cat_best.set_ylim(0.0, 1.0)
        ax_cat_best.tick_params("y", labelsize=12, direction="in")
        ax_cat_counts.set_title(
            f"{reizman_case_to_name[i]} (Aux. SB1)", fontsize=heading_fontsize
        )
        ax_cat_counts.set_ylim(0.0, 1.0)
        ax_cat_counts.tick_params("y", labelsize=12, direction="in")
        k += 1

    # Format and save figure
    # fig.suptitle("Reizman Optimization", fontsize=21)
    fig_yld.supxlabel("Experiment number", fontsize=heading_fontsize)
    fig_yld.supylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig_yld.tight_layout()
    figure_dir = Path(figure_dir)
    fig_yld.savefig(
        figure_dir
        / "reizman_suzuki_baumgartner_suzuki_all_cotraining_optimization.png",
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
        / "reizman_suzuki_baumgartner_suzuki_all_cotraining_optimization_catalyst_best.png",
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
        / "reizman_suzuki_baumgartner_suzuki_all_cotraining_optimization_catalyst_counts.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    logger.info("Plots saved to %s", figure_dir)


def reizman_suzuki_auxiliary_all_reizman_suzuki(
    num_iterations: int = 20,
    num_repeats: int = 20,
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
    fig_yld = plt.figure(figsize=(15, 5))
    fig_yld.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_cat_best = plt.figure(figsize=(15, 5))
    fig_cat_best.subplots_adjust(wspace=0.2, hspace=0.5)
    fig_cat_counts = plt.figure(figsize=(15, 5))
    fig_cat_counts.subplots_adjust(wspace=0.2, hspace=0.5)
    k = 1
    axis_fontsize = 14
    heading_fontsize = 18

    # Make subplots for each case
    logger.info(
        "Making plots for Reizman Suzuki optimization with auxiliary of all Reizman Suzuki"
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
            extra_filters={"config.ct_dataset_names": []},
        )
        stbo_dfs = [standardize_catalyst_smiles(stbo_df) for stbo_df in stbo_dfs][
            :num_repeats
        ]

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
        mtbo_dfs = [standardize_catalyst_smiles(mtbo_df) for mtbo_df in mtbo_dfs][
            :num_repeats
        ]
        stbo_head_start_dfs = get_wandb_run_dfs(
            api,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            model_name=f"reizman_suzuki_case_{i}",
            strategy="STBO",
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
        stbo_head_start_dfs = [
            standardize_catalyst_smiles(stbo_df) for stbo_df in stbo_head_start_dfs
        ][:num_repeats]

        logger.info(
            f"Found {len(stbo_dfs)} STBO and {len(mtbo_dfs)} MTBO runs for Reizman Suzuki case {i} with auxiliary of Baumgarnter Suzuki",
        )

        # Make subplot
        ax_yld = fig_yld.add_subplot(1, 4, k)
        make_yld_comparison_plot(
            dict(results=stbo_dfs, label="STBO", color="#a50026"),
            dict(results=stbo_head_start_dfs, label="STBO HS", color="#FDAE61"),
            dict(results=mtbo_dfs, label="MTBO", color="#313695"),
            output_name="yld_best",
            ax=ax_yld,
        )

        # Format subplot
        aux_names = ",".join(
            [f"{reizman_case_to_name[j]}" for j in range(1, 5) if j != i]
        )
        ax_yld.set_title(
            f"{reizman_case_to_name[i]} (Aux. {aux_names})", fontsize=heading_fontsize
        )
        xlabels = np.arange(0, 21, 5)
        ax_yld.set_xticks(xlabels)
        ax_yld.set_xticklabels(xlabels, fontsize=axis_fontsize)
        ax_yld.set_yticks([0, 20, 40, 60, 80, 100, 120])
        ax_yld.set_yticklabels([0, 20, 40, 60, 80, 100, ""], fontsize=axis_fontsize)
        ax_yld.tick_params(direction="in")

        # Make catalyst comparison plot
        ax_cat_best = fig_cat_best.add_subplot(1, 4, k)
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
        ax_cat_counts = fig_cat_counts.add_subplot(1, 4, k)
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

        # Format subplot
        ax_cat_best.set_title(
            f"{reizman_case_to_name[i]} (Aux. {aux_names})", fontsize=heading_fontsize
        )
        ax_cat_best.set_ylim(0.0, 1.0)
        ax_cat_best.tick_params("y", labelsize=12, direction="in")
        ax_cat_counts.set_title(
            f"{reizman_case_to_name[i]} (Aux. {aux_names})", fontsize=heading_fontsize
        )
        ax_cat_counts.set_ylim(0.0, 1.0)
        ax_cat_counts.tick_params("y", labelsize=12, direction="in")
        k += 1

    # Format and save figure
    # fig.suptitle("Reizman Optimization", fontsize=21)
    fig_yld.supxlabel("Experiment number", fontsize=heading_fontsize)
    fig_yld.supylabel("Best Yield (%)", fontsize=heading_fontsize)
    fig_yld.tight_layout()
    figure_dir = Path(figure_dir)
    fig_yld.savefig(
        figure_dir / "reizman_suzuki_reizman_suzuki_all_cotraining_optimization.png",
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
        / "reizman_suzuki_reizman_suzuki_all_cotraining_optimization_catalyst_best.png",
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
        / "reizman_suzuki_reizman_suzuki_all_cotraining_optimization_catalyst_counts.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
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
    reizman_suzuki_auxiliary_all_reizman_suzuki(
        num_iterations=num_iterations,
        include_tags=include_tags,
        filter_tags=filter_tags,
        only_finished_runs=only_finished_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        figure_dir=figure_dir,
    )
