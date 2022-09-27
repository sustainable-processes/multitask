from pathlib import Path
from summit import *
from rdkit import Chem
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from typing import Dict, List, Optional
import typer
import wandb
from wandb.apis.public import Run


font_manager.findfont("Calibri", fontext="afm", rebuild_if_missing=True)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        # Use LaTeX default serif font.
        "font.serif": ["Calibri"],
    }
)


def make_average_plot(
    results: List[pd.DataFrame],
    output_name: str,
    ax,
    label: Optional[str] = None,
    color: Optional[str] = None,
):
    yields = [df[output_name] for df in results]
    yields = np.array(yields)
    mean_yield = np.mean(yields, axis=0)
    std_yield = np.std(yields, axis=0)
    x = np.arange(0, len(mean_yield), 1).astype(int)
    ax.plot(x, mean_yield, label=label, linewidth=4, c=color)
    ax.fill_between(
        x,
        mean_yield - 1.96 * std_yield,
        mean_yield + 1.96 * std_yield,
        alpha=0.1,
        color=color,
    )


def make_repeats_plot(
    results: List[pd.DataFrame], output_name: str, ax, label=None, color=None
):
    yields = [df[output_name] for df in results]
    x = np.arange(0, len(yields[0]), 1).astype(int)
    line = ax.plot(
        x,
        yields[0],
        label=label,
        linewidth=2,
        alpha=0.25,
        c=color,
        solid_capstyle="round",
    )
    for data in yields[1:]:
        line = ax.plot(
            x, data, linewidth=2, alpha=0.25, c=color, solid_capstyle="round"
        )


def make_comparison_plot(*args, output_name: str, ax, plot_type: str = "average"):
    for arg in args:
        if plot_type == "average":
            make_average_plot(
                arg["results"],
                output_name,
                ax,
                label=arg["label"],
                color=arg.get("color"),
            )
        elif plot_type == "repeats":
            make_repeats_plot(
                arg["results"],
                output_name,
                ax,
                label=arg["label"],
                color=arg.get("color"),
            )
        else:
            raise ValueError(f"{plot_type} must be average or repeats")
    fontdict = fontdict = {"size": 12}
    if plot_type == "average":
        ax.legend(prop=fontdict, framealpha=0.0)
    else:
        ax.legend(prop=fontdict)
    ax.set_xlim(0, 20)
    ax.set_xticks(np.arange(0, 20, 2).astype(int))
    ax.tick_params(direction="in")
    return ax


def remove_frame(ax, sides=["top", "left", "right"]):
    for side in sides:
        ax_side = ax.spines[side]
        ax_side.set_visible(False)


def download_runs_wandb(
    api: wandb.Api,
    wandb_entity: str = "ceb-sre",
    wandb_project: str = "multitask",
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
) -> List[Run]:
    """
    Parameters
    ----------
    api : wandb.Api
        The wandb API object.
    wandb_entity : str, optional
        The wandb entity to search, by default "ceb-sre"
    wandb_project : str, optional
        The wandb project to search, by default "multitask"
    include_tags : Optional[List[str]], optional
        A list of tags that the run must have, by default None
    filter_tags : Optional[List[str]], optional
        A list of tags that the run must not have, by default None

    """
    runs = api.runs(f"{wandb_entity}/{wandb_project}")

    final_runs = []
    for run in runs:
        # Filtering
        if include_tags is not None:
            if not all([tag in run.tags for tag in include_tags]):
                continue
            if any([tag in run.tags for tag in filter_tags]):
                continue
        if only_finished_runs and run.state != "finished":
            continue
        # Append runs
        final_runs.append(run)
    return final_runs


def make_baumgartner_plots(runs: List[Run]):
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


def make_reizman_auxiliary_baumgartner_plots(
    runs: List[Run], num_iterations: int = 20, figure_dir: str = "figures"
):
    # Setup figure
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(wspace=0.2, hspace=0.5)
    k = 1
    axis_fontsize = 14

    # Make subplots for each case
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

        # Make subplot
        ax = fig.add_subplot(1, 4, k)
        make_comparison_plot(
            dict(results=stbo_dfs, label="STBO", color="#a50026"),
            dict(results=mtbo_dfs, label="MTBO", color="#313695"),
            output_name="yld_best",
            ax=ax,
        )

        # Format subplot
        ax.set_title(f"Case {i}", fontsize=21)
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


def main(
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    wandb_entity: str = "ceb-sre",
    wandb_project: str = "multitask",
    figure_dir: str = "figures",
    num_iterations: int = 20,
):
    """ "

    Parameters
    ----------
    include_tags : list of str, optional
        Only include runs with these tags, by default None
    filter_tags : list of str, optional
        Exclude runs with these tags, by default None
    only_finished_runs : bool, optional
        Only include runs that have finished, by default True
    wandb_entity : str, optional
        The wandb entity to search, by default "ceb-sre"
    wandb_project : str, optional
        The wandb project to search, by default "multitask"
    figure_dir : str, optional
        The directory to save figures to, by default "figures"
    num_iterations : int, optional


    """
    # Setup wandb
    wandb.login()
    api = wandb.Api()

    # Get runs
    runs = download_runs_wandb(
        api,
        wandb_entity,
        wandb_project,
        only_finished_runs=only_finished_runs,
        include_tags=include_tags,
        filter_tags=filter_tags,
    )

    # Make plots
    # make_baumgartner_plots(runs)
    make_reizman_auxiliary_baumgartner_plots(
        runs, num_iterations=num_iterations, figure_dir=figure_dir
    )


if __name__ == "__main__":
    typer.run(main)
