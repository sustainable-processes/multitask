from multitask.utils import download_runs_wandb
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional


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
    x = np.arange(1, len(mean_yield) + 1, 1).astype(int)
    ax.plot(x, mean_yield, label=label, linewidth=4, c=color)
    top = mean_yield + 1.96 * std_yield
    bottom = mean_yield - 1.96 * std_yield
    bottom = np.clip(bottom, 0, 100)
    top = np.clip(top, 0, 100)
    ax.fill_between(
        x,
        bottom,
        top,
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


def get_wandb_run_dfs(
    api,
    wandb_entity: str,
    wandb_project: str,
    model_name: str,
    strategy: str,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    extra_filters: Optional[Dict] = None,
    num_iterations: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> List[pd.DataFrame]:
    """Get data from wandb"""
    filters = {
        "config.model_name": model_name,
        "config.strategy": strategy,
    }
    if extra_filters is not None:
        filters.update(extra_filters)
    runs = download_runs_wandb(
        api,
        wandb_entity,
        wandb_project,
        only_finished_runs=only_finished_runs,
        include_tags=include_tags,
        filter_tags=filter_tags,
        extra_filters=filters,
    )
    if columns is None:
        columns = ["yld_best"]
    dfs = [run.history(x_axis="iteration", keys=columns) for run in tqdm(runs)]
    if num_iterations is not None:
        dfs = [df for df in dfs if df.shape[0] == 20]
    if len(dfs) == 0:
        raise ValueError(f"No {model_name} {strategy} runs found")
    return dfs
