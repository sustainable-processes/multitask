import logging
import time
from multitask.utils import download_runs_wandb
from summit.utils.dataset import DataSet
import pandas as pd
import numpy as np
import json
from requests import ConnectionError
from wandb import Artifact
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
from matplotlib.axes import Axes


def make_average_plot(
    results: List[pd.DataFrame],
    output_name: str,
    ax: Axes,
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
    results: List[pd.DataFrame], output_name: str, ax: Axes, label=None, color=None
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


def make_yld_comparison_plot(
    *args, output_name: str, ax: Axes, plot_type: str = "average"
):
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


def make_best_plot(
    *args,
    output_name: str,
    categorical_variable: str,
    ax: Axes,
    categorical_map: Optional[Dict[str, str]] = None,
    normalize: bool = True,
):
    best_dict = {}
    for arg in args:
        dfs = arg["results"]
        best_values = []
        for df in dfs:
            ind = df[output_name].argmax()
            best_values.append(df.loc[ind, categorical_variable].values[0])
        best_dict[arg["label"]] = best_values
    display_df = pd.DataFrame(best_dict)
    if categorical_map is not None:
        display_df = display_df.replace(categorical_map)
    ser = [
        display_df[col].value_counts(normalize=normalize) for col in display_df.columns
    ]
    count_df = pd.DataFrame(ser).T
    color = {arg["label"]: arg["color"] for arg in args}
    count_df.plot.bar(ax=ax, rot=45, color=color)
    return ax


def make_counts_plot(
    *args,
    output_name: str,
    categorical_variable: str,
    ax: Axes,
    categorical_map: Optional[Dict[str, str]] = None,
    normalize: bool = True,
):
    count_dfs = []
    for arg in args:
        big_df = pd.concat(arg["results"])
        if categorical_map is not None:
            big_df = big_df.replace(categorical_map)
        count_series = big_df.groupby(categorical_variable).size()
        count_dfs.append(count_series)
    count_df = pd.concat(count_dfs, axis=1, join="outer").fillna(0)
    count_df = count_df.rename(columns={i: arg["label"] for i, arg in enumerate(args)})

    if normalize:
        count_df = count_df / count_df.sum()
    color = {arg["label"]: arg["color"] for arg in args}
    count_df.plot.bar(ax=ax, rot=45, color=color)
    return ax


def make_categorical_comparison_plot(
    *args,
    output_name: str,
    categorical_variable: str,
    ax: Axes,
    categorical_map: Optional[Dict[str, str]] = None,
    normalize: bool = True,
    plot_type: str = "best",
):
    """Make a comparison of how often each categorical variable is the best

    Parameters
    ----------
    args : List[Dict]
        A list of dictionaries with the following keys:
            - results: List[pd.DataFrame]
                A list of dataframes with the results of the experiment
            - label: str
                The label for the plot
            - color: str
                The color for the plot
    output_name : str
        The name of the output variable
    categorical_variable : str
        The name of the categorical variable
    ax : Axes
        The matplotlib axes to plot on
    categorical_map : Dict[str, str], optional
        A dictionary mapping the categorical variable to a string, by default None
    normalize : bool, optional
        Whether to normalize the results, by default True


    """
    # Plot frequency of each categorical value selection
    # each categorical value was the best
    if plot_type == "best":
        make_best_plot(
            *args,
            output_name=output_name,
            categorical_variable=categorical_variable,
            ax=ax,
            categorical_map=categorical_map,
            normalize=normalize,
        )
    elif plot_type == "counts":
        make_counts_plot(
            *args,
            output_name=output_name,
            categorical_variable=categorical_variable,
            ax=ax,
            categorical_map=categorical_map,
            normalize=normalize,
        )

    # Format plot
    fontdict = fontdict = {"size": 12}
    ax.legend(prop=fontdict, framealpha=0.0)
    ax.tick_params(direction="in")
    ax.set_xlabel("")
    return ax


def remove_frame(ax, sides=["top", "left", "right"]):
    for side in sides:
        ax_side = ax.spines[side]
        ax_side.set_visible(False)


def get_ds_from_json(filepath: str) -> DataSet:
    path = Path(filepath)
    with open(path, "r") as f:
        data = json.load(f)
    data = data["experiment"]["data"]
    return DataSet.from_dict(data)


def download_with_retries(artifact: Artifact, n_retries: int = 5):
    logger = logging.getLogger(__name__)
    for i in range(n_retries):
        try:
            return artifact.download()
        except ConnectionError as e:
            logger.error(e)
            logger.info(f"Retrying download of {artifact.name}")
            time.sleep(2**i)


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
    limit: Optional[int] = 20,
) -> List[DataSet]:
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

    dfs = []
    for i, run in enumerate(runs):
        if limit is not None and i >= limit:
            continue
        artifacts = run.logged_artifacts()
        path = None
        for artifact in artifacts:
            if artifact.type == "optimization_result":
                path = download_with_retries(artifact)
                break
        if path is not None:
            path = list(Path(path).glob("repeat_*.json"))[0]
            ds = get_ds_from_json(path)
            ds["yld_best", "DATA"] = ds["yld"].astype(float).cummax()
            dfs.append(ds)

    # if columns is None:
    #     columns = ["yld_best"]
    # dfs = [run.history(x_axis="iteration", keys=columns) for run in tqdm(runs)]
    if num_iterations is not None:
        dfs = [
            df.iloc[:num_iterations, :] for df in dfs if df.shape[0] >= num_iterations
        ]
    if len(dfs) == 0:
        raise ValueError(f"No {model_name} {strategy} runs found")
    return dfs
