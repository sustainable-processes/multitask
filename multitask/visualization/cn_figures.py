"""
Make figures for publication
"""
from .plots import make_comparison_plot
from multitask.utils import download_runs_wandb
from summit import *
import wandb
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
import string

import logging


logger = logging.getLogger(__name__)


def baumgartner_cn_auxiliary_baumgartner_cn(
    num_iterations: int = 20,
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = "multitask",
    figure_dir: Optional[str] = "figures",
):
    """Make plots for Baumgartner C-N optimization with auxiliary of Baumgartner C-N."""
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
    logger.info(
        "Making plots for Baumgartner C-N optimization with auxiliary of Baumgartner C-N"
    )
    letters = list(string.ascii_lowercase)
    for i in range(1, 5):
        # Filter STBO data
        stbo_dfs = [
            run.history()
            for run in runs
            if run.config.get("model_name") == f"baumgartner_cn_case_{i}"
            and run.config.get("strategy") == "STBO"
        ]
        stbo_dfs = [
            stbo_df for stbo_df in stbo_dfs if stbo_df.shape[0] == num_iterations
        ]
        if len(stbo_dfs) == 0:
            raise ValueError("No Baumgartner STBO runs found")
        for j in range(1, 5):
            if i != j:
                # Filter MTBO data
                mtbo_dfs = [
                    run.history()
                    for run in runs
                    if run.config.get("strategy") == "MTBO"
                    and run.config.get("model_name") == f"baumgartner_cn_case_{i}"
                    and run.config.get("ct_dataset_names")[0]
                    == f"baumgartner_cn_case_{j}"
                    and len(run.config.get("ct_dataset_names")) == 1
                ]

                mtbo_dfs = [
                    mtbo_df
                    for mtbo_df in mtbo_dfs
                    if mtbo_df.shape[0] == num_iterations
                ]
                if len(mtbo_dfs) == 0:
                    raise ValueError(
                        f"No Baumgartner MTBO runs found for case {i} (auxiliary Baumgartner {j})"
                    )
                logger.info(
                    f"Found {len(stbo_dfs)} STBO and {len(mtbo_dfs)} MTBO runs for Baumgartner C-N case {i} with auxiliary of Baumgartner C-N case {j}"
                )

                # Make subplot
                ax = fig.add_subplot(4, 3, k)
                make_comparison_plot(
                    dict(results=stbo_dfs, label="STBO", color="#a50026"),
                    dict(results=mtbo_dfs, label="MTBO", color="#313695"),
                    output_name="yld_best",
                    ax=ax,
                )
                ax.set_title(f"({letters[k-1]}) Case {i} - Auxiliary {j}")
                ax.set_ylim(0, 100)
                k += 1

    # Format plot
    fig.suptitle("Baumgarnter Optimization")
    fig.supxlabel("Number of reactions")
    fig.supylabel("Yield (%)")
    fig.tight_layout()
    figure_dir = Path(figure_dir)
    fig.savefig(
        figure_dir / "baumgartner_cn_baumgartner_cn_one_cotraining_optimization.png",
        dpi=300,
    )
