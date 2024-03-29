from multitask.utils import *
from multitask.etl.cn_data_utils import get_cn_dataset
from summit import *
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import wandb
import numpy as np
from skorch.callbacks import WandbLogger

logger = logging.getLogger(__name__)


class PatchWandbLogger(WandbLogger):
    def on_train_begin(self, net, **kwargs):
        return super().on_train_begin(net, **kwargs)


def train_benchmark(
    dataset_name: str,
    save_path: str,
    figure_path: str,
    wandb_dataset_artifact_name: Optional[str] = None,
    data_file: Optional[str] = None,
    max_epochs: Optional[int] = 1000,
    cv_folds: Optional[int] = 5,
    verbose: Optional[int] = 0,
    wandb_benchmark_artifact_name: str = None,
    use_wandb: bool = True,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = "multitask",
) -> ExperimentalEmulator:
    """Train a C-N benchmark"""
    # Setup wandb
    config = dict(locals())
    if use_wandb:
        run = wandb.init(
            job_type="training",
            entity=wandb_entity,
            project=wandb_project,
            tags=["benchmark"],
            config=config,
        )

    # Download data from wandb if not provided
    if data_file is None and use_wandb:
        dataset_artifact = run.use_artifact(wandb_dataset_artifact_name)
        data_file = Path(dataset_artifact.download()) / f"{dataset_name}.pb"
    elif data_file is None and not use_wandb:
        raise ValueError("Must provide data path if not using wandb")

    # Get data
    ds, domain = prepare_domain_data(data_file=data_file)
    logger.info(f"Dataset size: {ds.shape[0]}")
    if use_wandb:
        wandb.config.update({"dataset_size": ds.shape[0]})
        wandb.config.update({"domain": domain.to_dict()})

    # Create emulator benchmark
    emulator = ExperimentalEmulator(dataset_name, domain, dataset=ds)

    # Train emulator
    scores = emulator.train(max_epochs=max_epochs, cv_folds=cv_folds, verbose=verbose)
    if use_wandb:
        wandb.run.summary.update(scores)

    # Parity plot
    axis_fontsize = 14
    heading_fontsize = 18
    fig, axes = emulator.parity_plot(include_test=True)
    ax = axes[0]
    ax.set_title("")
    ax.set_xlabel("Measured Yield (%)", fontsize=heading_fontsize)
    ax.set_ylabel("Predicted Yield (%)", fontsize=heading_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=axis_fontsize)
    figure_path = Path(figure_path)
    figure_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path / f"{dataset_name}_parity_plot.png", dpi=300)

    # Save emulator
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    emulator.save(save_dir=save_path)

    # Upload results to wandb
    if use_wandb:
        if wandb_benchmark_artifact_name is None:
            wandb_benchmark_artifact_name = f"benchmark_{dataset_name}"
        artifact = wandb.Artifact(wandb_benchmark_artifact_name, type="model")
        artifact.add_dir(save_path)
        wandb.log({"parity_plot": wandb.Image(fig)})
        figure_path = Path(figure_path)
        artifact.add_file(figure_path / f"{dataset_name}_parity_plot.png")
        run.log_artifact(artifact)
        run.finish()

    return emulator


def create_cn_domain(
    catalysts: List[str], bases: List[str], base_equiv_bounds: List[float]
) -> Domain:
    """Create the domain for the optimization"""
    domain = Domain()

    # Decision variables
    des_1 = "Catalyst"
    domain += CategoricalVariable(name="catalyst", description=des_1, levels=catalysts)

    des_2 = "Base"
    domain += CategoricalVariable(
        name="base",
        description=des_2,
        levels=bases,
    )

    des_3 = "Base equivalents with respect to p-tolyl triflate (electrophile)"
    domain += ContinuousVariable(
        name="base_equiv",
        description=des_3,
        bounds=base_equiv_bounds,
    )

    des_4 = "Residence time in seconds (s)"
    domain += ContinuousVariable(name="time", description=des_4, bounds=[60, 6000])

    des_5 = "Reactor temperature in degrees Celsius (ºC)"
    domain += ContinuousVariable(
        name="temperature",
        description=des_5,
        bounds=[30, 100],
    )

    # Objectives
    des_6 = "Yield"
    domain += ContinuousVariable(
        name="yld",
        description=des_6,
        bounds=[0, 100],
        is_objective=True,
        maximize=True,
    )
    return domain


def prepare_domain_data(
    data_file: str,
) -> Tuple[dict, Domain]:
    """Prepare domain and data for downstream tasks"""
    logger = logging.getLogger(__name__)
    # Get data
    ds = get_cn_dataset(data_file)

    # Create domain
    # Base equivalents bounds vary between cases
    bounds = round(ds["base_equiv"].min(), 1), round(ds["base_equiv"].max(), 1)
    catalysts = ds["catalyst"].unique().tolist()
    bases = ds["base"].unique().tolist()
    domain = create_cn_domain(
        catalysts=catalysts, bases=bases, base_equiv_bounds=bounds
    )
    return ds, domain
