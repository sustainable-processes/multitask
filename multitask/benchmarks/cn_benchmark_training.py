from multitask.utils import *
from multitask.etl.cn_data_utils import get_cn_dataset
from summit import *
from pathlib import Path
from typing import Tuple, Optional
import logging


def train_benchmark(
    data_path: str,
    save_path: str,
    figure_path: str,
    dataset_name: Optional[str] = None,
    max_epochs: Optional[int] = 1000,
    cv_folds: Optional[int] = 5,
    verbose: Optional[int] = 0,
) -> ExperimentalEmulator:
    """Train a C-N benchmark"""
    # Get data
    ds, domain = prepare_domain_data(
        data_path=data_path,
    )

    if dataset_name is None:
        dataset_name = Path(data_path).parts[-1].rstrip(".pb")

    # Create emulator benchmark
    emulator = ExperimentalEmulator(dataset_name, domain, dataset=ds)

    # Train emulator
    emulator.train(max_epochs=max_epochs, cv_folds=cv_folds, verbose=verbose)

    # Parity plot
    fig, axes = emulator.parity_plot(include_test=True)
    ax = axes[0]
    ax.set_title("")
    ax.set_xlabel("Measured Yield (%)")
    ax.set_ylabel("Predicted Yield (%)")
    figure_path = Path(figure_path)
    fig.savefig(figure_path / f"{dataset_name}_parity_plot.png", dpi=300)

    # Save emulator
    emulator.save(save_dir=save_path)

    return emulator


def create_cn_domain() -> Domain:
    """Create the domain for the optimization"""
    domain = Domain()

    # Decision variables
    des_1 = "Catalyst"
    domain += CategoricalVariable(
        name="catalyst",
        description=des_1,
        levels=[
            "cycloPd tBuXPhos 4-Chlorotoluene",
            "cycloPd EPhos 4-Chlorotoluene",
            "cycloPd AlPhos 4-Chlorotoluene",
            "cycloPd tBuBrettPhos 4-Chlorotoluene",
        ],
    )

    des_2 = "Base"
    domain += CategoricalVariable(
        name="base",
        description=des_2,
        levels=[
            "TEA",
            "Triethylamine",
            "TMG",
            "BTMG",
            "DBU",
            "MTBD",
            "BTTP",
            "P2Et",
        ],
    )

    des_3 = "Solvent"
    domain += CategoricalVariable(
        name="solvent",
        description=des_3,
        levels=[
            "2-MeTHF",
            "DMSO",
        ],
    )

    des_4 = "Residence time in seconds (s)"
    domain += ContinuousVariable(name="time", description=des_4, bounds=[60, 6000])

    des_5 = "Reactor temperature in degrees Celsius (ºC)"
    domain += ContinuousVariable(
        name="temperature",
        description=des_5,
        bounds=[30, 110],
    )

    # Objectives
    des_6 = "Yield"
    domain += ContinuousVariable(
        name="yld",
        description=des_6,
        bounds=[0, 105],
        is_objective=True,
        maximize=True,
    )
    return domain


def prepare_domain_data(
    data_path: str,
) -> Tuple[dict, Domain]:
    """Prepare domain and data for downstream tasks"""
    logger = logging.getLogger(__name__)
    # Get data
    ds = get_cn_dataset(data_path)

    # Create domain
    domain = create_cn_domain()
    return ds, domain
