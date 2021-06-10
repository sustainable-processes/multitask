from multitask.utils import *
from multitask.suzuki_data_utils import *
from multitask.suzuki_emulator import SuzukiEmulator
from summit import *

import ord_schema
from ord_schema import message_helpers, validations
from ord_schema.proto import dataset_pb2
from ord_schema.proto.reaction_pb2 import *

from rdkit import Chem
from pint import UnitRegistry

import typer
from tqdm.auto import tqdm, trange
from pathlib import Path
import pkg_resources
from typing import Iterable, Tuple, Dict, Union, List, Optional
import pandas as pd
import logging
import json


def train_benchmark(
    data_path: str,
    save_path: str,
    figure_path: str,
    dataset_name: Optional[str] = None,
    include_reactant_concentrations: Optional[bool] = False,
    print_warnings: Optional[bool] = True,
    split_catalyst: Optional[bool] = True,
    max_epochs: Optional[int] = 1000,
    cv_folds: Optional[int] = 5,
    verbose: Optional[int] = 0,
) -> None:
    """Train a Suzuki benchmark"""
    # Get data
    ds, domain = prepare_domain_data(
        data_path=data_path,
        include_reactant_concentrations=include_reactant_concentrations,
        split_catalyst=split_catalyst,
        print_warnings=print_warnings,
    )

    if dataset_name is None:
        dataset_name = Path(data_path).parts[-1].rstrip(".pb")

    # Create emulator benchmark
    emulator = SuzukiEmulator(
        dataset_name, domain, dataset=ds, split_catalyst=split_catalyst
    )

    # Train emulator
    emulator.train(max_epochs=max_epochs, cv_folds=cv_folds, verbose=verbose)

    # Parity plot
    fig, _ = emulator.parity_plot(include_test=True)
    figure_path = Path(figure_path)
    fig.savefig(figure_path / f"{dataset_name}_parity_plot.png", dpi=300)

    # Save emulator
    emulator.save(save_dir=save_path)


def create_suzuki_domain(
    include_reactant_concentrations: Optional[bool] = False,
    split_catalyst: Optional[bool] = True,
    catalyst_list: Optional[list] = None,
    pre_catalyst_list: Optional[list] = None,
    ligand_list: Optional[list] = None,
) -> Domain:
    """Create the domain for the optimization"""
    domain = Domain()

    # Decision variables
    if include_reactant_concentrations:
        domain += ContinuousVariable(
            name="electrophile_concentration",
            description="Concentration of electrophile in molar",
            bounds=[0, 2],
        )
        domain += ContinuousVariable(
            name="nucleophile_concentration",
            description="Concentration of nucleophile in molar",
            bounds=[0, 2],
        )

    if split_catalyst:
        domain += CategoricalVariable(
            name="pre_catalyst_smiles",
            description="SMILES of the pre-catalyst",
            levels=pre_catalyst_list,
        )

        domain += CategoricalVariable(
            name="ligand_smiles",
            description="SMILES of the ligand",
            levels=ligand_list,
        )
    else:
        domain += CategoricalVariable(
            name="catalyst_smiles",
            description="Catalyst including pre-catalyst and ligand",
            levels=catalyst_list,
        )

    domain += ContinuousVariable(
        name="catalyst_concentration",
        description="Concentration of pre_catalyst in molar",
        bounds=[0, 2],
    )
    domain += ContinuousVariable(
        name="ligand_ratio",
        description="Ratio of pre-catalyst to ligand",
        bounds=[0, 5],
    )

    domain += ContinuousVariable(
        name="temperature",
        description="Reaction temperature in deg C",
        bounds=[20, 120],
    )

    domain += ContinuousVariable(
        name="time", description="Reaction time in seconds", bounds=[60, 120 * 60]
    )

    # Objectives
    domain += ContinuousVariable(
        name="yld",
        description="Reaction yield",
        bounds=[0, 100],
        is_objective=True,
        maximize=True,
    )
    return domain


def prepare_domain_data(
    data_path: str,
    include_reactant_concentrations: Optional[bool] = False,
    split_catalyst: Optional[bool] = True,
    print_warnings: Optional[bool] = True,
) -> Tuple[dict, Domain]:
    """Prepare domain and data for downstream tasks"""
    logger = logging.getLogger(__name__)
    # Get data
    ds = get_suzuki_dataset(
        data_path,
        split_catalyst=split_catalyst,
        print_warnings=print_warnings,
    )

    # Create domains
    if split_catalyst:
        pre_catalysts = ds["pre_catalyst_smiles"].unique().tolist()
        logger.info("Number of pre-catalysts:", len(pre_catalysts))
        ligands = ds["ligand_smiles"].unique().tolist()
        logger.info("Number of ligands:", len(ligands))
        domain = create_suzuki_domain(
            split_catalyst=True,
            pre_catalyst_list=pre_catalysts,
            ligand_list=ligands,
            include_reactant_concentrations=include_reactant_concentrations,
        )
    else:
        catalysts = ds["catalyst_smiles"].unique().tolist()
        logger.info("Number of catalysts:", len(catalysts))
        domain = create_suzuki_domain(split_catalyst=False, catalyst_list=catalysts)
    return ds, domain


if __name__ == "__main__":
    typer.run(train_benchmark)
