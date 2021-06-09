from multitask.utils import *
from summit import *

import ord_schema
from ord_schema import message_helpers, validations
from ord_schema.proto import dataset_pb2
from ord_schema.proto.reaction_pb2 import *

from rdkit import Chem
from pint import UnitRegistry

import typer
from pathlib import Path
import pkg_resources
from typing import Iterable, Tuple, Dict, Union, List, Optional
import pandas as pd

__all__ = ["get_suzuki_datasets", "suzuki_reaction_to_dataframe", "prepare_domain_data"]


def contains_boron(smiles: str) -> bool:
    """Check for the presence of Boron"""
    boron = Chem.MolFromSmarts("[B]")
    mol = Chem.MolFromSmiles(smiles)
    res = mol.GetSubstructMatches(boron)
    if len(res) > 0:
        return True
    else:
        False


def get_suzuki_row(reaction: Reaction, split_catalyst: bool = True) -> dict:
    """Convert a Suzuki ORD reaction into a dictionary"""
    row = {}
    rxn_inputs = reaction.inputs
    reactants = 0
    total_volume = calculate_total_volume(rxn_inputs, include_workup=False)
    # Components and concentrations
    for k in rxn_inputs:
        for component in rxn_inputs[k].components:
            amount = component.amount
            pint_value = get_pint_amount(amount)
            conc = pint_value / total_volume
            if component.reaction_role == ReactionRole.CATALYST:
                # TODO: make sure the catalyst and ligand are one of the approved ones
                smiles = get_smiles(component)
                if split_catalyst:
                    pre_catalyst, ligand = split_cat_ligand(smiles)
                    row["pre_catalyst_smiles"] = pre_catalyst
                    row["ligand_smiles"] = ligand
                else:
                    row["catalyst_smiles"] = smiles
                row["catalyst_concentration"] = conc.to(
                    ureg.moles / ureg.liters
                ).magnitude
                row["ligand_ratio"] = 1.0
            if component.reaction_role == ReactionRole.REACTANT:
                if reactants > 2:
                    raise ValueError(
                        f"Suzuki couplings can only have 2 reactants but this has {reactants} reactants."
                    )
                smiles = get_smiles(component)
                # Nucleophile in Suzuki is always a boronic acid
                boron = contains_boron(smiles)
                name = "nucleophile" if boron else "electrophile"
                row[f"{name}_smiles"] = smiles
                row[f"{name}_concentration"] = conc.to(
                    ureg.moles / ureg.liters
                ).magnitude
                reactants += 1
            if component.reaction_role == ReactionRole.REAGENT:
                # TODO: make base is one of the approved bases
                row[f"reagent"] = get_smiles(component)
                conc = get_pint_amount(component.amount) / total_volume
                row[f"reagent_concentration"] = conc.to(
                    ureg.moles / ureg.liters
                ).magnitude
            if component.reaction_role == ReactionRole.SOLVENT:
                row["solvent"] = get_smiles(component)

    # Temperature
    sp = reaction.conditions.temperature.setpoint
    units = Temperature.TemperatureUnit.Name(sp.units)
    temp = ureg.Quantity(sp.value, ureg(units.lower()))
    row["temperature"] = temp.to(ureg.degC).magnitude

    # Reaction time
    time = reaction.outcomes[0].reaction_time
    units = Time.TimeUnit.Name(time.units)
    time = time.value * ureg(units.lower())
    row["time"] = time.to(ureg.minute).magnitude

    # Yield
    row["yld"] = get_rxn_yield(reaction.outcomes[0])

    # TODO reaction quench
    return row


def suzuki_reaction_to_dataframe(
    reactions: Iterable[Reaction], split_catalyst: bool = True
) -> pd.DataFrame:
    """Convert a list of reactions into a dataframe that can be used for machine learning"""
    # Conversion
    df = pd.DataFrame(
        [
            get_suzuki_row(reaction, split_catalyst=split_catalyst)
            for reaction in reactions
        ]
    )
    # Assert that all rows have the same electrophile and nucleophile
    electrophiles = df["electrophile_smiles"].unique()
    if len(electrophiles) > 1:
        raise ValueError(
            f"Each dataset should contain only one electrophile. "
            f"Electrophiles in this dataset: {electrophiles}"
        )
    nucleophiles = df["nucleophile_smiles"].unique()
    if len(nucleophiles) > 1:
        raise ValueError(
            f"Each dataset should contain only one nucleophile. "
            f"Nucleophile in this dataset: {nucleophiles}"
        )

    return df


def create_suzuki_domain(
    split_catalyst: bool = True,
    catalyst_list: list = None,
    pre_catalyst_list: list = None,
    ligand_list: list = None,
) -> Domain:
    """Create the domain for the optimization"""
    domain = Domain()

    # Decision variables
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

    # Objectives
    domain += ContinuousVariable(
        name="yld", description="Reaction yield", bounds=[0, 100], is_objective=True
    )
    return domain


def get_suzuki_datasets(data_paths, split_catalyst=True, print_warnings=True):
    """Get all suzuki ORD datasets"""
    directories = [Path(data_path) for data_path in data_paths]
    dss = {}
    for directory in directories:
        # Get all protobuf files
        data_paths = directory.glob("*.pb")
        for data_path in data_paths:
            dataset = message_helpers.load_message(str(data_path), dataset_pb2.Dataset)
            valid_output = validations.validate_message(dataset)
            if print_warnings:
                print(valid_output.warnings)
            df = suzuki_reaction_to_dataframe(
                dataset.reactions, split_catalyst=split_catalyst
            )
            name = data_path.parts[-1].rstrip(".pb")
            dss[name] = DataSet.from_df(df)
    return dss


def prepare_domain_data(
    data_paths: List[str],
    split_catalyst: Optional[bool] = True,
    print_warnings: Optional[bool] = True,
) -> Tuple[dict, Domain]:
    """Prepare domain and data for downstream tasks"""
    # Get data
    dfs = get_suzuki_datasets(
        data_paths, split_catalyst=split_catalyst, print_warnings=print_warnings
    )
    big_df = pd.concat(list(dfs.values()))

    # Create domains
    if split_catalyst:
        pre_catalysts = big_df["pre_catalyst_smiles"].unique().tolist()
        ligands = big_df["ligand_smiles"].unique().tolist()
        domain = create_suzuki_domain(
            split_catalyst=True, pre_catalyst_list=pre_catalysts, ligand_list=ligands
        )
    else:
        catalysts = big_df["catalyst_smiles"].unique().tolist()
        domain = create_suzuki_domain(split_catalyst=False, catalyst_list=catalysts)

    return dfs, domain


def train_benchmark(
    dataset_name,
    data_paths: List[str],
    save_path: str,
    figure_path: str,
    print_warnings: Optional[bool] = True,
    split_catalyst: Optional[bool] = True,
    max_epochs: Optional[int] = 1000,
    verbose: Optional[int] = 0,
) -> None:
    # Get data
    dfs, domain = prepare_domain_data(
        data_paths=data_paths,
        split_catalyst=split_catalyst,
        print_warnings=print_warnings,
    )

    # Create emulator benchmark
    emulator = ExperimentalEmulator(dataset_name, domain, dataset=dfs[dataset_name])

    # Train emulator
    emulator.train(max_epochs=max_epochs, verbose=verbose)

    # Parity plot
    fig, _ = emulator.parity_plot(include_test=True)
    figure_path = Path(figure_path)
    fig.savefig(figure_path / f"{dataset_name}_parity_plot.png", dpi=300)

    # Save emulator
    emulator.save(save_dir=save_path)


if __name__ == "__main__":
    typer.run(train_benchmark)
