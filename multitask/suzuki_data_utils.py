from multitask.utils import *
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

__all__ = ["get_suzuki_dataset", "suzuki_reaction_to_dataframe", "get_suzuki_row"]


def get_suzuki_dataset(data_path, split_catalyst=True, print_warnings=True) -> DataSet:
    """
    Get a Suzuki ORD dataset as a Summit DataSet

    :param data_path:
    :param split_catalyst:
    :param print_warnings:
    :return:
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise ImportError(f"Could not import {data_path}")
    dataset = message_helpers.load_message(str(data_path), dataset_pb2.Dataset)
    valid_output = validations.validate_message(dataset)
    if print_warnings:
        print(valid_output.warnings)
    df = suzuki_reaction_to_dataframe(dataset.reactions, split_catalyst=split_catalyst)
    return DataSet.from_df(df)


def suzuki_reaction_to_dataframe(
    reactions: Iterable[Reaction], split_catalyst: bool = True
) -> pd.DataFrame:
    """Convert a list of reactions into a dataframe that can be used for machine learning

    Parameters
    ---------
    reactions: list of Reaction
        A list of ORD reaction objects
    split_catalyst: bool
        Whether to split the catalyst into pre-catalyst and ligand

    Returns
    -------
    df: pd.DataFrame
        The dataframe with each row as a reaction

    """
    # Do the transformation
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
            f"Nucleophiles in this dataset: {nucleophiles}"
        )

    return df


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
