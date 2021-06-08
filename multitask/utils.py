import ord_schema
from ord_schema import message_helpers, validations
from ord_schema.proto import dataset_pb2
from ord_schema.proto.reaction_pb2 import *
from ord_schema.message_helpers import find_submessages
from ord_schema import units

from pint import UnitRegistry

from pathlib import Path
from typing import Iterable
import pandas as pd

ureg = UnitRegistry()

__all__ = [
    "get_pint_amount",
    "get_smiles",
    "split_cat_ligand",
    "calculate_total_volume",
    "suzuki_reaction_to_dataframe",
]


def get_pint_amount(amount: Amount):
    """Get an amount in terms of pint units"""
    kind = amount.WhichOneof("kind")
    units = getattr(amount, kind).units
    value = getattr(amount, kind).value
    pint_value = None
    if kind == "moles":
        units_str = Moles.MolesUnit.Name(units)
    elif kind == "volume":
        units_str = Volume.VolumeUnit.Name(units)
    elif kind == "mass":
        units_str = Mass.MassUnit.Name(units)
    return value * ureg(units_str.lower())


def get_smiles(compound: Compound):
    for identifier in compound.identifiers:
        if identifier.type == CompoundIdentifier.SMILES:
            return identifier.value


def split_cat_ligand(smiles: str):
    compounds = smiles.split(".")
    pre_catalyst = ""
    ligand = ""
    for compound in compounds:
        # Pre-catalyst always contain Pd
        if "Pd" in compound:
            pre_catalyst = compound
        # Only considering organophosphorus lignads
        elif "P" in compound and "Pd" not in compound:
            ligand = compound
    return pre_catalyst, ligand


def calculate_total_volume(rxn_inputs, include_workup=False):
    # Calculate total reaction volume
    total_volume = 0.0 * ureg.ml
    for k in rxn_inputs:
        for component in rxn_inputs[k].components:
            amount = component.amount
            include_workup = (
                True
                if include_workup
                else component.reaction_role != ReactionRole.WORKUP
            )
            if amount.WhichOneof("kind") == "volume" and include_workup:
                total_volume += get_pint_amount(amount)
    return total_volume


def get_suzuki_row(reaction: Reaction) -> dict:
    """Convert a Suzuki ORD reaction into a dictionary"""
    row = {}
    reaction = baumgartner_datasets[0].reactions[0]
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
                pre_catalyst, ligand = split_cat_ligand(smiles)
                row["pre_catalyst_smiles"] = pre_catalyst
                row["catalyst_concentration (M)"] = conc.to(
                    ureg.moles / ureg.liters
                ).magnitude
                row["ligand_smiles"] = ligand
                row["ligand_ratio"] = 1.0
            if component.reaction_role == ReactionRole.REACTANT:
                row[f"reactant_{reactants+1}"] = get_smiles(component)
                row[f"reactant_{reactants+1}_concentration (M)"] = conc.to(
                    ureg.moles / ureg.liters
                ).magnitude
                reactants += 1
            if component.reaction_role == ReactionRole.REAGENT:
                # TODO: make base is one of the approved bases
                row[f"reagent"] = get_smiles(component)
                conc = get_pint_amount(component.amount) / total_volume
                row[f"reagent_concentration (M)"] = conc.to(
                    ureg.moles / ureg.liters
                ).magnitude
            if component.reaction_role == ReactionRole.SOLVENT:
                row["solvent"] = get_smiles(component)
    # Temperature
    sp = reaction.conditions.temperature.setpoint
    units = Temperature.TemperatureUnit.Name(sp.units)
    temp = ureg.Quantity(sp.value, ureg(units.lower()))
    row["temperature (deg C)"] = temp.to(ureg.degC).magnitude
    # TODO reaction quench
    return row


def suzuki_reaction_to_dataframe(reactions: Iterable[Reaction]) -> pd.DataFrame:
    """Convert a list of reactions into a dataframe that can be used for machine learning"""
    return pd.DataFrame([get_reaction_row(reaction) for reaction in reactions])
