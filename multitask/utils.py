import ord_schema
from summit import *
from ord_schema import message_helpers, validations
from ord_schema.proto import dataset_pb2
from ord_schema.proto.reaction_pb2 import *
from ord_schema.message_helpers import find_submessages
from ord_schema import units

from rdkit import Chem
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
    "get_rxn_yield",
    "ureg"
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


def get_smiles(compound: Compound, canonicalize=True):
    for identifier in compound.identifiers:
        if identifier.type == CompoundIdentifier.SMILES:
            smiles = identifier.value
            return Chem.CanonSmiles(smiles)


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


def get_rxn_yield(outcome):
    yields = []
    for product in outcome.products:
        for measurement in product.measurements:
            if measurement.type == ProductMeasurement.YIELD:
                yields.append(measurement.percentage.value)
    if len(yields)>1:
        raise ValueError("More than one product with a yield in reaction outcome. This is ambiguous.")
    elif len(yields)==0:
        raise ValueError("No reaction yield found in reaction outcome.")
    return yields[0]

