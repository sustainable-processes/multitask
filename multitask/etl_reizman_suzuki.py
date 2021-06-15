"""
Create an ORD dataset from the the paper by reizman et al.

python etl_reizman_suzuki.py ../data/reizman_suzuki/c8re00032h2.xlsx ../data/reizman_suzuki/

"""

from multitask.utils import *
from ord_schema.proto.reaction_pb2 import *
from ord_schema.proto.dataset_pb2 import *
from ord_schema.message_helpers import find_submessages
from ord_schema import validations

import typer
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import json


def main(input_path: str, output_path: str):
    """Entrypoint for running ETL job"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    # Extract data
    for i in range(4):
        name = f"reizman_suzuki_case_{i+1}"
        df = pd.read_csv(input_path / f"{name}.csv")

        # Transform
        tqdm.pandas(desc="Converting to ORD")
        case = i + 1
        reactions = df.progress_apply(inner_loop, axis=1, args=(case,))

        # Create dataset
        dataset = Dataset()
        dataset.name = "Reizman Suzuki Cross-Coupling"
        dataset.reactions.extend(reactions)
        dataset.dataset_id = str(2 + i)

        # Validate dataset
        validation_output = validations.validate_message(dataset)
        print(
            f"First 5 Warnings. See {output_path / f'warnings_{name}.json'} for full output"
        )
        print(validation_output.warnings[:5])
        with open(output_path / f"warnings_{name}.json", "w") as f:
            json.dump(validation_output.warnings, f)

        # Load back
        with open(output_path / f"{name}.pb", "wb") as f:
            f.write(dataset.SerializeToString())


def inner_loop(row: pd.Series, case) -> Reaction:
    """Innter loop for creating a Reaction object for each row in the spreadsheet"""
    # Create reaction
    reaction = Reaction()
    reaction.identifiers.add(value=r"Suzuki Coupling", type=5)
    # Add reactants
    add_electrophile(case, reaction, row)
    add_nucleophile(case, reaction, row)
    add_catalyst(reaction, row)
    add_solvent(reaction, row)
    add_base(reaction, row)

    # Specify conditions
    specify_temperature(reaction, row)
    specify_flow_conditions(reaction, row)

    # Cross checks
    cross_checks(reaction, row)

    # Quench
    quench_reaction(reaction, row)

    # Specify reaction outcome
    specify_outcome(case, reaction, row)

    return reaction


def add_thf_solvent(
    stock: ReactionInput,
    final_solute_conc: float,
    stock_conc: float,
    droplet_volume: float = 14.0,
) -> None:
    """
    final_solute_conc is in molar
    droplet volume in microliters
    stock_conc in molar
    """
    thf = stock.components.add()
    thf.identifiers.add(value="THF", type=CompoundIdentifier.NAME)
    thf.identifiers.add(value=r"C1CCOC1", type=CompoundIdentifier.SMILES)
    thf.reaction_role = ReactionRole.SOLVENT
    thf.amount.volume.units = Volume.MICROLITER
    thf.amount.volume.value = final_solute_conc * droplet_volume / stock_conc
    thf.amount.volume_includes_solutes = True


electrophiles_by_case = {
    1: {"name": "3-bromoquinoline", "smiles": r"Brc1cnc2ccccc2c1"},
    2: {"name": "3-Chloropyridine", "smiles": r"C1=CC(=CN=C1)Cl"},
    3: {"name": "2-Chloropyridine", "smiles": r"Clc1ccccn1"},
    3: {"name": "2-Chloropyridine", "smiles": r"Clc1ccccn1"},
    4: {"name": "2-Chloropyridine", "smiles": r"Clc1ccccn1"},
}


def add_electrophile(case: int, reaction: Reaction, row: pd.Series):
    # Chloropyridine
    electrophile_stock = reaction.inputs["Electrophile"]
    electrophile_stock.addition_order = 1

    # Reactant
    electrophile = electrophile_stock.components.add()
    electrophile.reaction_role = ReactionRole.REACTANT
    details = electrophiles_by_case[case]
    electrophile.identifiers.add(value=details["name"], type=CompoundIdentifier.NAME)
    electrophile.identifiers.add(
        value=details["smiles"], type=CompoundIdentifier.SMILES
    )
    electrophile_conc = 0.167
    amount = electrophile.amount
    amount.moles.units = Moles.MICROMOLE
    amount.moles.value = electrophile_conc * 14.0  # 40 µL droplet
    electrophile.is_limiting = True

    # Internal standard
    internal_std = electrophile_stock.components.add()
    internal_std.reaction_role = ReactionRole.INTERNAL_STANDARD
    internal_std.identifiers.add(value="Naphthelene", type=CompoundIdentifier.NAME)
    internal_std.identifiers.add(
        value=r"c1c2ccccc2ccc1", type=CompoundIdentifier.SMILES
    )
    istd_conc = 0.0059
    amount = internal_std.amount
    amount.mass.units = Mass.MICROGRAM
    amount.mass.value = istd_conc * 14.0

    # Solvent
    add_thf_solvent(electrophile_stock, electrophile_conc, stock_conc=1.4)


nucleophiles_by_case = {
    1: {
        "name": "3,5-Dimethylisoxazole-4-boronic acid pinacol ester",
        "smiles": r"Cc1noc(C)c1B2OC(C)(C)C(C)(C)O2",
    },
    2: {
        "name": "3,5-Dimethylisoxazole-4-boronic acid pinacol ester",
        "smiles": r"Cc1noc(C)c1B2OC(C)(C)C(C)(C)O2",
    },
    3: {"name": "2-benzofuranboronic acid", "smiles": r"OB(O)c1cc2ccccc2o1"},
    4: {"name": "N-Boc-2-pyrroleboronic acid", "smiles": r"CC(C)(C)OC(=O)n1cccc1B(O)O"},
}


def add_nucleophile(case, reaction: Reaction, row: pd.Series):
    # Pinacol ester
    pinacol_ester_stock = reaction.inputs["Nucleophile"]
    pinacol_ester_stock.addition_order = 1

    # Solute
    pinacol_ester = pinacol_ester_stock.components.add()
    pinacol_ester.reaction_role = ReactionRole.REACTANT
    details = nucleophiles_by_case[case]
    pinacol_ester.identifiers.add(
        value=details["name"],
        type=CompoundIdentifier.NAME,
    )
    pinacol_ester.identifiers.add(
        value=details["smiles"], type=CompoundIdentifier.SMILES
    )
    pe_conc = 0.250
    amount = pinacol_ester.amount
    amount.moles.units = Moles.MICROMOLE
    amount.moles.value = pe_conc * 14.0  # 40 µL droplet

    # Solvent
    add_thf_solvent(pinacol_ester_stock, pe_conc, stock_conc=1.0)


pre_catalysts = {
    "P1": {"SMILES": "CS(=O)(=O)O[Pd]c1ccccc1-c2ccccc2N", "name": "Pd G3 µ-OMS"},
    "P2": {"SMILES": "Cl[Pd]c1ccccc1-c2ccccc2N", "name": "Pd G3 Cl"},
}

ligands = {
    "L1": {
        "SMILES": "CC(C)c1cc(C(C)C)c(c(c1)C(C)C)-c2ccccc2P(C3CCCCC3)C4CCCCC4",
        "name": "XPhos",
    },
    "L2": {
        "SMILES": "COc1cccc(OC)c1-c2ccccc2P(C3CCCCC3)C4CCCCC4",
        "name": "SPhos",
    },
    "L3": {
        "SMILES": "CC(C)Oc1cccc(OC(C)C)c1-c2ccccc2P(C3CCCCC3)C4CCCCC4",
        "name": "RuPhos",
    },
    "L4": {
        "SMILES": "CC1(C)c2cccc(P(c3ccccc3)c4ccccc4)c2Oc5c(cccc15)P(c6ccccc6)c7ccccc7",
        "name": "XantPhos",
    },
    "L5": {"SMILES": "C1CCC(CC1)P(C2CCCCC2)C3CCCCC3", "name": "tricyclohexylphosphine"},
    "L6": {"SMILES": "c1ccc(cc1)P(c2ccccc2)c3ccccc3", "name": "triphenylphosphine"},
    "L7": {"SMILES": "CC(C)(C)P(C(C)(C)C)C(C)(C)C", "name": "tri-tert-butylphosphine"},
}


def catalyst_details(pre_catalyst_id: str, ligand_id: str) -> (str, str):
    pre_cat = pre_catalysts[pre_catalyst_id]
    ligand = ligands[ligand_id]
    smiles = f"""{pre_cat["SMILES"]}.{ligand["SMILES"]}"""
    name = pre_cat["name"] + " " + pre_cat["name"]
    return name, smiles


def add_catalyst(reaction: Reaction, row: pd.Series):
    # Catalyst
    catalyst_stock = reaction.inputs["Catalyst"]
    catalyst_stock.addition_order = 1

    # Solute
    catalyst = catalyst_stock.components.add()
    catalyst.reaction_role = ReactionRole.CATALYST
    pre_cat, lig = row["catalyst"].split("-")
    name, smiles = catalyst_details(pre_cat, lig)
    catalyst.identifiers.add(value=name, type=CompoundIdentifier.NAME)
    catalyst.identifiers.add(value=smiles, type=CompoundIdentifier.SMILES)
    cat_conc = row["catalyst_loading"] / 100 * 0.167
    catalyst.amount.moles.units = Moles.MICROMOLE
    catalyst.amount.moles.value = cat_conc * 14.0

    # Solvent
    add_thf_solvent(catalyst_stock, cat_conc, stock_conc=0.018)


def add_solvent(reaction: Reaction, row: pd.Series):
    # Solvent
    solvent_mix = reaction.inputs["Solvent"]
    solvent_mix.addition_order = 1

    # Calculate volume of solvent needed for 40 microliter droplet
    reactants_volume = calculate_total_volume(reaction.inputs, include_workup=False)
    reactants_volume = reactants_volume.to(ureg.microliters).magnitude
    solvent_volume = 14.0 - reactants_volume
    thf_volume = 5 / 6 * solvent_volume
    water_volume = 1 / 6 * solvent_volume

    # THF
    thf = solvent_mix.components.add()
    thf.identifiers.add(value="THF", type=CompoundIdentifier.NAME)
    thf.identifiers.add(value=r"C1CCOC1", type=CompoundIdentifier.SMILES)
    thf.reaction_role = ReactionRole.SOLVENT
    thf.amount.volume.value = thf_volume
    thf.amount.volume.units = Volume.MICROLITER

    # Water
    water = solvent_mix.components.add()
    water.identifiers.add(value="water", type=CompoundIdentifier.NAME)
    water.identifiers.add(value="O", type=CompoundIdentifier.SMILES)
    water.reaction_role = ReactionRole.SOLVENT
    water.amount.volume.value = water_volume
    water.amount.volume.units = Volume.MICROLITER


def add_base(reaction: Reaction, row: pd.Series):
    # Base
    base_stock = reaction.inputs["Base"]
    base_stock.addition_order = 2

    # Solute
    base = base_stock.components.add()
    base.reaction_role = ReactionRole.REAGENT
    base.identifiers.add(value="DBU", type=CompoundIdentifier.NAME)
    base.identifiers.add(value=r"N\2=C1\N(CCCCC1)CCC/2", type=CompoundIdentifier.SMILES)
    base.amount.moles.value = 1.66 * 3.5
    base.amount.moles.units = Moles.MICROMOLE

    # Solvent
    thf = base_stock.components.add()
    thf.identifiers.add(value="THF", type=CompoundIdentifier.NAME)
    thf.identifiers.add(value=r"C1CCOC1", type=CompoundIdentifier.SMILES)
    thf.reaction_role = ReactionRole.SOLVENT
    thf.amount.volume.value = 3.5
    thf.amount.volume.units = Volume.MICROLITER


def specify_temperature(reaction: Reaction, row: pd.Series):
    # Temperature
    temp_conditions = reaction.conditions.temperature
    details = "Flow reactor with four 40W cartridge heaters"
    control = temp_conditions.control
    control.type = TemperatureConditions.TemperatureControl.DRY_ALUMINUM_PLATE
    control.details = details
    temp_conditions.setpoint.value = row["temperature"]
    temp_conditions.setpoint.units = Temperature.TemperatureUnit.CELSIUS


def specify_flow_conditions(reaction: Reaction, row: pd.Series):
    flow_conditions = reaction.conditions.flow
    flow_conditions.type = FlowConditions.CUSTOM
    flow_conditions.details = (
        "A droplet flow reactor system consisting of a liquid handler"
        " and oscillatory flow reactor. Each droplet is like a mini-batch reactor."
    )
    flow_conditions.tubing.type = FlowConditions.Tubing.PFA
    flow_conditions.tubing.diameter.value = 500e-6
    flow_conditions.tubing.details = """500 µm diameter tubing, 240µL reactor"""
    flow_conditions.tubing.diameter.units = Length.METER


def cross_checks(reaction: Reaction, row: pd.Series):
    # Check that reaction volume adds up properly
    vol = calculate_total_volume(reaction.inputs).to(ureg.microliters).magnitude
    try:
        assert np.isclose(vol, 17.5, rtol=1e-2)
    except AssertionError:
        raise ValueError(
            f"Total volume expected to be 43.5 µL but it is actually {vol}µL."
        )
    #
    # # Cross-check concentration calculations using catalyst mol%
    # mol_electrophile = reaction.inputs["Electrophile"].components[0].amount.moles.value
    # cat_mols_check = mol_electrophile * row["Reagent 3 Conc in mol%"]
    # catalyst = reaction.inputs["Catalyst"].components[0]
    # try:
    #     assert np.isclose(catalyst.amount.moles.value, cat_mols_check)
    # except AssertionError:
    #     raise ValueError(
    #         f"Inconsistent amounts. Catalyst should be: {cat_mols_check} micromols"
    #         f", but it is actually {catalyst.amount.moles.value} micromols."
    #     )


def quench_reaction(reaction: Reaction, row: pd.Series):
    # Quench
    quench = reaction.inputs["Quench"]
    quench.addition_order = 3
    reaction_volume = calculate_total_volume(reaction.inputs)

    # Acetone
    acetone = quench.components.add()
    acetone.reaction_role = ReactionRole.WORKUP
    acetone.identifiers.add(value="acetone", type=CompoundIdentifier.NAME)
    acetone.identifiers.add(value="CC(=O)C", type=CompoundIdentifier.SMILES)
    acetone.amount.volume.value = 0.5 * 16.0
    acetone.amount.volume.units = Volume.MICROLITER

    # Water
    water = quench.components.add()
    water.reaction_role = ReactionRole.WORKUP
    water.identifiers.add(value="water", type=CompoundIdentifier.NAME)
    water.identifiers.add(value="O", type=CompoundIdentifier.SMILES)
    water.amount.volume.value = 0.5 * 16.0
    water.amount.volume.units = Volume.MICROLITER

    # Workup specification
    workup = reaction.workups.add()
    workup.amount.volume.value = reaction_volume.to(ureg.microliter).magnitude
    workup.amount.volume.units = Volume.MICROLITER
    workup.type = ReactionWorkup.ADDITION
    details = (
        "As soon as the slug leaves the reactor and is detected at PS2,"
        "a quench solution volume equal to the reaction slug volume"
        "(prepared slug volume + base injection volume) is injected into the reaction slug"
    )
    workup.details = details
    workup.input.CopyFrom(quench)


def define_measurement(measurement: ProductMeasurement, row: pd.Series):
    measurement.analysis_key = "LCMS"
    measurement.type = ProductMeasurement.YIELD
    measurement.percentage.value = row["yld"]


products_by_case = {
    1: {
        "name": "3,5-dimethyl-4-(quinolin-3-yl)isoxazole",
        "smiles": r"CC1=C(C2=CC(C=CC=C3)=C3N=C2)C(C)=NO1",
    },
    2: {
        "name": "3,5-dimethyl-4-(quinolin-3-yl)isoxazole",
        "smiles": r"CC1=C(C2=CC=CN=C2)C(C)=NO1",
    },
    3: {
        "name": "3-(benzofuran-2-yl)pyridine",
        "smiles": r"C1(C2=CC(C=CC=C3)=C3O2)=CC=CN=C1",
    },
    4: {
        "name": "tert-butyl 2-(pyridin-2-yl)-1H-pyrrole-1-carboxylate",
        "smiles": r"CC(OC(N1C=CC=C1C2=CC=CC=N2)=O)(C)C",
    },
}


def specify_outcome(case, reaction: Reaction, row: pd.Series):
    # Reaction Outcome
    outcome = reaction.outcomes.add()

    # Time
    outcome.reaction_time.value = row["t_res"]
    outcome.reaction_time.units = Time.SECOND

    # TODO: conversion calculation

    # Product
    product = outcome.products.add()
    details = products_by_case[case]
    product.identifiers.add(value=details["name"], type=CompoundIdentifier.NAME)
    product.identifiers.add(value=details["smiles"], type=CompoundIdentifier.SMILES)
    product.is_desired_product = True
    product.reaction_role = ReactionRole.PRODUCT

    # Analysis
    analysis = outcome.analyses["LCMS"]
    analysis.type = Analysis.LCMS

    # Measurement
    measurement = product.measurements.add()
    define_measurement(measurement, row)


def add_provenance(reaction: Reaction):
    provenance = reaction.provenance
    provenance.doi = "110.1039/C6RE00153J"
    provenance.publication_url = "http://doi.org/10.1039/C6RE00153J."
    creator = provenance.record_created.person
    creator.username = "marcosfelt"
    creator.name = "Kobi Felton"
    creator.orcid = "0000-0002-3616-4766"
    creator.organization = "University of Cambridge"
    creator.email = "kobi.c.f@gmail.com"


if __name__ == "__main__":
    typer.run(main)
