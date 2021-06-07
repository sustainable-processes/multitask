"""
Create an ORD dataset from the the paper by Baumgartner et al.

python etl_baumgartner_suzuki.py ../data/baumgartner_suzuki/c8re00032h2.xlsx ../data/baumgartner_suzuki/

"""

from ord_schema.proto.reaction_pb2 import *
from ord_schema.proto.dataset_pb2 import *
from ord_schema.message_helpers import find_submessages

import typer
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
import numpy as np



def main(input_file: str, output_path:str, sheet_name: str="MINLP1 optimization"):
    """Entrypoint for running ETL job"""
    # Extract data
    df = pd.read_excel(input_file,sheet_name=sheet_name)

    # Transform
    tqdm.pandas(desc="Converting to ORD")
    reactions = df.progress_apply(inner_loop, axis=1)

    # Create dataset
    dataset = Dataset()
    dataset.name = "Baumgartner Suzuki Cross-Coupling"
    dataset.reactions.extend(reactions)
    dataset.dataset_id = "10453"

    # Load back
    case = sheet_name.replace(" ", "-").lower()
    output_path = Path(output_path)
    with open(output_path / f"baumgartner_suzuki-{case}.pb", "wb") as f:
        f.write(dataset.SerializeToString())


def inner_loop(row: pd.Series) -> Reaction:
    """Innter loop for creating a Reaction object for each row in the spreadsheet"""
    # Create reaction
    reaction = Reaction()
    reaction.identifiers.add(
        value=r"Suzuki Coupling", type=5
    )
    # Add reactants
    add_electrophile(reaction, row)
    add_nucleophile(reaction, row)
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
    specify_outcome(reaction, row)

    return reaction


def add_thf_solvent(
    stock: ReactionInput,
    final_solute_conc: float,
    stock_conc: float,
    droplet_volume: float=40.0,
)-> None:
    """
    final_solute_conc is in molar
    droplet volume in microliters
    stock_conc in molar
    """
    thf = stock.components.add()
    thf.identifiers.add(
        value="THF",
        type=CompoundIdentifier.NAME
    )
    thf.identifiers.add(
        value=r"C1CCOC1",
        type=CompoundIdentifier.SMILES
    )
    thf.reaction_role = ReactionRole.SOLVENT
    thf.amount.volume.units=Volume.MICROLITER
    thf.amount.volume.value=final_solute_conc*droplet_volume/stock_conc
    thf.amount.volume_includes_solutes=True

def add_electrophile(reaction: Reaction, row: pd.Series):
    # Chloropyridine
    chloropyridine_stock = reaction.inputs["Electrophile"]
    chloropyridine_stock.addition_order = 1

    # Reactant
    chloropyridine = chloropyridine_stock.components.add()
    chloropyridine.reaction_role = ReactionRole.REACTANT
    chloropyridine.identifiers.add(
        value="3-Chloropyridine",
        type=CompoundIdentifier.NAME
    )
    chloropyridine.identifiers.add(
        value=r"C1=CC(=CN=C1)Cl",
        type=CompoundIdentifier.SMILES
    )
    chloropyridine_conc = row["Reagent 1 Conc. (M)"]
    amount = chloropyridine.amount
    amount.moles.units = Moles.MICROMOLE
    amount.moles.value = chloropyridine_conc * 40.0  # 40 µL droplet
    chloropyridine.is_limiting = True

    # Internal standard
    internal_std = chloropyridine_stock.components.add()
    internal_std.reaction_role = ReactionRole.INTERNAL_STANDARD
    internal_std.identifiers.add(
        value="Naphthelene",
        type=CompoundIdentifier.NAME
    )
    internal_std.identifiers.add(
        value=r"c1c2ccccc2ccc1",
        type=CompoundIdentifier.SMILES
    )
    istd_conc = row["Internal Standard Conc. (g/L)"]
    amount = internal_std.amount
    amount.mass.units = Mass.MICROGRAM
    amount.mass.value = istd_conc * 40.0

    # Solvent
    add_thf_solvent(
        chloropyridine_stock,
        chloropyridine_conc,
        stock_conc=0.996
    )

def add_nucleophile(reaction: Reaction, row: pd.Series):
    # Pinacol ester
    pinacol_ester_stock = reaction.inputs["Nucleophile"]
    pinacol_ester_stock.addition_order = 1

    # Solute
    pinacol_ester = pinacol_ester_stock.components.add()
    pinacol_ester.reaction_role = ReactionRole.REACTANT
    pinacol_ester.identifiers.add(
        value="2-fluoropyridine-3-boronic acid pincacol ester",
        type=CompoundIdentifier.NAME
    )
    pinacol_ester.identifiers.add(
        value=r"CC1(C)OB(OC1(C)C)c2ccc(F)nc2",
        type=CompoundIdentifier.SMILES
    )
    pe_conc = row["Reagent 1 Conc. (M)"]
    amount = pinacol_ester.amount
    amount.moles.units = Moles.MICROMOLE
    amount.moles.value = pe_conc * 40.0  # 40 µL droplet

    # Solvent
    add_thf_solvent(
        pinacol_ester_stock,
        pe_conc,
        stock_conc=0.996
    )
pre_catalysts = {
    "P1": {
        "SMILES": "CS(=O)(=O)O[Pd]c1ccccc1-c2ccccc2N",
        "name": "Pd G3 µ-OMS"
    },
    "P2": {
        "SMILES": "Cl[Pd]c1ccccc1-c2ccccc2N",
        "name": "Pd G3 Cl"
    }
}

ligands = {
    "L1": {
        "SMILES": "CC(C)c1cc(C(C)C)c(c(c1)C(C)C)-c2ccccc2P(C3CCCCC3)C4CCCCC4",
        "name": "XPhos"
    },
    "L2": {
        "SMILES": "COc1cccc(OC)c1-c2ccccc2P(C3CCCCC3)C4CCCCC4",
        "name": "SPhos",
    },
    "L3": {
        "SMILES": "CC(C)Oc1cccc(OC(C)C)c1-c2ccccc2P(C3CCCCC3)C4CCCCC4",
        "name": "RuPhos"
    },
    "L4": {
        "SMILES": "CC1(C)c2cccc(P(c3ccccc3)c4ccccc4)c2Oc5c(cccc15)P(c6ccccc6)c7ccccc7",
        "name": "XantPhos"
    },
    "L5": {
        "SMILES": "C1CCC(CC1)P(C2CCCCC2)C3CCCCC3",
        "name": "tricyclohexylphosphine"
    },
    "L6": {
        "SMILES": "c1ccc(cc1)P(c2ccccc2)c3ccccc3",
        "name": "triphenylphosphine"
    },
    "L7": {
        "SMILES": "CC(C)(C)P(C(C)(C)C)C(C)(C)C",
        "name": "tri-tert-butylphosphine"
    }
}

def catalyst_details(pre_catalyst_id:str, ligand_id:str)-> (str, str):
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
    cat_id = row["Reagent 3 ID"]
    name, smiles = catalyst_details(cat_id[:2], cat_id[2:4])
    catalyst.identifiers.add(
        value=name,
        type=CompoundIdentifier.NAME
    )
    catalyst.identifiers.add(
        value=smiles,
        type=CompoundIdentifier.SMILES
    )
    cat_conc = row["Reagent 3 Conc. (M)"]
    catalyst.amount.moles.units = Moles.MICROMOLE
    catalyst.amount.moles.value = cat_conc * 40.0

    # Solvent
    add_thf_solvent(
        catalyst_stock,
        cat_conc,
        stock_conc=0.017
    )

def add_solvent(reaction: Reaction, row: pd.Series):
    # Solvent
    solvent_mix = reaction.inputs["Solvent"]
    solvent_mix.addition_order = 1

    # Calculate volume of solvent needed for 40 microliter droplet
    reactants = [
        reaction.inputs[v]
        for v in
        ["Electrophile", "Nucleophile", "Catalyst"]
    ]
    reactants_volume = sum([r.components[1].amount.volume.value for r in reactants])
    solvent_volume = 40.0 - reactants_volume
    thf_volume = 5 / 6 * solvent_volume
    water_volume = 1 / 6 * solvent_volume

    # THF
    thf = solvent_mix.components.add()
    thf.identifiers.add(
        value="THF",
        type=CompoundIdentifier.NAME
    )
    thf.identifiers.add(
        value=r"C1CCOC1",
        type=CompoundIdentifier.SMILES
    )
    thf.reaction_role = ReactionRole.SOLVENT
    thf.amount.volume.value = thf_volume
    thf.amount.volume.units = Volume.MICROLITER

    # Water
    water = solvent_mix.components.add()
    water.identifiers.add(
        value="water",
        type=CompoundIdentifier.NAME
    )
    water.identifiers.add(
        value="O",
        type=CompoundIdentifier.SMILES
    )
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
    base.identifiers.add(
        value="DBU",
        type=CompoundIdentifier.NAME
    )
    base.identifiers.add(
        value=r"N\2=C1\N(CCCCC1)CCC/2",
        type=CompoundIdentifier.SMILES
    )
    base.amount.moles.value = row["Inlet Injection (µL)"] * 1.645
    base.amount.moles.units = Moles.MICROMOLE

    # Solvent
    thf = base_stock.components.add()
    thf.identifiers.add(
        value="THF",
        type=CompoundIdentifier.NAME
    )
    thf.identifiers.add(
        value=r"C1CCOC1",
        type=CompoundIdentifier.SMILES
    )
    thf.reaction_role = ReactionRole.SOLVENT
    thf.amount.volume.value = row["Inlet Injection (µL)"]
    thf.amount.volume.units = Volume.MICROLITER

def specify_temperature(reaction: Reaction, row: pd.Series):
    # Temperature
    temp_conditions = reaction.conditions.temperature
    details = "Oscillatory flow reactor with two cartridge heaters and a thermocouple"
    control = temp_conditions.control
    control.type = TemperatureConditions.TemperatureControl.DRY_ALUMINUM_PLATE
    control.details = details
    temp_conditions.setpoint.value = row["Temperature (°C)"]
    temp_conditions.setpoint.units = Temperature.TemperatureUnit.CELSIUS

def specify_flow_conditions(reaction: Reaction, row: pd.Series):
    flow_conditions = reaction.conditions.flow
    flow_conditions.type = FlowConditions.CUSTOM
    flow_conditions.details = (
        "A droplet flow reactor system consisting of a liquid handler"
        " and oscillatory flow reactor. Each droplet is like a mini-batch reactor."
    )
    flow_conditions.tubing.type = FlowConditions.Tubing.PFA
    flow_conditions.tubing.diameter.value = 1 / 16
    flow_conditions.tubing.details = """1/16 in. ID tubing for main reactor."""
    flow_conditions.tubing.diameter.units = Length.INCH

def calculate_total_volume(reaction: Reaction):
    # Calculate total reaction volume
    reactants = [
        reaction.inputs[v]
        for v in
        ["Electrophile", "Nucleophile", "Catalyst", "Base"]
    ]
    reaction_volume = sum([r.components[1].amount.volume.value for r in reactants])
    solvent_mix = reaction.inputs["Solvent"]
    reaction_volume += sum([s.value for s in find_submessages(solvent_mix, Volume)])
    return reaction_volume

def cross_checks(reaction: Reaction, row: pd.Series):
    # Check that reaction volume adds up properly
    vol = calculate_total_volume(reaction)
    try:
        assert np.isclose(vol, 43.5, rtol=1e-2)
    except AssertionError:
        raise ValueError(
            f"Total volume expected to be 43.5 µL but it is actually {vol}µL."
        )

    # Cross-check concentration calculations using catalyst mol%
    mol_electrophile = reaction.inputs["Electrophile"].components[0].amount.moles.value
    cat_mols_check = mol_electrophile * row["Reagent 3 Conc in mol%"]
    catalyst = reaction.inputs["Catalyst"].components[0]
    try:
        assert np.isclose(catalyst.amount.moles.value, cat_mols_check)
    except AssertionError:
        raise ValueError(
            f"Inconsistent amounts. Catalyst should be: {cat_mols_check} micromols"
            f", but it is actually {catalyst.amount.moles.value} micromols."
        )

def quench_reaction(reaction: Reaction, row: pd.Series):
    # Quench
    quench = reaction.inputs["Quench"]
    quench.addition_order = 3
    reaction_volume = calculate_total_volume(reaction)

    # Acetone
    acetone = quench.components.add()
    acetone.reaction_role = ReactionRole.WORKUP
    acetone.identifiers.add(
        value="acetone",
        type=CompoundIdentifier.NAME
    )
    acetone.identifiers.add(
        value="CC(=O)C",
        type=CompoundIdentifier.SMILES
    )
    acetone.amount.volume.value = 0.5 * reaction_volume
    acetone.amount.volume.value = Volume.MICROLITER

    # Water
    water = quench.components.add()
    water.reaction_role = ReactionRole.WORKUP
    water.identifiers.add(
        value="water",
        type=CompoundIdentifier.NAME
    )
    water.identifiers.add(
        value="O",
        type=CompoundIdentifier.SMILES
    )
    water.amount.volume.value = 0.5 * reaction_volume
    water.amount.volume.units = Volume.MICROLITER

    # Workup specification
    workup = reaction.workups.add()
    workup.amount.volume.value = reaction_volume
    workup.amount.volume.units = Volume.MICROLITER
    workup.type = ReactionWorkup.ADDITION
    details = (
        "As soon as the slug leaves the reactor and is detected at PS2,"
        "a quench solution volume equal to the reaction slug volume"
        "(prepared slug volume + base injection volume) is injected into the reaction slug"
    )
    workup.details = details
    workup.input.CopyFrom(quench)

def add_standard(measurement: ProductMeasurement):
    # Standard
    measurement.uses_internal_standard = True
    measurement.uses_authentic_standard = True
    standard = measurement.authentic_standard
    standard.identifiers.add(
        value="2'-fluoro-2,3'-bipyridine",
        type=CompoundIdentifier.NAME
    )
    standard.identifiers.add(
        value="FC1=C(C2=NC=CC=C2)C=CC=N1",
        type=CompoundIdentifier.SMILES
    )
    standard.reaction_role = ReactionRole.AUTHENTIC_STANDARD
    standard_prep = standard.preparations.add()
    standard_prep.type = CompoundPreparation.SYNTHESIZED
    details = (
        "The product 2-fluoro-3,3’-bipyridine 11 was synthesized in batch following the procedure of Reizman et al. "
        "A magnetic stir bar, SPhos Pd G2 (72 mg, 0.10 mmol, 0.05 equiv.), THF (8 mL) and water (2 mL) was added "
        "to a dry and nitrogen-filled 20-mL septum vial under nitrogen atmosphere. Using syringes, 3-chloropyridine 9 "
        "(190 µL, 2.0 mmol, 1.0 equiv.) and DBU (598 µL, 4.0 mmol, 2.0 equiv.) were added sequentially. The reaction"
        "mixture was then heated to 65 ◦C, followed by addition of the THF (1 mL) solution of 2-fluoropyridine-3-"
        "boronic acid pincol ester 10 (669 mg, 3.0 mmol, 1.5 equiv.). The reaction mixture was allowed to stir overnight."
        " The next day, the reaction was diluted with ethyl acetate, washed with brine and dried over Na2SO4"
        " , filtered and concentrated under reduced pressure. The resulting residue was then purifed by flash column chromatography"
        "(ethyl acetate/heptane = 1:1) to afford the desired product 11 (330 mg, 95% yield) as a white solid. The purity"
        "was confirmed with LC/MS (m/z = 174.06, Fig. 6) and NMR (Fig. 7 and Fig. 8). 1H NMR (400 MHz, CDCl3)"
        "δ 8.86-8.84 (m, 1H), 8.70 (dt, J = 4.8, 1.2 Hz, 1H), 8.32-8.30 (m, 1H), 7.99-7.94 (m, 1H), 7.93-7.91 (m, 1H),"
        "7.50-7.46 (m, 1H), 7.37 (ddt, J = 7.3, 4.9, 1.3 Hz, 1H)."
    )
    standard_prep.details = details

def define_measurement(measurement: ProductMeasurement, row: pd.Series):
    measurement.type = ProductMeasurement.YIELD
    measurement.percentage.value = row["Reaction Yield"]
    measurement.retention_time.value = row["2-Fluoro-3,3'-bipyridine Retention time in min"]
    measurement.retention_time.units = Time.MINUTE


def specify_outcome(reaction: Reaction, row: pd.Series):
    # Reaction Outcome
    outcome = reaction.outcomes.add()

    # Time
    outcome.reaction_time.value = row["Residence Time Actual (s)"]
    outcome.reaction_time.units = Time.SECOND

    # TODO: conversion calculation

    # Product
    product = outcome.products.add()
    product.identifiers.add(
        value="2'-fluoro-2,3'-bipyridine",
        type=CompoundIdentifier.NAME
    )
    product.identifiers.add(
        value="FC1=C(C2=NC=CC=C2)C=CC=N1",
        type=CompoundIdentifier.SMILES
    )
    product.is_desired_product = True
    product.reaction_role = ReactionRole.PRODUCT

    # Measurement
    measurement = product.measurements.add()
    define_measurement(measurement, row)
    add_standard(measurement)

def add_provenance(reaction: Reaction):
    provenance = reaction.provenance
    provenance.doi = "10.1039/c8re00032h"
    provenance.publication_url = "http://doi.org/10.1039/c8re00032h"
    creator = provenance.record_created.person
    creator.username = "maracosfelt"
    creator.name = "Kobi Felton"
    creator.orcid = "0000-0002-3616-4766"
    creator.organization = "University of Cambridge"
    creator.email = "kobi.c.f@gmail.com"

if __name__ == "__main__":
    typer.run(main)




