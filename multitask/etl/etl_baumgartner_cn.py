"""
Create an ORD dataset from the the paper by Baumgartner et al.

python etl_baumgartner_C-N.py ../data/baumgartner_C-N/op9b00236_si_002.xlsx ../data/baumgartner_C-N/

NB 1: First 36 and last 10 reactions don't have a corresponding figure in the paper and are 
performed at equivalence ratios different to the others (perhaps they were used as prescreening?). 
The data for these 46 reactions will still be recorded in the .pb file.

NB 2: droplet_volume = row["Quench Outlet Injection (uL)"]
    droplet volume = quench volume because "The injected solvent volume was set equal 
    to that of the reaction droplet, resulting in a 1:1 dilution" (page 1599)

NB 3: There seems to be a mistake in the spreadsheet.
    The 'Optimization' column is in the format '{nucleophile} - {precatalyst}'
    The 'N-H nucleophile' column is in the format '{nucleophile}'
    The nucleophile in the Optimization and N-H nucleophile should thus be the same (and indeed 
    they are for almost all rows), however in rows 252-285 the optimization nucleophile is 
    Phenethylamine, while the N-H nucleophile is Benzamide. I trust the Optimization column to be correct.

NB 4: Twice in the spreadsheet does "TEA" appear as "Triethylamine"

NB 5: Row 368 and 371 are part of the "Morpholine - tBuBrettPhos (Preliminary)" campaign. In these rows, the base specified is 'Triethylamine'.
    However, in the 'Stock solutions' sheet, there is no stock solution for 'Triethylamine' with 'Morpholine - tBuBrettPhos'. I noticed that the
    stock solution of TEA was constant with all campaigns (around 7.17), so I set the stock solution conc for these two rows to the same number.

NB 6: There are warnings about some reaction yields being very low (e.g. 0.27% yield, validation script thinks this is a yield mistakenly written
    as a decimal rather than percentage) and very high (ie. above 100%) however this is what's actually in the dataset!

NB 7: No stock concentrations specified for the  precatalyst/catalyst

NB 8: It seems I'm not allowed to specify an internal standard within the add_standard function

"""

from ord_schema.proto.reaction_pb2 import *
from ord_schema.proto.dataset_pb2 import *
from ord_schema import validations
import typer
from pint import UnitRegistry
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import json

ureg = UnitRegistry()


def main(
    input_file: str,
    output_path: str,
    rxn_sheet_name="Reaction data",
    stock_sheet_name="Stock solutions",
    drop_preliminary=True,
):
    """Extracts the Baumgartner C-N Excel file (input_file) and saves as ORD protobuf (output_file)."""
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    # Extract data
    df_all_rxn_data = pd.read_excel(input_file, sheet_name=rxn_sheet_name)
    df_stock_solutions = pd.read_excel(input_file, sheet_name=stock_sheet_name)
    cat_nucleophile = df_all_rxn_data["Optimization"].str.split(" - ", expand=True)
    df_all_rxn_data["nucleophile"] = cat_nucleophile[0]
    if drop_preliminary:
        df_all_rxn_data["Campaign/Figure reaction number  "] = df_all_rxn_data[
            "Campaign/Figure reaction number  "
        ].replace("-", None)
        df_all_rxn_data = df_all_rxn_data.dropna(
            subset="Campaign/Figure reaction number  "
        )

    # Transform
    for i, (_, df_rxn_data) in enumerate(df_all_rxn_data.groupby("nucleophile")):
        tqdm.pandas(desc="Converting to ORD")
        reactions = df_rxn_data.progress_apply(
            inner_loop, axis=1, args=(df_stock_solutions,)
        )

        for j, reaction in enumerate(reactions):
            reaction.reaction_id = str(j)

        # Create dataset
        dataset = Dataset()
        dataset.name = "Baumgartner C-N Cross-Coupling"
        dataset.reactions.extend(reactions)
        dataset.dataset_id = str(1)

        # Validate dataset
        validation_output = validations.validate_message(dataset)
        print(
            f"First 5 Warnings. See {output_path / f'warnings_baumgartner_cn_case_{i+1}.json'} for full output"
        )
        print(validation_output.warnings[:5])
        with open(output_path / f"warnings_baumgartner_cn_case_{i+1}.json", "w") as f:
            json.dump(validation_output.warnings, f)

        # Load back
        with open(output_path / f"baumgartner_cn_case_{i+1}.pb", "wb") as f:
            f.write(dataset.SerializeToString())


def inner_loop(row: pd.Series, stock_df: pd.DataFrame) -> Reaction:
    """Innter loop for creating a Reaction object for each row in the spreadsheet"""
    # Create reaction
    reaction = Reaction()
    reaction.identifiers.add(value=r"C-N Coupling", type=5)
    # Add reactants
    add_electrophile(reaction, row, stock_df)
    add_nucleophile(reaction, row, stock_df)
    add_catalyst(reaction, row)
    add_base(reaction, row, stock_df)
    add_solvent(reaction, row)

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


nucleophile_to_case = {
    1: "Aniline",
    2: "Benzamide",
    3: "Benzylamine",
    4: "Morpholine",
}

nucleophiles = {
    "Aniline": {"smiles": "c1ccc(cc1)N", "name": "Aniline"},
    "Benzamide": {"smiles": "c1ccc(cc1)C(=O)N", "name": "Benzamide"},
    "Phenethylamine": {"smiles": "NCCc1ccccc1", "name": "Phenethylamine"},
    "Morpholine": {"smiles": "C1CNCCO1", "name": "Morpholine"},
}


ligands = {
    # "EPhos": {
    #     "smiles": "CC(C)C1=CC(=C(C(=C1)C(C)C)C2=C(C(=CC=C2)OC(C)C)P(C3CCCCC3)C4CCCCC4)C(C)C",
    #     "name": "EPhos",
    # },
    "tBuXPhos": {
        "smiles": "CC(C)C1=CC(=C(C(=C1)C(C)C)C2=CC=CC=C2P(C(C)(C)C)C(C)(C)C)C(C)C",
        "name": "tBuXPhos",
    },
    "tBuBrettPhos": {
        "smiles": "CC(C)C1=CC(=C(C(=C1)C(C)C)C2=C(C=CC(=C2P(C(C)(C)C)C(C)(C)C)OC)OC)C(C)C",
        "name": "tBuBrettPhos",
    },
    "AlPhos": {
        "smiles": "CCCCC1=C(C(=C(C(=C1F)F)C2=C(C=C(C(=C2C(C)C)C3=C(C(=CC=C3)OC)P(C45CC6CC(C4)CC(C6)C5)C78CC9CC(C7)CC(C9)C8)C(C)C)C(C)C)F)F",
        "name": "AlPhos",
    },
}

solvents = {
    "2-MeTHF": {"smiles": "O1C(C)CCC1", "name": "2-MeTHF"},
    "DMSO": {"smiles": "CS(C)=O", "name": "DMSO"},
}

bases = {
    "TEA": {"smiles": "CCN(CC)CC", "name": "TEA"},
    "TMG": {"smiles": "CN(C)C(=N)N(C)C", "name": "TMG"},
    "BTMG": {"smiles": "CC(C)(C)N=C(N(C)C)N(C)C", "name": "BTMG"},
    "DBU": {"smiles": "C1CCC2=NCCCN2CC1", "name": "DBU"},
    "MTBD": {"smiles": "CN1CCCN2C1=NCCC2", "name": "MTBD"},
    "BTTP": {"smiles": "CC(C)(C)N=P(N1CCCC1)(N2CCCC2)N3CCCC3", "name": "BTTP"},
    "P2Et": {"smiles": "CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C", "name": "P2Et"},
}

# the key for the products is actually the N-H nucleophile,
# however, since the electrophile is always the same, we can
# use the nucleophile as key to link the the appropriate product
products = {
    "Aniline": {
        "smiles": "CC1=CC=C(C=C1)NC2=CC=CC=C2",
        "name": "4-methyl-N-phenylaniline",
    },
    "Benzamide": {
        "smiles": "CC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2",
        "name": "N-(4-Methylphenyl)benzamide",
    },
    "Phenethylamine": {
        "smiles": "CC1=CC=C(C=C1)NCCC2=CC=CC=C2",
        "name": "4-methyl-N-phenethylaniline",
    },
    "Morpholine": {"smiles": "CC1=CC=C(C=C1)N2CCOCC2", "name": "4-(p-tolyl)morpholine"},
}


def stock_concentration(Reagent_Name: str, row: pd.Series, stock_df) -> (float):
    opt_run = row["Optimization"]  # Optimization run
    nuc_id, precat_id = opt_run.split(" - ")  # cat_id_nucleophile, precatalyst_id
    precat_id = precat_id.replace(" (Preliminary)", "")

    Substrate = nuc_id
    Precatalyst = precat_id

    # Please see header for explanation
    if Reagent_Name == "Triethylamine":
        return 7.17462199822117

    stock_df = stock_df.loc[stock_df["Substrate / campaign"].isin([Substrate])]
    stock_df = stock_df.loc[stock_df["Precatalyst"].isin([Precatalyst])]
    if Reagent_Name == "Aryl triflate":
        stock_df = stock_df.iloc[0]
    elif True:
        stock_df = stock_df.loc[stock_df["Reagent Name"].isin([Reagent_Name])]

    return stock_df["Reagent Conc (M)"]


def solvent_details(solvent_id: str):
    sol = solvents[solvent_id]
    smiles = sol["smiles"]
    name = sol["name"]
    return name, smiles


def specify_solvent(
    stock: ReactionInput,
    row: pd.Series,
    final_solute_conc: float,
    stock_conc: float,
) -> None:
    """
    This function isn't being called anywhere anymore.
    final_solute_conc is in molar
    droplet volume in microliters
    stock_conc in molar
    """
    droplet_volume = row["Quench Outlet Injection (uL)"]

    # do we need these lines?
    # solvent = reaction.inputs["Solvent"]
    # solvent.addition_order = 1

    solvent = stock.components.add()
    solvent.reaction_role = ReactionRole.SOLVENT
    sol_id = row["Make-Up Solvent ID"]
    name, smiles = solvent_details(sol_id)
    solvent.identifiers.add(value=name, type=CompoundIdentifier.NAME)
    solvent.identifiers.add(value=smiles, type=CompoundIdentifier.SMILES)

    solvent.amount.volume.units = Volume.MICROLITER
    solvent.amount.volume.value = final_solute_conc * droplet_volume / stock_conc

    solvent.amount.volume_includes_solutes = True

    # # Alternative way of calculating solvent volume
    # reactants_volume = calculate_total_volume(reaction, include_workup=False)
    # solvent.amount.volume.value = droplet_volume - reactants_volume


def add_electrophile(reaction: Reaction, row: pd.Series, stock_df):
    droplet_volume = row["Quench Outlet Injection (uL)"]
    # p-Tolyl triflate
    pTTf_stock = reaction.inputs["Electrophile"]
    pTTf_stock.addition_order = 1

    # Stock concentration
    Reagent_Name = "Aryl triflate"
    pTTf_stock_conc = stock_concentration(Reagent_Name, row, stock_df)

    # Reactant
    pTTf = pTTf_stock.components.add()
    pTTf.reaction_role = ReactionRole.REACTANT
    pTTf.identifiers.add(value="p-Tolyl triflate", type=CompoundIdentifier.NAME)
    pTTf.identifiers.add(
        value="CC1=CC=C(C=C1)OS(=O)(=O)C(F)(F)F", type=CompoundIdentifier.SMILES
    )
    pTTf_conc = row["Aryl triflate concentration (M)"]
    pTTf.amount.moles.units = Moles.MICROMOLE
    pTTf.amount.moles.value = pTTf_conc * droplet_volume
    pTTf.is_limiting = True

    # Internal standard
    FNaph_stock = reaction.inputs["Internal_Standard"]  # 1-Fluoronaphtalene
    internal_std = FNaph_stock.components.add()
    internal_std.reaction_role = ReactionRole.INTERNAL_STANDARD

    internal_std.identifiers.add(
        value="1-fluoronaphthalene", type=CompoundIdentifier.NAME
    )
    internal_std.identifiers.add(
        value="C1=CC=C2C(=C1)C=CC=C2F", type=CompoundIdentifier.SMILES
    )
    istd_conc = row["Internal Standard Concentration 1-fluoronaphthalene (g/L)"]
    amount = internal_std.amount
    amount.mass.units = Mass.MICROGRAM
    amount.mass.value = istd_conc * droplet_volume

    # Solvent
    # specify_solvent(pTTf_stock, row, pTTf_conc, pTTf_stock_conc)


def nucleophile_details(nucleophile_id: str):
    nuc = nucleophiles[nucleophile_id]
    smiles = nuc["smiles"]
    name = nuc["name"]
    return name, smiles


def add_nucleophile(reaction: Reaction, row: pd.Series, stock_df):
    droplet_volume = row["Quench Outlet Injection (uL)"]
    # nucleophiles
    nucleophile_stock = reaction.inputs["Nucleophile"]
    nucleophile_stock.addition_order = 1

    # Solute
    nucleophile = nucleophile_stock.components.add()
    nucleophile.reaction_role = ReactionRole.REACTANT

    opt_run = row["Optimization"]  # Optimization run
    nuc_id, precat_id = opt_run.split(" - ")  # cat_id_nucleophile, precatalyst_id
    precat_id = precat_id.replace(" (Preliminary)", "")

    name, smiles = nucleophile_details(nuc_id)
    nucleophile.identifiers.add(value=name, type=CompoundIdentifier.NAME)
    nucleophile.identifiers.add(value=smiles, type=CompoundIdentifier.SMILES)
    nuc_conc = row["N-H nucleophile concentration (M)"]
    nucleophile.amount.moles.units = Moles.MICROMOLE
    nucleophile.amount.moles.value = nuc_conc * droplet_volume

    # Stock concentration
    nuc_stock_conc = stock_concentration(nuc_id, row, stock_df)

    # Solvent
    # specify_solvent(nucleophile_stock, row, nuc_conc, nuc_stock_conc)


def catalyst_details(ligand_id: str):
    # https://www.strem.com/catalog/v/46-0308/51/palladium_225931-80-6
    pre_cat = {"smiles": "C1CC=CCCC=C1.C[Si](C)(C)C[Pd]C[Si](C)(C)C", "name": "cycloPd"}
    additive = {"smiles": "CC1=CC=C(C=C1)Cl", "name": "4-Chlorotoluene"}
    ligand = ligands[ligand_id]
    smiles = f"""{pre_cat["smiles"]}.{ligand["smiles"]}.{additive["smiles"]}"""
    name = pre_cat["name"] + " " + ligand["name"] + " " + additive["name"]
    return name, smiles


def add_catalyst(reaction: Reaction, row: pd.Series):
    # Catalyst
    catalyst_stock = reaction.inputs["Catalyst"]
    catalyst_stock.addition_order = 1

    # Solute
    catalyst = catalyst_stock.components.add()
    catalyst.reaction_role = ReactionRole.CATALYST
    opt_run = row["Optimization"]  # optimization run
    nuc_id, precat_id = opt_run.split(" - ")  # nucleophile_id, precatalyst_id
    precat_id = precat_id.replace(" (Preliminary)", "")
    name, smiles = catalyst_details(precat_id)
    catalyst.identifiers.add(value=name, type=CompoundIdentifier.NAME)
    catalyst.identifiers.add(value=smiles, type=CompoundIdentifier.SMILES)
    cat_mol_percent = row["Precatalyst loading in mol%"]
    aryl_triflate_conc = row["Aryl triflate concentration (M)"]
    droplet_volume = row["Quench Outlet Injection (uL)"]
    catalyst.amount.moles.units = Moles.MICROMOLE
    # cat_mol = cat_mol% * moles of limiting reagent (aryl triflate)
    # moles of aryl triflate = aryl_triflate_conc * droplet volume
    catalyst.amount.moles.value = cat_mol_percent * aryl_triflate_conc * droplet_volume

    # Solvent
    # There's no stock solution of precatalyst?
    # specify_solvent(catalyst_stock, row, cat_mol_percent, stock_conc=??)


def add_solvent(reaction: Reaction, row: pd.Series):
    injection_volume = row["N-H nucleophile Inlet Injection (uL)"]
    droplet_volume = row["Quench Outlet Injection (uL)"]
    total_volume = injection_volume + droplet_volume
    # Solvent
    solvent_stock = reaction.inputs["Solvent"]
    solvent_stock.addition_order = 1

    # Calculate volume of solvent needed
    reactants_volume = calculate_total_volume(reaction, include_workup=False)
    solvent_volume = total_volume - reactants_volume

    # specify solvent
    solvent = solvent_stock.components.add()
    solvent.reaction_role = ReactionRole.SOLVENT
    sol_id = row["Make-Up Solvent ID"]
    name, smiles = solvent_details(sol_id)
    solvent.identifiers.add(value=name, type=CompoundIdentifier.NAME)
    solvent.identifiers.add(value=smiles, type=CompoundIdentifier.SMILES)

    solvent.amount.volume.units = Volume.MICROLITER
    solvent.amount.volume.value = solvent_volume
    # import pdb
    # pdb.set_trace()


def base_details(base_id: str):
    base = bases[base_id]
    smiles = base["smiles"]
    name = base["name"]
    return name, smiles


def add_base(reaction: Reaction, row: pd.Series, stock_df):
    droplet_volume = row["Quench Outlet Injection (uL)"]
    # Base
    base_stock = reaction.inputs["Base"]
    base_stock.addition_order = 1

    # Solute
    base = base_stock.components.add()
    base.reaction_role = ReactionRole.REAGENT
    base_id = row["Base"]
    name, smiles = base_details(base_id)
    base.identifiers.add(value=name, type=CompoundIdentifier.NAME)
    base.identifiers.add(value=smiles, type=CompoundIdentifier.SMILES)
    base_conc = row["Base concentration (M)"]
    base.amount.moles.value = base_conc * droplet_volume
    base.amount.moles.units = Moles.MICROMOLE

    base_stock_conc = stock_concentration(base_id, row, stock_df)

    # specify_solvent(base_stock, row, base_conc, base_stock_conc)


def specify_temperature(reaction: Reaction, row: pd.Series):
    # Temperature
    temp_conditions = reaction.conditions.temperature
    details = """This oscillatory reactor was housed in a
                thermocouple-controlled block that was heated by electrical
                cartridge heaters or cooled by a fan between runs"""
    control = temp_conditions.control
    control.type = TemperatureConditions.TemperatureControl.DRY_ALUMINUM_PLATE
    control.details = details
    temp_conditions.setpoint.value = row["Temperature (degC)"]
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


def get_pint(amount: Amount):
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


def calculate_total_volume(reaction, include_workup=False):
    # Calculate total reaction volume in microliters
    total_volume = 0.0 * ureg.ml
    rxn_inputs = reaction.inputs
    for k in rxn_inputs:
        for component in rxn_inputs[k].components:
            amount = component.amount
            include_workup = (
                True
                if include_workup
                else component.reaction_role != ReactionRole.WORKUP
            )
            if amount.WhichOneof("kind") == "volume" and include_workup:
                total_volume += get_pint(amount)

    return total_volume.to(ureg.microliter).magnitude


def cross_checks(reaction: Reaction, row: pd.Series):
    # Check that reaction volume adds up properly

    injection_volume = row["N-H nucleophile Inlet Injection (uL)"]
    droplet_volume = row["Quench Outlet Injection (uL)"]
    expected_vol = injection_volume + droplet_volume
    vol = calculate_total_volume(reaction)
    try:
        assert np.isclose(vol, expected_vol, rtol=1e-2)
    except AssertionError:
        raise ValueError(
            f"Total volume expected to be {expected_vol}µL but it is actually {vol}µL."
        )

    # Cross-check concentration calculations using catalyst mol%
    mol_electrophile = reaction.inputs["Electrophile"].components[0].amount.moles.value
    cat_mols_check = mol_electrophile * row["Precatalyst loading in mol%"]
    catalyst = reaction.inputs["Catalyst"].components[0]
    try:
        assert np.isclose(catalyst.amount.moles.value, cat_mols_check)
    except AssertionError:
        raise ValueError(
            f"Inconsistent amounts. Catalyst should be: {cat_mols_check} micromols"
            f", but it is actually {catalyst.amount.moles.value} micromols."
        )


def quench_reaction(reaction: Reaction, row: pd.Series):
    """
    The injected solvent volume was set equal to that of the
    reaction droplet, resulting in a 1:1 dilution. We assumed
    that this dilution and cooling effectively quenched the reaction.
    """
    # Quench
    quench = reaction.inputs["Quench"]
    quench.addition_order = 2
    reaction_volume = calculate_total_volume(reaction)

    # solvent is used to quench
    quench_solvent = quench.components.add()
    quench_solvent.reaction_role = ReactionRole.WORKUP
    quench_solvent_id = row["Make-Up Solvent ID"]
    name, smiles = solvent_details(quench_solvent_id)
    quench_solvent.identifiers.add(value=name, type=CompoundIdentifier.NAME)
    quench_solvent.identifiers.add(value=smiles, type=CompoundIdentifier.SMILES)
    quench_solvent.amount.volume.value = reaction_volume
    quench_solvent.amount.volume.units = Volume.MICROLITER

    # Workup specification
    workup = reaction.workups.add()
    workup.amount.volume.value = reaction_volume * 2
    workup.amount.volume.units = Volume.MICROLITER
    workup.type = ReactionWorkup.CUSTOM
    details = (
        "As soon as the slug leaves the reactor and is detected at PS2,"
        "the quenching solvent of volume equal to the reaction slug volume"
        "(prepared slug volume + base injection volume) is injected into the reaction slug"
    )
    workup.details = details
    workup.input.CopyFrom(quench)


def add_standard(measurement: ProductMeasurement, row: pd.Series):
    # Standard
    # int_standard: internal standard
    # auth_standard: authentic standard
    measurement.uses_internal_standard = True
    measurement.uses_authentic_standard = True

    # # Seems I'm not allowed to specify an internal standard within this function
    # int_standard = measurement.internal_standard
    # int_standard.identifiers.add(
    #     value="1-fluoronaphthalene", type=CompoundIdentifier.NAME
    # )
    # int_standard.identifiers.add(
    #     value="C1=CC=C2C(=C1)C=CC=C2F", type=CompoundIdentifier.SMILES
    # )
    # int_standard.reaction_role = ReactionRole.INTERNAL_STANDARD
    # int_standard_prep = standard.preparations.add()
    # # it seems that the "CompoundPreparation" has to be "CompoundPreparation.SYNTHESIZED" from an error message
    # # I got in terminal when writing "CompoundPreparation.PURCHASED"
    # int_standard_prep.type = CompoundPreparation.SYNTHESIZED
    # int_details = (
    #     """
    #     It is not specified where 1-fluoronaphthalene came from.
    #     Probably purchased.
    #     """
    # )
    # int_standard_prep.details = int_details

    # Authentic standard is the product of the experiment
    auth_standard = measurement.authentic_standard
    # Product

    opt_run = row["Optimization"]  # Optimization run
    nuc_id, precat_id = opt_run.split(" - ")  # cat_id_nucleophile, precatalyst_id
    precat_id = precat_id.replace(" (Preliminary)", "")

    name, smiles = specify_outcome_details(nuc_id)
    auth_standard.identifiers.add(value=name, type=CompoundIdentifier.NAME)
    auth_standard.identifiers.add(value=smiles, type=CompoundIdentifier.SMILES)
    auth_standard.reaction_role = ReactionRole.AUTHENTIC_STANDARD
    auth_standard_prep = auth_standard.preparations.add()
    auth_standard_prep.type = CompoundPreparation.SYNTHESIZED
    auth_details = """
        S2: Purchased
        S1, S3-S5: According to literature precedent
        """
    auth_standard_prep.details = auth_details


def define_measurement(measurement: ProductMeasurement, row: pd.Series):
    measurement.analysis_key = "LCMS"
    measurement.type = ProductMeasurement.YIELD
    rxn_yield = row["Reaction Yield"]

    try:
        rxn_yield = rxn_yield.replace("≥", "")
        rxn_yield = rxn_yield.replace("%", "")
        rxn_yield = float(rxn_yield)
        rxn_yield = rxn_yield / 100
    except:
        pass
    # try:
    #     rxn_yield = rxn_yield.replace("≥", "")
    # except:
    #     pass
    # try:
    #     rxn_yield = rxn_yield.replace("%", "")
    # except:
    #     pass
    # try:
    #     rxn_yield = float(rxn_yield)
    # except:
    #     pass
    if rxn_yield < 200:
        measurement.percentage.value = rxn_yield * 100

    # measurement.retention_time.value = row[
    #    "2-Fluoro-3,3'-bipyridine Retention time in min"
    # ]
    # measurement.retention_time.units = Time.MINUTE


# nucleophile ID must be given. Since electrophile stays constant, the product
# can be inferred from the nucleophile alone
def specify_outcome_details(nuc_id: str):
    prod = products[nuc_id]
    smiles = prod["smiles"]
    name = prod["name"]
    return name, smiles


def specify_outcome(reaction: Reaction, row: pd.Series):
    # Reaction Outcome
    outcome = reaction.outcomes.add()

    # Time
    outcome.reaction_time.value = row["Residence Time Actual (s)"]
    outcome.reaction_time.units = Time.SECOND

    ## TODO: conversion calculation

    # Product
    product = outcome.products.add()

    opt_run = row["Optimization"]  # Optimization run
    nuc_id, precat_id = opt_run.split(" - ")  # cat_id_nucleophile, precatalyst_id
    precat_id = precat_id.replace(" (Preliminary)", "")

    name, smiles = specify_outcome_details(nuc_id)
    product.identifiers.add(value=name, type=CompoundIdentifier.NAME)
    product.identifiers.add(value=smiles, type=CompoundIdentifier.SMILES)
    product.is_desired_product = True
    product.reaction_role = ReactionRole.PRODUCT

    # Analysis
    analysis = outcome.analyses["LCMS"]
    analysis.type = Analysis.LCMS

    # Measurement
    measurement = product.measurements.add()
    define_measurement(measurement, row)
    add_standard(measurement, row)


def add_provenance(reaction: Reaction):
    provenance = reaction.provenance
    provenance.doi = "10.1021/acs.oprd.9b00236"
    provenance.publication_url = "http://doi.org/10.1021/acs.oprd.9b00236"
    creator = provenance.record_created.person
    creator.username = "dswigh"
    creator.name = "Daniel Wigh"
    creator.orcid = "0000-0002-0494-643X"
    creator.organization = "University of Cambridge"
    creator.email = "dswigh@gmail.com"


if __name__ == "__main__":
    typer.run(main)
