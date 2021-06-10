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

__all__ = ["get_suzuki_datasets", "suzuki_reaction_to_dataframe", "prepare_domain_data"]

app = typer.Typer()


@app.command()
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
    # Get data
    df, domain = prepare_domain_data(
        data_path=data_path,
        include_reactant_concentrations=include_reactant_concentrations,
        split_catalyst=split_catalyst,
        print_warnings=print_warnings,
    )

    if dataset_name is None:
        dataset_name = Path(data_path).parts[-1].rstrip(".pb")

    # Create emulator benchmark
    emulator = SuzukiEmulator(
        dataset_name, domain, dataset=df, split_catalyst=split_catalyst
    )

    # Train emulator
    emulator.train(max_epochs=max_epochs, cv_folds=cv_folds, verbose=verbose)

    # Parity plot
    fig, _ = emulator.parity_plot(include_test=True)
    figure_path = Path(figure_path)
    fig.savefig(figure_path / f"{dataset_name}_parity_plot.png", dpi=300)

    # Save emulator
    emulator.save(save_dir=save_path)


@app.command()
def stbo_optimization(
    benchmark_path: str,
    output_path: str,
    max_experiments: Optional[int] = 20,
    batch_size: Optional[int] = 1,
    repeats: Optional[int] = 20,
    print_warnings: Optional[bool] = True,
):
    """Optimization of a Suzuki benchmark with Single-Task Bayesian Optimziation"""
    # Load benchmark
    exp = SuzukiEmulator.load(benchmark_path)

    # Single-Task Bayesian Optimization
    max_iterations = max_experiments // batch_size
    max_iterations += 1 if max_experiments % batch_size != 0 else 0
    output_path = Path(output_path)
    for i in trange(repeats):
        result = run_stbo(exp, max_iterations=max_iterations, batch_size=batch_size)
        result.save(output_path / f"repeat_{i}.json")


@app.command()
def mtbo_optimization(
    benchmark_path: str,
    ct_data_path: str,
    output_path: str,
    max_experiments: Optional[int] = 20,
    batch_size: Optional[int] = 1,
    repeats: Optional[int] = 20,
    print_warnings: Optional[bool] = True,
):
    """Optimization of a Suzuki benchmark with Multitask Bayesian Optimziation"""
    # Load benchmark
    exp = SuzukiEmulator.load(benchmark_path)

    # Load suzuki dataset
    ds = get_suzuki_dataset(ct_data_path, split_catalyst=exp.split_catalyst)

    # Single-Task Bayesian Optimization
    max_iterations = max_experiments // batch_size
    max_iterations += 1 if max_experiments % batch_size != 0 else 0
    output_path = Path(output_path)
    for i in trange(repeats):
        result = run_mtbo(exp, max_iterations=max_iterations, batch_size=batch_size)
        result.save(output_path / f"repeat_{i}.json")


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
            f"Nucleophiles in this dataset: {nucleophiles}"
        )

    return df


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


def get_suzuki_dataset(data_path, split_catalyst=True, print_warnings=True) -> DataSet:
    data_path = Path(data_path)
    if not data_path.exists():
        raise ImportError(f"Could not import {data_path}")
    dataset = message_helpers.load_message(str(data_path), dataset_pb2.Dataset)
    valid_output = validations.validate_message(dataset)
    if print_warnings:
        print(valid_output.warnings)
    df = suzuki_reaction_to_dataframe(dataset.reactions, split_catalyst=split_catalyst)
    return DataSet.from_df(df)


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
    data_path: str,
    include_reactant_concentrations: Optional[bool] = False,
    split_catalyst: Optional[bool] = True,
    print_warnings: Optional[bool] = True,
) -> Tuple[dict, Domain]:
    """Prepare domain and data for downstream tasks"""
    logger = logging.getLogger(__name__)
    # Get data
    df = get_suzuki_dataset(
        data_path,
        split_catalyst=split_catalyst,
        print_warnings=print_warnings,
    )

    # Create domains
    if split_catalyst:
        pre_catalysts = df["pre_catalyst_smiles"].unique().tolist()
        logger.info("Number of pre-catalysts:", len(pre_catalysts))
        ligands = df["ligand_smiles"].unique().tolist()
        logger.info("Number of ligands:", len(ligands))
        domain = create_suzuki_domain(
            split_catalyst=True,
            pre_catalyst_list=pre_catalysts,
            ligand_list=ligands,
            include_reactant_concentrations=include_reactant_concentrations,
        )
    else:
        catalysts = df["catalyst_smiles"].unique().tolist()
        logger.info("Number of catalysts:", len(catalysts))
        domain = create_suzuki_domain(split_catalyst=False, catalyst_list=catalysts)
    return df, domain


class SuzukiEmulator(ExperimentalEmulator):
    """Standard experimental emulator with some extra features for Suzuki
    Train a machine learning model based on experimental data.
    The model acts a benchmark for testing optimisation strategies.
    Parameters
    ----------
    model_name : str
        Name of the model, ideally with no spaces
    domain : :class:`~summit.domain.Domain`
        The domain of the emulator
    dataset : :class:`~summit.dataset.Dataset`, optional
        Dataset used for training/validation
    regressor : :class:`torch.nn.Module`, optional
        Pytorch LightningModule class. Defaults to the ANNRegressor
    output_variable_names : str or list, optional
        The names of the variables that should be trained by the predictor.
        Defaults to all objectives in the domain.
    descriptors_features : list, optional
        A list of input categorical variable names that should be transformed
        into their descriptors instead of using one-hot encoding.
    clip : bool or list, optional
        Whether to clip predictions to the limits of
        the objectives in the domain. True (default) means
        clipping is activated for all outputs and False means
        it is not activated at all. A list of specific outputs to clip
        can also be passed.

    """

    def __init__(self, model_name, domain, split_catalyst=True, **kwargs):
        self.split_catalyst = split_catalyst
        super().__init__(model_name, domain, **kwargs)

    def save(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        with open(save_dir / f"{self.model_name}.json", "w") as f:
            d = self.to_dict()
            d["split_catalyst"] = self.split_catalyst
            json.dump(d, f)
        self.save_regressor(save_dir)

    @classmethod
    def load(cls, model_name, save_dir, **kwargs):
        save_dir = pathlib.Path(save_dir)
        with open(save_dir / f"{model_name}.json", "r") as f:
            d = json.load(f)
        exp = cls.from_dict(d, **kwargs)
        exp.split_catalyst = d["split_catalyst"]
        exp.load_regressor(save_dir)
        return exp


def run_stbo(
    exp: Experiment, max_iterations: int = 10, categorical_method: str = "one-hot"
):
    """Run Single Task Bayesian Optimization (AKA normal BO)"""
    exp.reset()
    strategy = STBO(exp.domain, categorical_method=categorical_method)
    r = Runner(strategy=strategy, experiment=exp, max_iterations=max_iterations)
    r.run()
    return r


def run_mtbo(
    exp: Experiment, ct_data: DataSet, max_iterations: int = 10, task: int = 1
):
    """Run Multitask Bayesian optimization"""
    strategy = MTBO(
        exp.domain, pretraining_data=ct_data, categorical_method="one-hot", task=task
    )
    r = Runner(strategy=strategy, experiment=exp, max_iterations=max_iterations)
    r.run()
    return r


if __name__ == "__main__":
    app()
