import os
from summit import *
from fastprogress.fastprogress import progress_bar
from ord_schema import message_helpers, validations
from ord_schema.proto import dataset_pb2
from ord_schema.proto.reaction_pb2 import *
from ord_schema import units
from rdkit import Chem
from pint import UnitRegistry
from pathlib import Path
import wandb
from wandb.apis.public import Run
from typing import List, Optional
import pandas as pd
import numpy as np
import pkg_resources
import logging
import uuid

ureg = UnitRegistry()

__all__ = [
    "get_pint_amount",
    "get_smiles",
    "calculate_total_volume",
    "get_rxn_yield",
    "ureg",
    "get_reactant_smiles",
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
    if len(yields) > 1:
        raise ValueError(
            "More than one product with a yield in reaction outcome. This is ambiguous."
        )
    elif len(yields) == 0:
        raise ValueError("No reaction yield found in reaction outcome.")
    return yields[0]


def get_reactant_smiles(reaction: Reaction) -> List[str]:
    inputs = reaction.inputs
    reactants = []
    for inp in inputs:
        components = inputs[inp].components
        for c in components:
            if c is not None:
                if c.reaction_role == ReactionRole.REACTANT:
                    reactants.extend(
                        [
                            id_.value
                            for id_ in c.identifiers
                            if id_.type == CompoundIdentifier.SMILES
                        ]
                    )
    return reactants


def fullfact(levels):
    """
    Create a general full-factorial design

    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.

    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels 0 to k-1 for a k-level factor


    Notes
    ------
    This code is copied from pydoe2: https://github.com/clicumu/pyDOE2/blob/master/pyDOE2/doe_factorial.py

    """
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))

    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j] * level_repeat
        rng = lvl * range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng

    return H


def download_runs_wandb(
    api: wandb.Api,
    wandb_entity: str = "ceb-sre",
    wandb_project: str = "multitask",
    include_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    only_finished_runs: bool = True,
) -> List[Run]:
    """
    Parameters
    ----------
    api : wandb.Api
        The wandb API object.
    wandb_entity : str, optional
        The wandb entity to search, by default "ceb-sre"
    wandb_project : str, optional
        The wandb project to search, by default "multitask"
    include_tags : Optional[List[str]], optional
        A list of tags that the run must have, by default None
    filter_tags : Optional[List[str]], optional
        A list of tags that the run must not have, by default None

    """
    runs = api.runs(f"{wandb_entity}/{wandb_project}")

    final_runs = []
    for run in runs:
        # Filtering
        if include_tags is not None:
            if not all([tag in run.tags for tag in include_tags]):
                continue
            if any([tag in run.tags for tag in filter_tags]):
                continue
        if only_finished_runs and run.state != "finished":
            continue
        # Append runs
        final_runs.append(run)
    return final_runs


class WandbRunner(Runner):
    """Run a closed-loop strategy and experiment cycle with logging to Wandb



    Parameters
    ----------
    strategy : :class:`~summit.strategies.base.Strategy`
        The summit strategy to be used. Note this should be an object
        (i.e., you need to call the strategy and then pass it). This allows
        you to add any transforms, options in advance.
    experiment : :class:`~summit.experiment.Experiment`
        The experiment or benchmark class to use for running experiments
    neptune_project : str
        The name of the Neptune project to log data to
    neptune_experiment_name : str
        A name for the neptune experiment
    netpune_description : str, optional
        A description of the neptune experiment
    files : list, optional
        A list of filenames to save to Neptune
    max_iterations: int, optional
        The maximum number of iterations to run. By default this is 100.
    batch_size: int, optional
        The number experiments to request at each call of strategy.suggest_experiments. Default is 1.
    f_tol : float, optional
        How much difference between successive best objective values will be tolerated before stopping.
        This is generally useful for nonglobal algorithms like Nelder-Mead. Default is None.
    max_same : int, optional
        The number of iterations where the objectives don't improve by more than f_tol. Default is max_iterations.
    max_restarts : int, optional
        Number of restarts if f_tol is violated. Default is 0.
    hypervolume_ref : array-like, optional
        The reference for the hypervolume calculation if it is a multiobjective problem.
        Should be an array of length the number of objectives. Default is at the origin.
    """

    def __init__(
        self,
        strategy: Strategy,
        experiment: Experiment,
        wandb_entity: str = None,
        wandb_project: str = None,
        wandb_run_name: Optional[str] = None,
        wandb_notes: str = None,
        wandb_tags: List[str] = None,
        wandb_save_code: bool = True,
        wandb_artifact: str = None,
        hypervolume_ref=None,
        **kwargs,
    ):

        super().__init__(strategy, experiment, **kwargs)

        # Hypervolume reference for multiobjective experiments
        n_objs = len(self.experiment.domain.output_variables)
        self.ref = hypervolume_ref if hypervolume_ref is not None else n_objs * [0]

        # Check that Neptune-client is installed
        installed = {pkg.key for pkg in pkg_resources.working_set}
        if "wandb" not in installed:
            raise RuntimeError(
                "Wandb is not installed. Use pip install summit[experiments] to add extra dependencies."
            )

        # Set up Neptune variables
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_notes = wandb_notes
        self.wandb_tags = wandb_tags
        self.wandb_save_code = wandb_save_code
        self.wandb_artifact = wandb_artifact

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def run(self, **kwargs):
        """Run the closed loop experiment cycle

        Parameters
        ----------
        save_freq : int, optional
            The frequency with which to checkpoint the state of the optimization. Defaults to None.
        save_at_end : bool, optional
            Save the state of the optimization at the end of a run, even if it is stopped early.
            Default is True.
        save_dir : str, optional
            The directory to save checkpoints locally. Defaults to `~/.summit/runner`.
        """
        # Set parameters
        prev_res = None
        self.restarts = 0
        n_objs = len(self.experiment.domain.output_variables)
        fbest_old = np.zeros(n_objs)
        fbest = np.zeros(n_objs)

        # Serialization
        save_freq = kwargs.get("save_freq")
        save_dir = kwargs.get("save_dir", str(get_summit_config_path()))
        self.uuid_val = uuid.uuid4()
        save_dir = pathlib.Path(save_dir) / "runner" / str(self.uuid_val)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_at_end = kwargs.get("save_at_end", True)
        if self.wandb_artifact:
            artifact = wandb.Artifact(self.wandb_artifact)
            artifact.add_dir(save_dir)

        # Create wandb run
        skip_wandb_intialization = kwargs.get("skip_wandb_intialization", False)
        if not skip_wandb_intialization:
            wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                name=self.wandb_run_name,
                tags=self.wandb_tags,
                notes=self.wandb_notes,
                save_code=self.wandb_save_code,
            )

        # Run optimization loop
        if kwargs.get("progress_bar", True):
            bar = progress_bar(range(self.max_iterations))
        else:
            bar = range(self.max_iterations)
        for i in bar:
            # Get experiment suggestions
            if i == 0:
                k = self.n_init if self.n_init is not None else self.batch_size
                next_experiments = self.strategy.suggest_experiments(num_experiments=k)
            else:
                next_experiments = self.strategy.suggest_experiments(
                    num_experiments=self.batch_size, prev_res=prev_res
                )
            prev_res = self.experiment.run_experiments(next_experiments)

            # Send best objective values to wandb
            for j, v in enumerate(self.experiment.domain.output_variables):
                if i > 0:
                    fbest_old[j] = fbest[j]
                if v.maximize:
                    fbest[j] = self.experiment.data[v.name].max()
                elif not v.maximize:
                    fbest[j] = self.experiment.data[v.name].min()

                wandb.log({f"{v.name}_best": fbest[j]})

            # Send hypervolume for multiobjective experiments
            if n_objs > 1:
                output_names = [v.name for v in self.experiment.domain.output_variables]
                data = self.experiment.data[output_names].copy()
                for v in self.experiment.domain.output_variables:
                    if v.maximize:
                        data[(v.name, "DATA")] = -1.0 * data[v.name]
                y_pareto, _ = pareto_efficient(data.to_numpy(), maximize=False)
                hv = hypervolume(y_pareto, self.ref)
                wandb.log("hypervolume", hv)

            # Save state
            if save_freq is not None:
                file = save_dir / f"iteration_{i}.json"
                if i % save_freq == 0:
                    self.save(file)
                    wandb.log_artifact(artifact)
                if not save_dir:
                    os.remove(file)

            # Stop if no improvement
            compare = np.abs(fbest - fbest_old) > self.f_tol
            if all(compare) or i <= 1:
                nstop = 0
            else:
                nstop += 1

            if self.max_same is not None:
                if nstop >= self.max_same and self.restarts >= self.max_restarts:
                    self.logger.info(
                        f"{self.strategy.__class__.__name__} stopped after {i+1} iterations and {self.restarts} restarts."
                    )
                    break
                elif nstop >= self.max_same:
                    nstop = 0
                    prev_res = None
                    self.strategy.reset()
                    self.restarts += 1

        # Save at end
        if save_at_end:
            file = save_dir / f"iteration_{i}.json"
            self.save(file)
            if self.wandb_artifact:
                wandb.log_artifact(artifact)
            if not save_dir:
                os.remove(file)

    def to_dict(
        self,
    ):
        d = super().to_dict()
        d["runner"].update(
            dict(
                hypervolume_ref=self.ref,
                wandb_entity=self.wandb_entity,
                wandb_project=self.wandb_project,
                wandb_run_name=self.wandb_run_name,
                wandb_notes=self.wandb_notes,
                wandb_tags=self.wandb_tags,
                wandb_save_code=self.wandb_save_code,
                wandb_artifact=self.wandb_artifact,
            )
        )
        return d
