from multitask.utils import *
from multitask.etl.suzuki_data_utils import *
from multitask.benchmarks.suzuki_emulator import SuzukiEmulator
from summit import *
from pathlib import Path
from typing import Tuple, Optional
import logging
import wandb


logger = logging.getLogger(__name__)


def train_benchmark(
    dataset_name: Optional[str],
    save_path: str,
    figure_path: str,
    wandb_dataset_artifact_name: Optional[str] = None,
    data_file: Optional[str] = None,
    include_reactant_concentrations: Optional[bool] = False,
    print_warnings: Optional[bool] = True,
    split_catalyst: Optional[bool] = True,
    max_epochs: Optional[int] = 1000,
    cv_folds: Optional[int] = 5,
    verbose: Optional[int] = 0,
    wandb_benchmark_artifact_name: str = None,
    use_wandb: bool = True,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = "multitask",
) -> SuzukiEmulator:
    """Train a Suzuki benchmark"""
    # Setup wandb
    config = dict(locals())
    if use_wandb:
        run = wandb.init(
            job_type="training",
            entity=wandb_entity,
            project=wandb_project,
            tags=["benchmark"],
            config=config,
        )

    # Download data from wandb if not provided
    if data_file is None and use_wandb:
        dataset_artifact = run.use_artifact(wandb_dataset_artifact_name)
        data_file = Path(dataset_artifact.download()) / f"{dataset_name}.pb"
    elif data_file is None and not use_wandb:
        raise ValueError("Must provide data path if not using wandb")

    # Get data
    ds, domain = prepare_domain_data(
        data_file=data_file,
        include_reactant_concentrations=include_reactant_concentrations,
        split_catalyst=split_catalyst,
        print_warnings=print_warnings,
    )
    logger.info(f"Dataset size: {ds.shape[0]}")
    if use_wandb:
        wandb.config.update({"dataset_size": ds.shape[0]})

    # Create emulator benchmark
    emulator = SuzukiEmulator(
        dataset_name, domain, dataset=ds, split_catalyst=split_catalyst
    )

    # Train emulator
    emulator.train(max_epochs=max_epochs, cv_folds=cv_folds, verbose=verbose)

    # Parity plot
    fig, axes = emulator.parity_plot(include_test=True)
    ax = axes[0]
    ax.set_title("")
    ax.set_xlabel("Measured Yield (%)")
    ax.set_ylabel("Predicted Yield (%)")
    figure_path = Path(figure_path)
    figure_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path / f"{dataset_name}_parity_plot.png", dpi=300)

    # Save emulator
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    emulator.save(save_dir=save_path)

    # Upload results to wandb
    if use_wandb:
        if wandb_benchmark_artifact_name is None:
            wandb_benchmark_artifact_name = f"benchmark_{dataset_name}"
        artifact = wandb.Artifact(wandb_benchmark_artifact_name, type="model")
        artifact.add_dir(save_path)
        wandb.log({"parity_plot": wandb.Image(fig)})
        figure_path = Path(figure_path)
        artifact.add_file(figure_path / f"{dataset_name}_parity_plot.png")
        run.log_artifact(artifact)
        run.finish()

    return emulator


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
        name="catalyst_loading",
        description="Concentration of pre_catalyst in molar",
        bounds=[0.005, 0.025],
    )
    # domain += ContinuousVariable(
    #     name="ligand_ratio",
    #     description="Ratio of pre-catalyst to ligand",
    #     bounds=[0, 5],
    # )

    domain += ContinuousVariable(
        name="temperature",
        description="Reaction temperature in deg C",
        bounds=[30, 120],
    )

    domain += ContinuousVariable(
        name="time", description="Reaction time in seconds", bounds=[60, 600]
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
    data_file: str,
    include_reactant_concentrations: Optional[bool] = False,
    split_catalyst: Optional[bool] = True,
    print_warnings: Optional[bool] = True,
) -> Tuple[dict, Domain]:
    """Prepare domain and data for downstream tasks"""
    logger = logging.getLogger(__name__)
    # Get data
    ds = get_suzuki_dataset(
        data_file,
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
        logger.info(f"Number of catalysts: {len(catalysts)}")
        domain = create_suzuki_domain(split_catalyst=False, catalyst_list=catalysts)
    return ds, domain
