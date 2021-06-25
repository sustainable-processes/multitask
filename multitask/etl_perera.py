from multitask.utils import get_reactant_smiles
from multitask.suzuki_data_utils import contains_boron

from ord_schema import message_helpers, validations
from ord_schema.proto import dataset_pb2
from ord_schema.proto.reaction_pb2 import *

from pathlib import Path
from typing import List
import pandas as pd
import typer


def split_perera_datasets(data_path: str, output_dir: str):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    #  Load Perera
    perera_big_dataset = message_helpers.load_message(
        str(data_path), dataset_pb2.Dataset
    )

    # Split into datasets for each reactant
    perera_datasets = {}
    print("Separating datasets")
    for reaction in perera_big_dataset.reactions:
        reactants = get_reactant_smiles(reaction)
        reactants_final = [""] * 2
        for reactant in reactants:
            if contains_boron(reactant):
                reactants_final[0] = reactant
            else:
                reactants_final[1] = reactant
        reactants = "".join(reactants_final)
        if reactants in perera_datasets:
            perera_datasets[reactants].append(reaction)
        else:
            perera_datasets[reactants] = [reaction]

    # Save datasets
    i = 1
    print("Saving datasets")
    for _, reactions in perera_datasets.items():
        dataset = dataset_pb2.Dataset()
        dataset.name = "Perera Suzuki Cross-Coupling"
        dataset.reactions.extend(reactions)
        dataset.dataset_id = str(1)
        print(f"Dataset {i} has {len(reactions)} reactions.")
        with open(output_dir / f"perera_dataset_case_{i}.pb", "wb") as f:
            f.write(dataset.SerializeToString())
        i += 1


if __name__ == "__main__":
    typer.run(split_perera_datasets)
