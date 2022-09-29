from multitask.utils import *
from summit import *
from ord_schema import message_helpers, validations
from ord_schema.proto import dataset_pb2
from ord_schema.proto.reaction_pb2 import *
from pathlib import Path
import pkg_resources
from typing import Iterable
import pandas as pd


__all__ = [
    "get_suzuki_dataset",
    "cn_reaction_to_dataframe",
    "get_suzuki_row",
    "split_cat_ligand",
]


def get_cn_dataset(data_path, print_warnings=True) -> DataSet:
    """
    Get a Suzuki ORD dataset as a Summit DataSet
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise ImportError(f"Could not import {data_path}")
    dataset = message_helpers.load_message(str(data_path), dataset_pb2.Dataset)

    valid_output = validations.validate_message(dataset)
    if print_warnings:
        print(valid_output.warnings)
    df = cn_reaction_to_dataframe(dataset.reactions)
    return DataSet.from_df(df)


def cn_reaction_to_dataframe(reactions: Iterable[Reaction]) -> pd.DataFrame:
    """Convert a list of reactions into a dataframe that can be used for machine learning

    Parameters
    ---------
    reactions: list of Reaction
        A list of ORD reaction objects

    Returns
    -------
    df: pd.DataFrame
        The dataframe with each row as a reaction

    """

    # Convert dataset to pandas dataframe
    df = message_helpers.messages_to_dataframe(reactions, drop_constant_columns=True)

    # Calculate base equivalents
    df["base_equiv"] = (
        df['inputs["Base"].components[0].amount.moles.value']
        / df['inputs["Electrophile"].components[0].amount.moles.value']
    )

    # Rename columns
    column_map = {
        'inputs["Catalyst"].components[0].identifiers[0].value': "catalyst",
        'inputs["Base"].components[0].identifiers[0].value': "base",
        # 'inputs["Solvent"].components[0].identifiers[0].value': "solvent",
        "outcomes[0].reaction_time.value": "time",
        "conditions.temperature.setpoint.value": "temperature",
        "outcomes[0].products[0].measurements[0].percentage.value": "yld",
    }
    df = df.rename(columns=column_map)

    # Only keep needed columns
    df = df[list(column_map.values()) + ["base_equiv"]]

    return df
