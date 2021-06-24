from summit import *

import ord_schema
from ord_schema import message_helpers, validations
from ord_schema.proto import dataset_pb2
from ord_schema.proto.reaction_pb2 import *
from ord_schema.message_helpers import find_submessages, get_reaction_smiles
from ord_schema import units

from rdkit import Chem
import rdkit.Chem.rdChemReactions as react
from pint import UnitRegistry

from pathlib import Path
from typing import Iterable
import pandas as pd

ligands = [
    "CC(C)(C)P(C(C)(C)C)C(C)(C)C",
    "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
    "CN(C)c1ccc(P(C(C)(C)C)C(C)(C)C)cc1",
    "C1CCC(P(C2CCCCC2)C2CCCCC2)CC1",
    "Cc1ccccc1P(c1ccccc1C)c1ccccc1C",
    "CCCCP([C@]12C[C@H]3C[C@H](C[C@H](C3)C1)C2)[C@]12C[C@H]3C[C@H](C[C@H](C3)C1)C2",
    "COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1",
    "CC(C)(C)P([C]1[CH][CH][CH][CH]1)C(C)(C)C",
    "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1",
    "c1ccc(P(c2ccccc2)[c-]2cccc2)cc1",
    "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21",
]


def split_perera_datasets(data_path):
    data_path = Path(data_path)
    # Perera
    perera_dataset = message_helpers.load_message(str(data_path), dataset_pb2.Dataset)
