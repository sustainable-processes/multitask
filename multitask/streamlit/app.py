import streamlit as st
from summit import *
from multitask.strategies.mt import NewMTBO
import pandas as pd


# Transform data
def transform_data(df):
    categorical_columns = ["Solvent", "Ligand"]
    for col in categorical_columns:
        df[col] = df[col].str.split(r" \(\d\)", expand=True)[0]
    for col in df.columns:
        if "Unnamed" in col:
            df = df.drop(col, axis=1)
    df = df.rename(
        columns={
            "ResT /min": "ResT",
            "Temp /°C": "Temp",
            "Mol%": "Mol",
            "Yield /%": "yld",
        }
    )
    ds = DataSet.from_df(df)
    return ds


def create_domain():
    # Create domain
    domain = Domain()

    # Solvents: Toluene (1), DMA (2), MeCN (3), DMSO (4), NMP (5)
    # Ligand: JohnPhos (1), Sphos (2), Xphos (3), DPEPhos (4)
    # ResT: 5 - 60 mins
    # Temp: 50 - 150 deg
    # Mol%: 2 - 10 %
    domain += CategoricalVariable(
        "Solvent",
        "Solvent used for the reaction",
        levels=["Toluene", "DMA", "MeCN", "DMSO", "NMP"],
    )
    domain += CategoricalVariable(
        "Ligand",
        "Ligand used for the reaction",
        levels=["JohnPhos", "SPhos", "Xphos", "DPEPhos"],
    )
    domain += ContinuousVariable("ResT", "Residence Time (minutes)", bounds=(5, 60))
    domain += ContinuousVariable(
        "Temp", "Reaction temperature in deg C", bounds=(50, 150)
    )
    domain += ContinuousVariable("Mol", "Catalyst mol percent", bounds=(2, 10))
    domain += ContinuousVariable(
        "yld", "Reaction yield", bounds=(0, 100), is_objective=True, maximize=True
    )
    return domain


columns = [
    "case",
    "Type",
    "Solvent",
    "Ligand",
    "ResT",
    "Temp",
    "Mol",
    "yld",
]

"""
# Summit Multitask Optimization

Optimize faster using data from past experiments.

## Step 1: Upload data for past cases
"""
# Step 1: Cotraining data
ct_csvs = st.file_uploader(
    "Each case must be a separate CSV", accept_multiple_files=True, type="csv"
)
case = 0
if len(ct_csvs) > 0:
    ct_dfs = [pd.read_csv(ct_csv, skiprows=1) for ct_csv in ct_csvs]
    # Transform data
    ct_dss = [transform_data(ct_df) for ct_df in ct_dfs]

    for ct_ds in ct_dss:
        ct_ds[("task", "METADATA")] = case
        case += 1
    st.write("Cotraining case data:")
    to_display = pd.concat(ct_dss)
    to_display = to_display.rename(columns={"task": "case"})
    st.dataframe(to_display[columns], height=200)
else:
    ct_dss = None


# Step 2: Current data
if len(ct_csvs) > 0:
    """
    ## Step 2: Upload current data
    """
    curr_csv = st.file_uploader("CSV with training data", type="csv")
else:
    curr_csv = None
if curr_csv is not None:
    df = pd.read_csv(curr_csv, skiprows=1)
    ds = transform_data(df)
    ds[("task", "METADATA")] = case

    st.write("Training data for current case")
    to_display = ds.rename(columns={"task": "case"})
    st.dataframe(to_display[columns], height=200)
else:
    ds = None


# Run suggestions
if ct_dss is not None and ds is not None:
    """
    ## Step 3: Get new suggestions
    """
    # Setup domain
    domain = create_domain()

    # Get suggestions
    with st.spinner("Generating suggestion"):
        strategy = NewMTBO(
            domain,
            pretraining_data=pd.concat(ct_dss),
            acquisition_function="qNEI",
            brute_force_categorical=False,
            task=case,
        )
        suggestions = strategy.suggest_experiments(int(1), prev_res=ds)
        suggestions = suggestions.round(0)
    st.write("Suggestion")
    st.write(suggestions)
