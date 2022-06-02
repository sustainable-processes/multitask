import streamlit as st
from summit import *
from multitask.mt import NewMTBO
import pandas as pd


st.title("Multitask Bayesian Optimization")


ct_csvs = st.file_uploader("Cotraining data", accept_multiple_files=True)

curr_csv = st.file_uploader("Training data")

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


# Read and transform data
if ct_csvs is not None:
    ct_dfs = [pd.read_csv(ct_csv, skiprows=1) for ct_csv in ct_csvs]
    st.write("Cotraining data")
    st.write(pd.concat(ct_dfs))
else:
    ct_dfs = None
if curr_csv is not None:
    df = pd.read_csv(curr_csv, skiprows=1)
    st.write("Trainin data")
    st.write(df)
else:
    df = None


# Run suggestions
if ct_dfs is None or df is None:
    st.write("Please upload training and cotraining data")
else:
    # Transform data
    ct_dss = [transform_data(ct_df) for ct_df in ct_dfs]
    i = 0
    for ct_ds in ct_dss:
        ct_ds[("task", "METADATA")] = i
        i += 1
    ds = transform_data(df)
    ds[("task", "METADATA")] = i

    # Setup domain
    domain = create_domain()
    st.write("Domain")
    st.write(domain)

    # Get suggestions
    with st.spinner("Generating suggestions"):
        strategy = NewMTBO(
            domain,
            pretraining_data=pd.concat(ct_dss),
            acquisition_function="qNEI",
            brute_force_categorical=False,
            #     model_type=NewMTBO.LCM,
        )
        suggestions = strategy.suggest_experiments(int(1))
        suggestions = suggestions.round(0)
    st.write("Suggestions")
    st.write(suggestions)


# ds_a = transform_data(df_a)
# ds_b = transform_data(df_b)
# ds_a[("task", "METADATA")] = 0
# ds_b[("task", "METADATA")] = 1
