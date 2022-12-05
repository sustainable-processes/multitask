from summit import *
from summit.domain import Domain
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from typing import Dict, List, Optional
from yaml import load as yaml_load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import warnings
from pathlib import Path


class MITKinetics(Experiment):
    """Benchmark representing a simulated kinetic reaction network and accompanying kinetic constants (see reference).
    The reactions occur in a batch reactor.
    The objective is to maximize yield (y), defined as the concentration of product dividen by the initial concentration of
    the limiting reagent (We can do this because the stoichiometry is 1:1).
    We optimize the reactions by changing the catalyst concentration, reaction time, choice of catalyst, and temperature.

    Parameters
    ----------
    noise_level: float, optional
        The mean of the random noise added to the concentration measurements in terms of
        percent of the signal. Default is 0.


    Notes
    -----
    This benchmark relies on the kinetics simulated by Jensen et al. The mechanistic
    model is integrated using scipy to find outlet concentrations of all species.

    References
    ----------
    K. Jensen et al., React. Chem. Eng., 2018, 3,301
    DOI: 10.1039/c8re00032h
    """

    def __init__(self, noise_level=0, case=1, noise_type: Optional[str] = None):
        domain = self._setup_domain()
        super().__init__(domain)
        self.rng = np.random.default_rng()
        self.noise_level = noise_level
        self.case = case
        self.noise_type = noise_type

    def _setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "catalyst concentration"
        domain += ContinuousVariable(
            name="conc_cat",
            description=des_1,
            bounds=[0.835 * 10 ** (-3), 4.175 * 10 ** (-3)],
        )

        des_2 = "reaction time"
        domain += ContinuousVariable(name="t", description=des_2, bounds=[60, 600])

        des_3 = "Choice of catalyst"
        domain += CategoricalVariable(
            name="cat_index", description=des_3, levels=[0, 1, 2, 3, 4, 5, 6, 7]
        )

        des_4 = "Reactor temperature in degress celsius"
        domain += ContinuousVariable(
            name="temperature", description=des_4, bounds=[30, 110]
        )

        # Objectives
        des_5 = "yield (%)"
        domain += ContinuousVariable(
            name="y",
            description=des_5,
            bounds=[0, 100],
            is_objective=True,
            maximize=True,
        )

        return domain

    def _run(self, conditions, **kwargs):
        conc_cat = float(conditions["conc_cat"])
        t = float(conditions["t"])
        cat_index = int(conditions["cat_index"])
        T = float(conditions["temperature"])
        y, res = self._integrate_equations(conc_cat, t, cat_index, T)
        conditions[("y", "DATA")] = y
        return conditions, {}

    def _integrate_equations(self, conc_cat, t, cat_index, T):
        # Initial Concentrations in mM
        self.C_i = np.zeros(6)
        self.C_i[0] = 0.167  # Initial conc of A
        self.C_i[1] = 0.250  # Initial conc of B
        self.C_i[2] = conc_cat  # Initial conc of cat

        # Integrate
        res = solve_ivp(
            self._integrand, [0, t], self.C_i, args=(cat_index, T, self.case)
        )
        C_final = res.y[:, -1]

        # Add measurment noise
        if self.noise_type == "constant":
            noise_level = self.noise_level
        elif self.noise_type == "knee":
            y = C_final[3] / self.C_i[0]
            if y < 0.2:
                noise_level = 5.0
            elif y >= 0.2 and y < 0.9:
                noise_level = 1.0
            elif y > 0.9:
                noise_level = 5.0
        C_final += C_final * self.rng.normal(scale=noise_level, size=len(C_final)) / 100
        C_final[
            C_final < 0
        ] = 0  # prevent negative values of concentration introduced by noise

        # calculate yield
        y = C_final[3] / self.C_i[0] * 100
        return y, res

    @staticmethod
    def _integrand(t, C, cat_index, T, case):
        # Kinetic Constants
        R = 8.314 / 1000  # kJ/K/mol
        T = T + 273.71  # Convert to deg K
        conc_cat = C[2]
        A_R = 3.1 * 10**7
        A_S1 = 1 * 10**12
        A_S2 = 3.1 * 10**5

        k = (
            lambda conc_cat, A, E_A, E_Ai, temp: np.sqrt(conc_cat)
            * A
            * np.exp(-(E_A + E_Ai) / (R * temp))
        )

        if case == 1:
            E_Ai = [0, 0.3, 0.3, 0.7, 0.7, 2.2, 3.8, 7.3]
            k_S1 = 0
            k_S2 = 0
        elif case == 2:
            E_Ai = [0, 0, 0.3, 0.7, 0.7, 2.2, 3.8, 7.3]
            k_S1 = 0
            k_S2 = 0
        elif case == 3:
            E_Ai = [0, 0.3, 0.3, 0.7, 0.7, 2.2, 3.8, 7.3]
            k_S1 = k(conc_cat, A_S1, 100, 0, T)
            k_S2 = 0
        elif case == 4:
            E_Ai = [0, 0.3, 0.3, 0.7, 0.7, 2.2, 3.8, 7.3]
            k_S1 = 0
            k_S2 = k(conc_cat, A_S2, 50, 0, T)
        elif case == 5:
            if T < 80 + 273.71:
                E_Ai = [-5.0, 0.3, 0.3, 0.7, 0.7, 2.2, 3.8, 7.3]
            else:
                E_Ai = [
                    -5.0 + 0.3 * (T - 80 - 273.71),
                    0.3,
                    0.3,
                    0.7,
                    0.7,
                    2.2,
                    3.8,
                    7.3,
                ]
            k_S1 = 0
            k_S2 = 0

        E_AR = 55
        k_R = k(conc_cat, A_R, E_AR, E_Ai[cat_index], T)

        # Reaction Rates
        r = np.zeros(6)
        r[0] = -k_R * C[0] * C[1]
        r[1] = -k_R * C[0] * C[1] - k_S1 * C[1] - k_S2 * C[1] * C[3]
        r[2] = 0
        r[3] = k_R * C[0] * C[1] - k_S2 * C[1] * C[3]
        r[4] = k_S1 * C[1]
        r[5] = k_S2 * C[1] * C[3]

        # Deltas
        dcdt = r
        return dcdt

    def to_dict(self, **kwargs):
        experiment_params = dict(noise_level=self.noise_level)
        return super().to_dict(**experiment_params)


def create_pcs_ds(
    solvent_ds: DataSet,
    ucb_ds: DataSet,
    solubilities: DataSet,
    num_components: int,
    ucb_filter: bool = True,
    verbose: bool = False,
):
    """Create dataset with principal components"""

    # Merge data sets
    solvent_ds_full = solvent_ds.join(solubilities)
    if ucb_filter:
        solvent_ds_final = pd.merge(
            solvent_ds_full, ucb_ds, left_index=True, right_index=True
        )
    else:
        solvent_ds_final = solvent_ds_full
    if verbose:
        print(f"{solvent_ds_final.shape[0]} solvents for optimization")

    # Double check that there are no NaNs in the descriptors
    values = solvent_ds_final.data_to_numpy()
    values = values.astype(np.float64)
    check = np.isnan(values)
    assert check.all() == False

    # Transform to principal componets
    pca = PCA(n_components=num_components)
    pca.fit(solvent_ds_full.standardize())
    pcs = pca.fit_transform(solvent_ds_final.standardize())
    if verbose:
        explained_var = round(pca.explained_variance_ratio_.sum() * 100)
        expl = f"{explained_var}% of variance is explained by {num_components} principal components."
        print(expl)

    # Create a new dataset with just the principal components
    metadata_df = solvent_ds_final.loc[:, solvent_ds_final.metadata_columns]
    pc_df = pd.DataFrame(
        pcs,
        columns=[f"PC_{i+1}" for i in range(num_components)],
        index=metadata_df.index,
    )
    pc_ds = DataSet.from_df(pc_df)
    return pd.concat([metadata_df, pc_ds], axis=1), pca


class StereoSelectiveReaction(Experiment):
    """Generate data to simulate a stereoselective chemical reaction

    Parameters
    -----------
    solvent_ds: Dataset
        Dataset with the solvent descriptors (must have cas numbers as index)
    random_state: `numpy.random.RandomState`, optional
        RandomState object. Creates a random state based ont eh computer clock
        if one is not passed
    pre_calculate: bool, optional
        If True, pre-calculates the experiments for all solvents. Defaults to False
    Notes
    -----
    This comes from Kobi Felton's MPhil Research Thesis.

    Pre-calculating will ensure that multiple calls to experiments will give the same result
    (as long as a random state is specified).

    """

    def __init__(
        self,
        solvent_ds: DataSet,
        random_state=None,
        use_descriptors: bool = True,
        initial_concentrations: List = [0.5, 0.5],
        pre_calculate: bool = False,
    ):
        self.solvent_ds = solvent_ds
        self.use_descriptors = use_descriptors
        self.initial_concentrations = initial_concentrations
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self.pre_calculate = pre_calculate
        self.cas_numbers = self.solvent_ds.index.values.tolist()
        domain = self.setup_domain(
            solvent_ds=solvent_ds, use_descriptors=use_descriptors
        )
        super().__init__(domain)
        if pre_calculate:
            all_experiments = [self._run(cas) for cas in self.cas_numbers]
            self.all_experiments = np.array(all_experiments)
        else:
            self.all_experiments = None

    @staticmethod
    def setup_domain(solvent_ds: DataSet, use_descriptors: bool):
        domain = Domain()
        if use_descriptors:
            kwargs = {"descriptors": solvent_ds}
        else:
            kwargs = {"levels": solvent_ds.index.values.tolist()}
        domain += CategoricalVariable(
            name="solvent",
            description="solvent for the borrowing hydrogen reaction",
            **kwargs,
        )
        domain += ContinuousVariable(
            name="temperature", description="Reaction temperature", bounds=[80, 120]
        )
        domain += ContinuousVariable(
            name="conversion",
            description="relative conversion to triphenylphosphine oxide determined by LCMS",
            bounds=[0, 100],
            is_objective=True,
        )
        domain += ContinuousVariable(
            name="de",
            description="diastereomeric excess determined by ratio of LCMS peaks",
            bounds=[0, 100],
            is_objective=True,
        )
        return domain

    def _run(self, conditions: DataSet):
        solvent_cas = str(conditions["solvent"].values[0])
        if self.all_experiments is None:
            result = self._run_rxn(solvent_cas)
        else:
            index = self.cas_numbers.index(solvent_cas)
            result = self.all_experiments[index, :]
        conditions["conversion", "DATA"] = result[0]
        conditions["de", "DATA"] = result[1]
        return conditions, {}

    def _run_rxn(self, solvent_cas, rxn_time=25200, step_size=200, time_series=False):
        """Generate fake experiment data for a stereoselective reaction"""
        rxn_time = rxn_time + self.random_state.randn(1) * 0.01 * rxn_time
        x = self._integrate_rate(solvent_cas, rxn_time, step_size)
        cd1 = x[:, 0]
        cd2 = x[:, 1]

        conversion = cd1 / np.min(self.initial_concentrations) * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            de = cd1 / (cd1 + cd2) * 100

        if not time_series:
            conversion = conversion[-1]
            de = de[-1]
        return np.array([conversion, de])

    def _integrate_rate(self, solvent_cas, rxn_time=25200, step_size=200):
        t0 = 0  # Have to start from time 0
        x0 = 0.0  # Always start with 0 extent of reaction
        trange = np.arange(t0, rxn_time, step_size)
        res = solve_ivp(
            self._rate, [t0, rxn_time], [x0, x0], args=(solvent_cas,), t_eval=trange
        )
        return res.y.T

    def _rate(self, t, X, solvent_cas):
        """Calculate  rates  for a given extent of reaction"""
        # Variables
        x = X
        # x = X[:, :2]  # product concentrations
        # T = X[:, 2]

        # Constants
        AD1 = 8.5e9  # L/(mol-s)
        AD2 = 8.3e9  # L/(mol-s)
        EAD1 = 105  # kJ/mol
        EAD2 = 110  # kJ/mol
        TRXN = 393  # K
        R = 8.314e-3  # kJ/mol/K
        Es1 = (
            lambda pc1, pc2, pc3: -np.log(
                abs((pc2 + 0.73 * pc1 - 4.46) * (pc2 + 2.105 * pc1 + 11.367))
            )
            + pc3
        )
        Es2 = (
            lambda pc1, pc2, pc3: -2 * np.log(abs((pc2 + 0.73 * pc1 - 4.46)))
            - 0.2 * pc3**2
        )

        # Solvent variable reaction rate coefficients
        pc_solvent = self.solvent_ds.loc[solvent_cas][
            self.solvent_ds.data_columns
        ].to_numpy()
        es1 = Es1(pc_solvent[0], pc_solvent[1], pc_solvent[2])
        es2 = Es2(pc_solvent[0], pc_solvent[1], pc_solvent[2])
        # T = 0.5 * self.random_state.randn() + TRXN
        T = TRXN
        kd1 = AD1 * np.exp(-(EAD1 + es1) / (R * T))
        kd2 = AD2 * np.exp(-(EAD2 + es2) / (R * T))

        # Calculate rates
        x1dot = kd1 * self.ca(x) * self.cb(x)
        x2dot = kd2 * self.ca(x) * self.cb(x)
        return np.array([x1dot, x2dot])

    def ca(self, x):
        try:
            ca = self.initial_concentrations[0] - x[:, 0] - x[:, 1]
        except IndexError:
            ca = self.initial_concentrations[0] - x[0] - x[1]
        return ca

    def cb(self, x):
        try:
            cb = self.initial_concentrations[1] - x[:, 0] - x[:, 1]
        except IndexError:
            cb = self.initial_concentrations[1] - x[0] - x[1]
        return cb


class MultitaskKinetics(Experiment):
    def __init__(
        self,
        name: str,
        ligand_constants: Dict[str, dict],
        solvent_constants: Dict[str, dict],
        noise_level: float = 2.0,
    ):
        self.name = name
        self.ligand_constants = ligand_constants
        self.solvent_constants = solvent_constants
        self.noise_level = noise_level
        self.rng = np.random.default_rng()
        domain = self.setup_domain(
            ligands=list(ligand_constants.keys()),
            solvents=list(solvent_constants.keys()),
        )

        super().__init__(domain)

    @staticmethod
    def setup_domain(ligands: List[str], solvents: List[str]):
        domain = Domain()
        domain += CategoricalVariable(
            name="ligand", description="Ligand for catalyst complex", levels=ligands
        )
        domain += CategoricalVariable(
            name="solvent", description="Reaction solvent", levels=solvents
        )
        domain += ContinuousVariable(
            name="temperature",
            description="Reaction temperature in degrees Celsius",
            bounds=(30, 100),
        )
        domain += ContinuousVariable(
            name="res_time", description="Residence time in minutes", bounds=(60, 240)
        )
        domain += ContinuousVariable(
            name="cat_conc", description="Catalyst concentration", bounds=(1, 5)
        )
        domain += ContinuousVariable(
            name="yld", description="Reaction yield", bounds=(0, 100), is_objective=True
        )
        return domain

    def _run(self, conditions: DataSet, **kwargs):
        rxn = conditions
        df = self._react(
            ligand=str(rxn["ligand"].values[0]),
            solvent=str(rxn["solvent"].values[0]),
            temperature=float(rxn["temperature"]),
            res_time=float(rxn["res_time"]),
            cat_conc=float(rxn["cat_conc"]),
            # step_size=float(rxn["res_time"]) / 2,
        )
        limiting = min(df["sm1"].iloc[0], df["sm2"].iloc[0])
        prod = df["prod"].iloc[-1]
        conditions["yld", "DATA"] = prod / limiting * 100
        return conditions, {}

    def _react(
        self,
        ligand: str,
        solvent: str,
        temperature: float,
        res_time: float,
        cat_conc: float,
        step_size: float = 15,
    ) -> pd.DataFrame:
        # Constants
        C0 = [
            0.0,  # Precat
            0.0,  # Ligand
            0.3,  # SM1
            0.25,  # SM2
            0.003 * cat_conc,  # Catalyst
            0.0,  # Product
            0.0,  # impurity1
            0.0,  # impurity2
            0.0,  # deactivatecat
            0.0,  # impurity3
        ]
        T = temperature + 273.15

        # Kinetic constants
        lig = self.ligand_constants[ligand]
        k = lig["k"]
        Ea = lig["Ea"]
        AD = [7e9, 8.3e9, 8.5e9, 8e9, 8.8e9, 8e9]
        solvent_multiplers = self.solvent_constants[solvent]["k"]
        kV = [
            ADi * ki * np.exp(-Eai / (8.314e-3 * T)) for ADi, ki, Eai in zip(AD, k, Ea)
        ]
        kV = [
            kVi * solvent_multipler_i
            for kVi, solvent_multipler_i in zip(kV, solvent_multiplers)
        ]
        trange = np.arange(0, res_time, step_size)

        # Solve differential equations
        res = solve_ivp(
            self._rate,
            t_span=[0, res_time],
            y0=C0,
            t_eval=trange,
            args=(kV,),
            method="LSODA",  # Works well for stiff systems
        )

        # Noise
        y = res.y.T
        y += y * self.rng.normal(scale=self.noise_level, size=y.shape) / 100

        return pd.DataFrame(
            y,
            columns=[
                "precat",
                "ligand",
                "sm1",
                "sm2",
                "cat",
                "prod",
                "impurity1",
                "impurity2",
                "deactivate_cat",
                "impurity3",
            ],
            index=trange,
        )

    def _rate(self, t, C, kV: list):
        # Concentrations
        # precat = C[0]
        lig = C[1]
        sm1 = C[2]
        sm2 = C[3]
        cat = C[4]
        product = C[5]
        # impurity_1 = C[6]
        # impurity_2 = C[7]
        # deactive_cat = C[8]

        # Equations
        r = np.zeros(10)
        r[0] = kV[0] * cat
        r[1] = kV[0] * cat
        r[2] = -kV[1] * sm1 * sm2 * lig - kV[2] * sm1 * sm2 * lig
        r[3] = -kV[1] * sm1 * sm2 * lig - kV[2] * sm1 * sm2 * lig - kV[3] * sm2
        r[4] = -kV[0] * cat - 2 * kV[4] * cat * cat
        r[5] = kV[1] * sm1 * sm2 * lig - kV[5] * product
        r[6] = kV[2] * sm1 * sm2 * lig
        r[7] = kV[3] * sm2
        r[8] = kV[4] * cat * cat
        r[9] = kV[5] * product

        return r

    @classmethod
    def load_yaml(cls, filepath: str):
        # Loading data
        with open(filepath, "r") as f:
            data = yaml_load(f, Loader=Loader)
        name = data["name"]
        ligands = data["ligands"]
        solvents = data["solvents"]

        # Instantiate class
        mtk = cls(name=name, ligand_constants=ligands, solvent_constants=solvents)

        return mtk
