# from .mixed_gp_regression import MixedMultiTaskGP, LCMMultitaskGP
from .acquistion import CategoricalEI, CategoricalqNEI, WeightedEI
from summit import *
from summit.strategies.base import Strategy, Transform
from botorch.acquisition import ExpectedImprovement as EI
from botorch.acquisition import qNoisyExpectedImprovement as qNEI
from botorch.acquisition import UpperConfidenceBound as UCB
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood,
)
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.priors.torch_priors import GammaPrior
import torch
import numpy as np
from typing import Optional


dtype = torch.double


class NewMTBO(Strategy):
    """Multitask Bayesian Optimisation using BOtorch

    This strategy enables pre-training a model with past reaction data
    in order to enable faster optimisation.

    Parameters
    ----------

    domain : :class:`~summit.domain.Domain`
        The domain of the optimization
    pretraining_data : :class:`~summit.utils.data.DataSet`, optional
        A DataSet with pretraining data. Must contain a metadata column named "task"
        that specfies the task for all data.
    transform : :class:`~summit.strategies.base.Transform`, optional
        A transform object. By default, no transformation will be done
        on the input variables or objectives.
    task : int, optional
        The index of the task being optimized. Defaults to 1.
    brute_force_categorical : bool, optional
        Whether or not to use the categorical kernel and "brute-force" enumeration of
        all categorical combinations.
    categorical_method : str, optional
        The method for transforming categorical variables. Pass None along with brute_force_categorical=True
        to use the categorical kenrel. Otherwise pass "one-hot" or "descriptors". Descriptors must be included in the
        categorical variables for the later.

    Notes
    -----

    This strategy is based on a paper from the NIPs2020 ML4Molecules workshop by Felton_. See
    Swersky_ for more information on multitask Bayesian optimization.

    References
    ----------

    .. [Felton] K. Felton, et al, in `ML4 Molecules 2020 workshop <https://chemrxiv.org/articles/preprint/Multi-task_Bayesian_Optimization_of_Chemical_Reactions/13250216>`_.
    .. [Swersky] K. Swersky et al., in `NIPS Proceedings <http://papers.neurips.cc/paper/5086-multi-task-bayesian-optimization>`_, 2013, pp. 2004–2012.


    Examples
    --------
    >>> from summit.benchmarks import MIT_case1, MIT_case2
    >>> from summit.strategies import LHS, MTBO
    >>> from summit import Runner
    >>> # Get pretraining data
    >>> exp_pt = MIT_case1(noise_level=1)
    >>> lhs = LHS(exp_pt.domain)
    >>> conditions = lhs.suggest_experiments(10)
    >>> pt_data = exp_pt.run_experiments((conditions))
    >>> pt_data[("task", "METADATA")] = 0
    >>> # Use MTBO on a new mechanism
    >>> exp = MIT_case2(noise_level=1)
    >>> strategy = MTBO(exp.domain,pretraining_data=pt_data, categorical_method="one-hot",task=1)
    >>> r = Runner(strategy=strategy, experiment=exp, max_iterations=2)
    >>> r.run(progress_bar=False)

    """

    ICM = "icm"
    LCM = "lcm"

    def __init__(
        self,
        domain: Domain,
        pretraining_data: DataSet,
        transform: Optional[Transform] = None,
        task: Optional[int] = 1,
        categorical_method: Optional[str] = "one-hot",
        acquisition_function: Optional[str] = "EI",
        model_type: Optional[str] = None,
        **kwargs,
    ):
        Strategy.__init__(self, domain, transform, **kwargs)
        self.pretraining_data = pretraining_data
        self.task = task
        self.categorical_method = categorical_method
        if self.categorical_method not in ["one-hot", "descriptors", None]:
            raise ValueError(
                "categorical_method must be one of 'one-hot' or 'descriptors'."
            )
        self.brute_force_categorical = kwargs.get("brute_force_categorical", False)
        self.acquistion_function = acquisition_function
        self.model_type = model_type if model_type is not None else self.ICM
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    def suggest_experiments(self, num_experiments, prev_res: DataSet = None, **kwargs):
        """Suggest experiments using MTBO

        Parameters
        ----------
        num_experiments : int
            The number of experiments (i.e., samples) to generate
        prev_res : :class:`~summit.utils.data.DataSet`, optional
            Dataset with data from previous experiments.
            If no data is passed, then latin hypercube sampling will
            be used to suggest an initial design.

        Returns
        -------
        next_experiments : :class:`~summit.utils.data.DataSet`
            A Dataset object with the suggested experiments

        Examples
        --------
        >>> from summit.benchmarks import MIT_case1, MIT_case2
        >>> from summit.strategies import LHS, MTBO
        >>> from summit import Runner
        >>> # Get pretraining data
        >>> exp_pt = MIT_case1(noise_level=1)
        >>> lhs = LHS(exp_pt.domain)
        >>> conditions = lhs.suggest_experiments(10)
        >>> pt_data = exp_pt.run_experiments(conditions)
        >>> pt_data["task", "METADATA"] = 0
        >>> # Use MTBO on a new mechanism
        >>> exp = MIT_case2(noise_level=1)
        >>> new_conditions = lhs.suggest_experiments(10)
        >>> data = exp.run_experiments(new_conditions)
        >>> data[("task", "METADATA")] = 1
        >>> strategy = MTBO(exp.domain,pretraining_data=pt_data, categorical_method="one-hot",task=1)
        >>> res = strategy.suggest_experiments(1, prev_res=data)

        """
        if num_experiments != 1:
            raise NotImplementedError(
                "Multitask does not support batch optimization yet. See https://github.com/sustainable-processes/summit/issues/119#"
            )

        # Suggest lhs initial design or append new experiments to previous experiments
        if prev_res is None:
            lhs = LHS(self.domain)
            self.iterations += 1
            k = num_experiments if num_experiments > 1 else 2
            conditions = lhs.suggest_experiments(k)
            conditions[("task", "METADATA")] = self.task
            return conditions
        elif prev_res is not None and self.all_experiments is None:
            self.all_experiments = prev_res
            data = self.all_experiments.append(self.pretraining_data)
        elif prev_res is not None and self.all_experiments is not None:
            self.all_experiments = self.all_experiments.append(prev_res)
            data = self.all_experiments.append(self.pretraining_data)
        else:
            data = self.pretraining_data
        self.iterations += 1

        # Combine pre-training and experiment data
        if "task" not in self.pretraining_data.metadata_columns:
            raise ValueError(
                """The pretraining data must have a METADATA column called "task" with the task number."""
            )

        # Get inputs (decision variables) and outputs (objectives)
        inputs, output = self.transform.transform_inputs_outputs(
            data,
            categorical_method=self.categorical_method,
            min_max_scale_inputs=True,
            # min_max_scale_outputs=True,
            standardize_outputs=True,
        )

        # Categorial transformation
        if self.categorical_method is None:
            cat_mappings = {}
            cat_dimensions = []
            for i, v in enumerate(self.domain.input_variables):
                if v.variable_type == "categorical":
                    cat_mapping = {l: i for i, l in enumerate(v.levels)}
                    inputs[v.name] = inputs[v.name].replace(cat_mapping)
                    cat_mappings[v.name] = cat_mapping
                    cat_dimensions.append(i)

        # Add column to inputs indicating task
        task_data = data["task"].dropna().to_numpy()
        if task_data.shape[0] != data.shape[0]:
            raise ValueError("Pretraining data must have a task for every row.")
        task_data = np.atleast_2d(task_data).T
        inputs_task = np.append(
            inputs.data_to_numpy().astype(float), task_data, axis=1
        ).astype(np.float)

        # Make it always a maximization problem
        objective = self.domain.output_variables[0]
        if not objective.maximize:
            output = -1.0 * output
        fbest_scaled = output[objective.name].max()

        # Train model
        n_tasks = int(task_data.max()) + 1
        output_tasks = [
            self.task
        ]  # if self.acquistion_function != "WeightedEI" else list(range(n_tasks))
        # sd_prior = GammaPrior(1.0, 0.15)
        # sd_prior._event_shape = torch.Size([n_tasks])
        # eta = 1.5
        # if not isinstance(eta, float) and not isinstance(eta, int):
        #     raise ValueError(f"eta must be a real number, your eta was {eta}.")
        # prior = LKJCovariancePrior(n_tasks, eta, sd_prior)

        if self.brute_force_categorical and self.categorical_method is None:
            raise NotImplementedError("LCM model not implemented yet.")
            # self.model = MixedMultiTaskGP(
            #     torch.tensor(inputs_task).double(),
            #     torch.tensor(output.data_to_numpy().astype(float)).double(),
            #     cat_dims=cat_dimensions,
            #     task_feature=-1,
            #     output_tasks=output_tasks,
            #     task_covar_prior=prior,
            # )
        elif self.model_type == self.LCM:
            raise NotImplementedError("LCM model not implemented yet.")
            # self.model = LCMMultitaskGP(
            #     torch.tensor(inputs_task).double(),
            #     torch.tensor(output.data_to_numpy().astype(float)).double(),
            #     task_feature=-1,
            #     output_tasks=output_tasks,
            #     num_independent_kernels=2,
            #     task_covar_prior=prior,
            # )
        elif self.model_type == self.ICM:
            self.model = MultiTaskGP(
                torch.tensor(inputs_task).double(),
                torch.tensor(output.data_to_numpy().astype(float)).double(),
                task_feature=-1,
                output_tasks=output_tasks,
                # task_covar_prior=prior,
            )
        else:
            raise ValueError(f"{self.model_type} not available")

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(self.mll, max_retries=20)

        # Train an extra model for the current task
        if self.acquistion_function == "WeightedEI":
            self.task_model = SingleTaskGP(
                torch.tensor(inputs.data_to_numpy().astype(float)).double(),
                torch.tensor(output.data_to_numpy().astype(float)).double(),
            )
            self.mll = ExactMarginalLogLikelihood(
                self.task_model.likelihood, self.task_model
            )
            fit_gpytorch_model(self.mll)

        # Optimize acquisition function
        if self.brute_force_categorical:
            if self.acquistion_function == "EI":
                self.acq = EI(self.model, best_f=fbest_scaled, maximize=True)
            elif self.acquistion_function == "qNEI":
                self.acq = qNEI(
                    self.model,
                    X_baseline=torch.tensor(inputs_task[:, :-1]).double(),
                )
            elif self.acquistion_function == "WeightedEI":
                self.acq = WeightedEI(
                    models=[self.task_model, self.model],
                    task=0,
                    best_f=fbest_scaled,
                    maximize=True,
                    weights=[1e4, 1],
                )
            elif self.acquistion_function == "UCB":
                self.acq = UCB(self.model, beta=0.8)
            else:
                raise ValueError(
                    f"{self.acquistion_function} not a valid acquisition function"
                )

            if self.categorical_method is None:
                combos = self.domain.get_categorical_combinations()
                fixed_features_list = []
                for v in self.domain.input_variables:
                    if v.variable_type == "categorical":
                        combos[v.name] = combos[v.name].replace(cat_mappings[v.name])
                fixed_features_list = []
                for k, combo in combos.iterrows():
                    fixed_features_list.append(
                        {dim: combo[i] for i, dim in enumerate(cat_dimensions)}
                    )
            else:
                fixed_features_list = self._get_fixed_features()
            results, _ = optimize_acqf_mixed(
                acq_function=self.acq,
                bounds=self._get_bounds(),
                num_restarts=kwargs.get("num_restarts", 100),
                fixed_features_list=fixed_features_list,
                q=num_experiments,
                raw_samples=kwargs.get("raw_samples", 2000),
            )
        else:
            if self.acquistion_function == "EI":
                self.acq = CategoricalEI(
                    self.domain, self.model, best_f=fbest_scaled, maximize=True
                )
            elif self.acquistion_function == "qNEI":
                self.acq = CategoricalqNEI(
                    self.domain,
                    self.model,
                    X_baseline=torch.tensor(inputs_task[:, :-1]).double(),
                )
            else:
                raise ValueError(
                    f"{self.acquistion_function} not a valid acquisition function"
                )
            results, _ = optimize_acqf(
                acq_function=self.acq,
                bounds=self._get_bounds(),
                num_restarts=kwargs.get("num_restarts", 100),
                q=num_experiments,
                raw_samples=kwargs.get("raw_samples", 2000),
            )

        # Convert result to datset
        result = DataSet(
            results.detach().numpy(),
            columns=inputs.data_columns,
        )

        # Untransform
        if self.categorical_method is None:
            for i, v in enumerate(self.domain.input_variables):
                if v.variable_type == "categorical":
                    cat_mapping = {i: l for i, l in enumerate(v.levels)}
                    result[v.name] = result[v.name].replace(cat_mapping)

        result = self.transform.un_transform(
            result,
            categorical_method=self.categorical_method,
            min_max_scale_inputs=True,
            standardize_outputs=True,
        )

        # Add metadata
        result[("strategy", "METADATA")] = "MTBO"
        result[("task", "METADATA")] = self.task
        return result

    def _get_fixed_features(self):
        combos = self.domain.get_categorical_combinations()
        encoded_combos = {
            v.name: self.transform.encoders[v.name].transform(combos[[v.name]])
            for v in self.domain.input_variables
            if v.variable_type == "categorical"
        }
        fixed_features_list = []
        for i in range(len(combos)):
            fixed_features = {}
            k = 0
            for v in self.domain.input_variables:
                # One-hot encoding
                if v.variable_type == "categorical":
                    for j in range(encoded_combos[v.name].shape[1]):
                        fixed_features[k] = float(encoded_combos[v.name][i, j])
                        k += 1
                else:
                    k += 1
            fixed_features_list.append(fixed_features)
        return fixed_features_list

    def _get_bounds(self):
        bounds = []
        for v in self.domain.input_variables:
            if isinstance(v, ContinuousVariable):
                var_min, var_max = v.bounds[0], v.bounds[1]
                # mean = self.transform.input_means[v.name]
                # std = self.transform.input_stds[v.name]
                v_bounds = np.array(v.bounds)
                # v_bounds = (v_bounds - mean) / std
                v_bounds = (v_bounds - var_min) / (var_max - var_min)
                bounds.append(v_bounds)
            elif (
                isinstance(v, CategoricalVariable)
                and self.categorical_method == "one-hot"
            ):
                bounds += [[0, 1] for _ in v.levels]
            elif isinstance(v, CategoricalVariable) and self.categorical_method is None:
                bounds.append([0, len(v.levels)])
        return torch.tensor(bounds).T.double()

    def reset(self):
        """Reset MTBO state"""
        self.all_experiments = None
        self.iterations = 0
        self.fbest = (
            float("inf") if self.domain.output_variables[0].maximize else -float("inf")
        )

    @staticmethod
    def standardize(X):
        mean, std = X.mean(), X.std()
        std[std < 1e-5] = 1e-5
        scaled = (X - mean.to_numpy()) / std.to_numpy()
        return scaled.to_numpy(), mean, std

    def to_dict(self):
        ae = (
            self.all_experiments.to_dict() if self.all_experiments is not None else None
        )
        strategy_params = dict(
            all_experiments=ae,
            categorical_method=self.categorical_method,
            task=self.task,
        )
        return super().to_dict(**strategy_params)

    @classmethod
    def from_dict(cls, d):
        exp = super().from_dict(d)
        ae = d["strategy_params"]["all_experiments"]
        exp.all_experiments = ae
        return exp
