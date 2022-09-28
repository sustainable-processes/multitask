from typing import List, Union
from summit import *
from botorch.acquisition import ExpectedImprovement as EI
from botorch.acquisition import qNoisyExpectedImprovement as qNEI
from botorch.models.model import Model
import torch
from torch import Tensor

dtype = torch.double


class CategoricalEI(EI):
    def __init__(
        self,
        domain: Domain,
        model,
        best_f,
        objective=None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model, best_f=best_f, objective=objective, maximize=maximize, **kwargs
        )
        self._domain = domain

    def forward(self, X):
        X = self.round_to_one_hot(X, self._domain)
        return super().forward(X)

    @staticmethod
    def round_to_one_hot(X, domain: Domain):
        """Round all categorical variables to a one-hot encoding"""
        num_experiments = X.shape[1]
        X = X.clone()
        for q in range(num_experiments):
            c = 0
            for v in domain.input_variables:
                if isinstance(v, CategoricalVariable):
                    n_levels = len(v.levels)
                    levels_selected = X[:, q, c : c + n_levels].argmax(axis=1)
                    X[:, q, c : c + n_levels] = 0
                    for j, l in zip(range(X.shape[0]), levels_selected):
                        X[j, q, int(c + l)] = 1

                    check = int(X[:, q, c : c + n_levels].sum()) == X.shape[0]
                    if not check:
                        raise ValueError(
                            (
                                f"Rounding to a one-hot encoding is not properly working. Please report this bug at "
                                f"https://github.com/sustainable-processes/summit/issues. Tensor: \n {X[:, :, c : c + n_levels]}"
                            )
                        )
                    c += n_levels
                else:
                    c += 1
        return X


class WeightedEI(EI):
    def __init__(
        self,
        models: List[Model],
        task: int,
        best_f: Union[float, Tensor],
        maximize: bool = True,
        weights=None,
        **kwargs,
    ):

        # could also implement weighting as a kw
        self.models = models
        self.task = task  # active task
        self.maximize = maximize
        n = len(models)
        self.weights = weights if weights != None else [1 / n] * n
        super().__init__(model=models[task], best_f=best_f, **kwargs)

    def forward(self, X):
        out = torch.zeros_like(X[:, 0, 0])
        for i, model in enumerate(self.models):
            if i == self.task:
                # Expected Improvement
                val = super().forward(X) * self.weights[i]
                # print(f"EI: {val}")
                out += val
            else:
                # Improvement
                self.best_f = self.best_f.to(X)
                posterior = model.posterior(
                    X=X,  # posterior_transform=self.posterior_transform
                )
                mean = posterior.mean
                diff = (mean - self.best_f).squeeze()
                if self.maximize:
                    # val = torch.clip(diff, min=0) * self.weights[i]
                    val = diff * self.weights[i]
                else:  # if we're minimising
                    val = torch.clip(diff, max=0) * self.weights[i]
                # print(f"Improvement: {val}")
                out += val

        out /= sum(self.weights)
        return out


class CategoricalqNEI(qNEI):
    def __init__(
        self,
        domain: Domain,
        model,
        X_baseline,
        **kwargs,
    ) -> None:
        super().__init__(model, X_baseline, **kwargs)
        self._domain = domain

    def forward(self, X):
        X = self.round_to_one_hot(X, self._domain)
        return super().forward(X)

    @staticmethod
    def round_to_one_hot(X, domain: Domain):
        """Round all categorical variables to a one-hot encoding"""
        num_experiments = X.shape[1]
        X = X.clone()
        for q in range(num_experiments):
            c = 0
            for v in domain.input_variables:
                if isinstance(v, CategoricalVariable):
                    n_levels = len(v.levels)
                    levels_selected = X[:, q, c : c + n_levels].argmax(axis=1)
                    X[:, q, c : c + n_levels] = 0
                    for j, l in zip(range(X.shape[0]), levels_selected):
                        X[j, q, int(c + l)] = 1

                    check = int(X[:, q, c : c + n_levels].sum()) == X.shape[0]
                    if not check:
                        raise ValueError(
                            (
                                f"Rounding to a one-hot encoding is not properly working. Please report this bug at "
                                f"https://github.com/sustainable-processes/summit/issues. Tensor: \n {X[:, :, c : c + n_levels]}"
                            )
                        )
                    c += n_levels
                else:
                    c += 1
        return X
