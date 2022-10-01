# #!/usr/bin/env python3
# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# from typing import Callable
# from typing import Dict, List, Optional, Any, Tuple

# import torch
# from botorch.exceptions.errors import UnsupportedError
# from botorch.models.gp_regression import SingleTaskGP
# from botorch.models.kernels.categorical import CategoricalKernel
# from botorch.models.transforms.input import InputTransform
# from botorch.models.transforms.outcome import OutcomeTransform
# from botorch.utils.containers import TrainingData
# from botorch.utils.transforms import normalize_indices
# from gpytorch.constraints import GreaterThan
# from gpytorch.kernels.kernel import Kernel
# from gpytorch.kernels.matern_kernel import MaternKernel
# from gpytorch.kernels.scale_kernel import ScaleKernel
# from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
# from gpytorch.likelihoods.likelihood import Likelihood
# from gpytorch.priors import GammaPrior
# from torch import Tensor

# from botorch.models.gpytorch import (
#     MultiTaskGPyTorchModel,
#     BatchedMultiOutputGPyTorchModel,
# )
# from botorch.models.transforms.input import InputTransform
# from botorch.utils.containers import TrainingData
# from gpytorch.distributions.multivariate_normal import MultivariateNormal
# from gpytorch.kernels.index_kernel import IndexKernel
# from gpytorch.kernels.matern_kernel import MaternKernel
# from gpytorch.kernels.scale_kernel import ScaleKernel
# from gpytorch.likelihoods.gaussian_likelihood import (
#     FixedNoiseGaussianLikelihood,
#     GaussianLikelihood,
# )
# from gpytorch.means.constant_mean import ConstantMean
# from gpytorch.models.exact_gp import ExactGP
# from gpytorch.priors.lkj_prior import LKJCovariancePrior
# from gpytorch.priors.prior import Prior
# from gpytorch.priors.torch_priors import GammaPrior
# from gpytorch.module import Module
# from torch import Tensor


# class MixedMultiTaskGP(
#     ExactGP,
#     MultiTaskGPyTorchModel,
# ):
#     r"""Mixed Multi-Task GP model using an ICM kernel, inferring observation noise.
#     Multi-task exact GP that uses a simple ICM kernel. Can be single-output or
#     multi-output. This model uses relatively strong priors on the base Kernel
#     hyperparameters, which work best when covariates are normalized to the unit
#     cube and outcomes are standardized (zero mean, unit variance).
#     This model infers the noise level. WARNING: It currently does not support
#     different noise levels for the different tasks. If you have known observation
#     noise, please use `FixedNoiseMultiTaskGP` instead.
#     """

#     def __init__(
#         self,
#         train_X: Tensor,
#         train_Y: Tensor,
#         task_feature: int,
#         cat_dims: List[int],
#         cont_kernel_factory: Optional[Callable[[int, List[int]], Kernel]] = None,
#         task_covar_prior: Optional[Prior] = None,
#         output_tasks: Optional[List[int]] = None,
#         rank: Optional[int] = None,
#         input_transform: Optional[InputTransform] = None,
#     ) -> None:
#         r"""Mixed Multi-Task GP model using an ICM kernel, inferring observation noise.
#         Args:
#             train_X: A `n x (d + 1)` or `b x n x (d + 1)` (batch mode) tensor
#                 of training data. One of the columns should contain the task
#                 features (see `task_feature` argument).
#             train_Y: A `n` or `b x n` (batch mode) tensor of training observations.
#             task_feature: The index of the task feature (`-d <= task_feature <= d`).
#             cat_dims: A list of indices corresponding to the columns of
#                 the input `X` that should be considered categorical features.
#             cont_kernel_factory: A method that accepts `ard_num_dims` and
#                 `active_dims` arguments and returns an instatiated GPyTorch
#                 `Kernel` object to be used as the ase kernel for the continuous
#                 dimensions. If omitted, this model uses a Matern-2.5 kernel as
#                 the kernel for the ordinal parameters.
#             output_tasks: A list of task indices for which to compute model
#                 outputs for. If omitted, return outputs for all task indices.
#             rank: The rank to be used for the index kernel. If omitted, use a
#                 full rank (i.e. number of tasks) kernel.
#             task_covar_prior : A Prior on the task covariance matrix. Must operate
#                 on p.s.d. matrices. A common prior for this is the `LKJ` prior.
#             input_transform: An input transform that is applied in the model's
#                 forward pass.
#         Example:
#             >>> X1, X2 = torch.rand(10, 2), torch.rand(20, 2)
#             >>> i1, i2 = torch.zeros(10, 1), torch.ones(20, 1)
#             >>> train_X = torch.cat([
#             >>>     torch.cat([X1, i1], -1), torch.cat([X2, i2], -1),
#             >>> ])
#             >>> train_Y = torch.cat(f1(X1), f2(X2)).unsqueeze(-1)
#             >>> model = MultiTaskGP(train_X, train_Y, task_feature=-1)
#         """
#         if input_transform is not None:
#             input_transform.to(train_X)
#         with torch.no_grad():
#             transformed_X = self.transform_inputs(
#                 X=train_X, input_transform=input_transform
#             )
#         self._validate_tensor_args(X=transformed_X, Y=train_Y)
#         all_tasks, task_feature, d = self.get_all_tasks(
#             transformed_X, task_feature, output_tasks
#         )
#         input_batch_shape, aug_batch_shape = self.get_batch_dimensions(
#             train_X=train_X, train_Y=train_Y
#         )
#         # squeeze output dim
#         train_Y = train_Y.squeeze(-1)
#         if output_tasks is None:
#             output_tasks = all_tasks
#         else:
#             if set(output_tasks) - set(all_tasks):
#                 raise RuntimeError("All output tasks must be present in input data.")
#         self._output_tasks = output_tasks
#         self._num_outputs = len(output_tasks)

#         # TODO (T41270962): Support task-specific noise levels in likelihood
#         min_noise = 1e-5 if train_X.dtype == torch.float else 1e-6
#         likelihood = GaussianLikelihood(
#             noise_prior=GammaPrior(0.9, 10.0),
#             noise_constraint=GreaterThan(min_noise, transform=None, initial_value=1e-3),
#         )

#         # construct indexer to be used in forward
#         self._task_feature = task_feature
#         self._base_idxr = torch.arange(d)
#         self._base_idxr[task_feature:] += 1  # exclude task feature

#         super().__init__(
#             train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
#         )

#         if cont_kernel_factory is None:

#             def cont_kernel_factory(
#                 batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
#             ) -> MaternKernel:
#                 return MaternKernel(
#                     nu=2.5,
#                     batch_shape=batch_shape,
#                     ard_num_dims=ard_num_dims,
#                     active_dims=active_dims,
#                 )

#         self.mean_module = ConstantMean()
#         cat_dims = normalize_indices(indices=cat_dims, d=d)
#         ord_dims = sorted(set(range(d)) - set(cat_dims))
#         if len(ord_dims) == 0:
#             self.covar_module = ScaleKernel(
#                 CategoricalKernel(
#                     batch_shape=aug_batch_shape,
#                     ard_num_dims=len(cat_dims),
#                 )
#             )
#         else:
#             sum_kernel = ScaleKernel(
#                 cont_kernel_factory(
#                     batch_shape=aug_batch_shape,
#                     ard_num_dims=len(ord_dims),
#                     active_dims=ord_dims,
#                 )
#                 + ScaleKernel(
#                     CategoricalKernel(
#                         batch_shape=aug_batch_shape,
#                         ard_num_dims=len(cat_dims),
#                         active_dims=cat_dims,
#                     )
#                 )
#             )
#             prod_kernel = ScaleKernel(
#                 cont_kernel_factory(
#                     batch_shape=aug_batch_shape,
#                     ard_num_dims=len(ord_dims),
#                     active_dims=ord_dims,
#                 )
#                 * CategoricalKernel(
#                     batch_shape=aug_batch_shape,
#                     ard_num_dims=len(cat_dims),
#                     active_dims=cat_dims,
#                 )
#             )
#             self.covar_module = sum_kernel + prod_kernel

#         num_tasks = len(all_tasks)
#         self._rank = rank if rank is not None else num_tasks

#         self.task_covar_module = IndexKernel(
#             num_tasks=num_tasks, rank=self._rank, prior=task_covar_prior
#         )
#         if input_transform is not None:
#             self.input_transform = input_transform
#         self.to(train_X)

#     def _split_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
#         r"""Extracts base features and task indices from input data.
#         Args:
#             x: The full input tensor with trailing dimension of size `d + 1`.
#                 Should be of float/double data type.
#         Returns:
#             2-element tuple containing
#             - A `q x d` or `b x q x d` (batch mode) tensor with trailing
#             dimension made up of the `d` non-task-index columns of `x`, arranged
#             in the order as specified by the indexer generated during model
#             instantiation.
#             - A `q` or `b x q` (batch mode) tensor of long data type containing
#             the task indices.
#         """
#         batch_shape, d = x.shape[:-2], x.shape[-1]
#         x_basic = x[..., self._base_idxr].view(batch_shape + torch.Size([-1, d - 1]))
#         task_idcs = (
#             x[..., self._task_feature]
#             .view(batch_shape + torch.Size([-1, 1]))
#             .to(dtype=torch.long)
#         )
#         return x_basic, task_idcs

#     def forward(self, x: Tensor) -> MultivariateNormal:
#         if self.training:
#             x = self.transform_inputs(x)
#         x_basic, task_idcs = self._split_inputs(x)
#         # Compute base mean and covariance
#         mean_x = self.mean_module(x_basic)
#         covar_x = self.covar_module(x_basic)
#         # Compute task covariances
#         covar_i = self.task_covar_module(task_idcs)
#         # Combine the two in an ICM fashion
#         covar = covar_x.mul(covar_i)
#         return MultivariateNormal(mean_x, covar)

#     @classmethod
#     def get_all_tasks(
#         cls,
#         train_X: Tensor,
#         task_feature: int,
#         output_tasks: Optional[List[int]] = None,
#     ) -> Tuple[List[int], int, int]:
#         if train_X.ndim != 2:
#             # Currently, batch mode MTGPs are blocked upstream in GPyTorch
#             raise ValueError(f"Unsupported shape {train_X.shape} for train_X.")
#         d = train_X.shape[-1] - 1
#         if not (-d <= task_feature <= d):
#             raise ValueError(f"Must have that -{d} <= task_feature <= {d}")
#         task_feature = task_feature % (d + 1)
#         all_tasks = train_X[:, task_feature].unique().to(dtype=torch.long).tolist()
#         return all_tasks, task_feature, d

#     @staticmethod
#     def get_batch_dimensions(
#         train_X: Tensor, train_Y: Tensor
#     ) -> Tuple[torch.Size, torch.Size]:
#         r"""Get the raw batch shape and output-augmented batch shape of the inputs.
#         Args:
#             train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
#                 features.
#             train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
#                 training observations.
#         Returns:
#             2-element tuple containing
#             - The `input_batch_shape`
#             - The output-augmented batch shape: `input_batch_shape x (m)`
#         """
#         input_batch_shape = train_X.shape[:-2]
#         aug_batch_shape = input_batch_shape
#         num_outputs = train_Y.shape[-1]
#         if num_outputs > 1:
#             aug_batch_shape += torch.Size([num_outputs])
#         return input_batch_shape, aug_batch_shape

#     @classmethod
#     def construct_inputs(cls, training_data: TrainingData, **kwargs) -> Dict[str, Any]:
#         r"""Construct kwargs for the `Model` from `TrainingData` and other options.
#         Args:
#             training_data: `TrainingData` container with data for single outcome
#                 or for multiple outcomes for batched multi-output case.
#             **kwargs: Additional options for the model that pertain to the
#                 training data, including:
#                 - `task_features`: Indices of the input columns containing the task
#                   features (expected list of length 1),
#                 - `task_covar_prior`: A GPyTorch `Prior` object to use as prior on
#                   the cross-task covariance matrix,
#                 - `prior_config`: A dict representing a prior config, should only be
#                   used if `prior` is not passed directly. Should contain:
#                   `use_LKJ_prior` (whether to use LKJ prior) and `eta` (eta value,
#                   float),
#                 - `rank`: The rank of the cross-task covariance matrix.
#         """

#         task_features = kwargs.pop("task_features", None)
#         if task_features is None:
#             raise ValueError(f"`task_features` required for {cls.__name__}.")
#         task_feature = task_features[0]
#         inputs = {
#             "train_X": training_data.X,
#             "train_Y": training_data.Y,
#             "task_feature": task_feature,
#             "rank": kwargs.get("rank"),
#         }

#         prior = kwargs.get("task_covar_prior")
#         prior_config = kwargs.get("prior_config")
#         if prior and prior_config:
#             raise ValueError(
#                 "Only one of `prior` and `prior_config` arguments expected."
#             )

#         if prior_config:
#             if not prior_config.get("use_LKJ_prior"):
#                 raise ValueError("Currently only config for LKJ prior is supported.")
#             all_tasks, _, _ = MultiTaskGP.get_all_tasks(training_data.X, task_feature)
#             num_tasks = len(all_tasks)
#             sd_prior = GammaPrior(1.0, 0.15)
#             sd_prior._event_shape = torch.Size([num_tasks])
#             eta = prior_config.get("eta", 0.5)
#             if not isinstance(eta, float) and not isinstance(eta, int):
#                 raise ValueError(f"eta must be a real number, your eta was {eta}.")
#             prior = LKJCovariancePrior(num_tasks, eta, sd_prior)

#         inputs["task_covar_prior"] = prior
#         return inputs


# class LCMMultitaskGP(ExactGP, MultiTaskGPyTorchModel):
#     """Use LCM kernel instead of ICM and see performance

#     https://docs.gpytorch.ai/en/stable/kernels.html#gpytorch.kernels.LCMKernel

#     """

#     r"""Mixed Multi-Task GP model using an LCM kernel, inferring observation noise.
#     """

#     def __init__(
#         self,
#         train_X: Tensor,
#         train_Y: Tensor,
#         task_feature: int,
#         covar_modules: List[Optional[Module]] = None,
#         num_independent_kernels: Optional[int] = 2,
#         task_covar_prior: Optional[Prior] = None,
#         output_tasks: Optional[List[int]] = None,
#         rank: Optional[int] = None,
#         input_transform: Optional[InputTransform] = None,
#         outcome_transform: Optional[OutcomeTransform] = None,
#     ) -> None:
#         r"""Multi-Task GP model using an ICM kernel, inferring observation noise.
#         Args:
#             train_X: A `n x (d + 1)` or `b x n x (d + 1)` (batch mode) tensor
#                 of training data. One of the columns should contain the task
#                 features (see `task_feature` argument).
#             train_Y: A `n x 1` or `b x n x 1` (batch mode) tensor of training
#                 observations.
#             task_feature: The index of the task feature (`-d <= task_feature <= d`).
#             output_tasks: A list of task indices for which to compute model
#                 outputs for. If omitted, return outputs for all task indices.
#             rank: The rank to be used for the index kernel. If omitted, use a
#                 full rank (i.e. number of tasks) kernel.
#             task_covar_prior : A Prior on the task covariance matrix. Must operate
#                 on p.s.d. matrices. A common prior for this is the `LKJ` prior.
#             input_transform: An input transform that is applied in the model's
#                 forward pass.
#         Example:
#             >>> X1, X2 = torch.rand(10, 2), torch.rand(20, 2)
#             >>> i1, i2 = torch.zeros(10, 1), torch.ones(20, 1)
#             >>> train_X = torch.cat([
#             >>>     torch.cat([X1, i1], -1), torch.cat([X2, i2], -1),
#             >>> ])
#             >>> train_Y = torch.cat(f1(X1), f2(X2)).unsqueeze(-1)
#             >>> model = MultiTaskGP(train_X, train_Y, task_feature=-1)
#         """
#         with torch.no_grad():
#             transformed_X = self.transform_inputs(
#                 X=train_X, input_transform=input_transform
#             )
#         self._validate_tensor_args(X=transformed_X, Y=train_Y)
#         all_tasks, task_feature, d = self.get_all_tasks(
#             transformed_X, task_feature, output_tasks
#         )
#         if outcome_transform is not None:
#             train_Y, _ = outcome_transform(train_Y)

#         # squeeze output dim
#         train_Y = train_Y.squeeze(-1)
#         if output_tasks is None:
#             output_tasks = all_tasks
#         else:
#             if set(output_tasks) - set(all_tasks):
#                 raise RuntimeError("All output tasks must be present in input data.")
#         self._output_tasks = output_tasks
#         self._num_outputs = len(output_tasks)

#         # TODO (T41270962): Support task-specific noise levels in likelihood
#         likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.1, 0.05))

#         # construct indexer to be used in forward
#         self._task_feature = task_feature
#         self._base_idxr = torch.arange(d)
#         self._base_idxr[task_feature:] += 1  # exclude task feature

#         super().__init__(
#             train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
#         )
#         self.mean_module = ConstantMean()

#         if covar_modules is None:
#             self.covar_modules = [
#                 ScaleKernel(
#                     base_kernel=MaternKernel(
#                         nu=2.5, ard_num_dims=d, lengthscale_prior=GammaPrior(3.0, 6.0)
#                     ),
#                     outputscale_prior=GammaPrior(2.0, 0.15),
#                 )
#                 for i in range(num_independent_kernels)
#             ]
#         else:
#             self.covar_modules = covar_modules

#         num_tasks = len(all_tasks)
#         self._rank = rank if rank is not None else num_tasks

#         self.task_covar_module = IndexKernel(
#             num_tasks=num_tasks, rank=self._rank, prior=task_covar_prior
#         )
#         if input_transform is not None:
#             self.input_transform = input_transform
#         if outcome_transform is not None:
#             self.outcome_transform = outcome_transform
#         self.to(train_X)

#     @staticmethod
#     def _create_covar_module(
#         cont_kernel_factory: callable,
#         aug_batch_shape: tuple,
#         ord_dims: list,
#         cat_dims: list,
#     ):
#         sum_kernel = ScaleKernel(
#             cont_kernel_factory(
#                 batch_shape=aug_batch_shape,
#                 ard_num_dims=len(ord_dims),
#                 active_dims=ord_dims,
#             )
#             + ScaleKernel(
#                 CategoricalKernel(
#                     batch_shape=aug_batch_shape,
#                     ard_num_dims=len(cat_dims),
#                     active_dims=cat_dims,
#                 )
#             )
#         )
#         prod_kernel = ScaleKernel(
#             cont_kernel_factory(
#                 batch_shape=aug_batch_shape,
#                 ard_num_dims=len(ord_dims),
#                 active_dims=ord_dims,
#             )
#             * CategoricalKernel(
#                 batch_shape=aug_batch_shape,
#                 ard_num_dims=len(cat_dims),
#                 active_dims=cat_dims,
#             )
#         )
#         return sum_kernel + prod_kernel

#     def _split_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
#         r"""Extracts base features and task indices from input data.
#         Args:
#             x: The full input tensor with trailing dimension of size `d + 1`.
#                 Should be of float/double data type.
#         Returns:
#             2-element tuple containing
#             - A `q x d` or `b x q x d` (batch mode) tensor with trailing
#             dimension made up of the `d` non-task-index columns of `x`, arranged
#             in the order as specified by the indexer generated during model
#             instantiation.
#             - A `q` or `b x q` (batch mode) tensor of long data type containing
#             the task indices.
#         """
#         batch_shape, d = x.shape[:-2], x.shape[-1]
#         x_basic = x[..., self._base_idxr].view(batch_shape + torch.Size([-1, d - 1]))
#         task_idcs = (
#             x[..., self._task_feature]
#             .view(batch_shape + torch.Size([-1, 1]))
#             .to(dtype=torch.long)
#         )
#         return x_basic, task_idcs

#     def forward(self, x: Tensor) -> MultivariateNormal:
#         if self.training:
#             x = self.transform_inputs(x)
#         x_basic, task_idcs = self._split_inputs(x)
#         # Compute base mean
#         mean_x = self.mean_module(x_basic)
#         # covar = sum([m(x_basic) for m in self.covar_modules])
#         covar = None
#         for m in self.covar_modules:
#             # Compute base covariance
#             # covar_x = m(x_basic)
#             # # Compute task covariances
#             # covar_i = self.task_covar_module(task_idcs)
#             # Combine the two in an ICM fashion
#             if covar is None:
#                 covar = covar_x.mul(covar_x)
#             else:
#                 covar += covar_x.mul(covar_x)
#         return MultivariateNormal(mean_x, covar)

#     @classmethod
#     def get_all_tasks(
#         cls,
#         train_X: Tensor,
#         task_feature: int,
#         output_tasks: Optional[List[int]] = None,
#     ) -> Tuple[List[int], int, int]:
#         if train_X.ndim != 2:
#             # Currently, batch mode MTGPs are blocked upstream in GPyTorch
#             raise ValueError(f"Unsupported shape {train_X.shape} for train_X.")
#         d = train_X.shape[-1] - 1
#         if not (-d <= task_feature <= d):
#             raise ValueError(f"Must have that -{d} <= task_feature <= {d}")
#         task_feature = task_feature % (d + 1)
#         all_tasks = train_X[:, task_feature].unique().to(dtype=torch.long).tolist()
#         return all_tasks, task_feature, d

#     @staticmethod
#     def get_batch_dimensions(
#         train_X: Tensor, train_Y: Tensor
#     ) -> Tuple[torch.Size, torch.Size]:
#         r"""Get the raw batch shape and output-augmented batch shape of the inputs.
#         Args:
#             train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
#                 features.
#             train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
#                 training observations.
#         Returns:
#             2-element tuple containing
#             - The `input_batch_shape`
#             - The output-augmented batch shape: `input_batch_shape x (m)`
#         """
#         input_batch_shape = train_X.shape[:-2]
#         aug_batch_shape = input_batch_shape
#         num_outputs = train_Y.shape[-1]
#         if num_outputs > 1:
#             aug_batch_shape += torch.Size([num_outputs])
#         return input_batch_shape, aug_batch_shape
