from summit import ExperimentalEmulator

from pathlib import Path
import json


class SuzukiEmulator(ExperimentalEmulator):
    """Standard experimental emulator with some extra features for Suzuki
    Train a machine learning model based on experimental data.
    The model acts a benchmark for testing optimisation strategies.
    Parameters
    ----------
    model_name : str
        Name of the model, ideally with no spaces
    domain : :class:`~summit.domain.Domain`
        The domain of the emulator
    dataset : :class:`~summit.dataset.Dataset`, optional
        Dataset used for training/validation
    regressor : :class:`torch.nn.Module`, optional
        Pytorch LightningModule class. Defaults to the ANNRegressor
    output_variable_names : str or list, optional
        The names of the variables that should be trained by the predictor.
        Defaults to all objectives in the domain.
    descriptors_features : list, optional
        A list of input categorical variable names that should be transformed
        into their descriptors instead of using one-hot encoding.
    clip : bool or list, optional
        Whether to clip predictions to the limits of
        the objectives in the domain. True (default) means
        clipping is activated for all outputs and False means
        it is not activated at all. A list of specific outputs to clip
        can also be passed.

    """

    def __init__(self, model_name, domain, split_catalyst=True, **kwargs):
        self.split_catalyst = split_catalyst
        super().__init__(model_name, domain, **kwargs)

    def save(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        with open(save_dir / f"{self.model_name}.json", "w") as f:
            d = self.to_dict()
            d["split_catalyst"] = self.split_catalyst
            json.dump(d, f)
        self.save_regressor(save_dir)

    @classmethod
    def load(cls, model_name, save_dir, **kwargs):
        save_dir = Path(save_dir)
        with open(save_dir / f"{model_name}.json", "r") as f:
            d = json.load(f)
        exp = cls.from_dict(d, **kwargs)
        exp.split_catalyst = d["split_catalyst"]
        exp.load_regressor(save_dir)
        return exp
