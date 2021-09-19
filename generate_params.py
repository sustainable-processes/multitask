from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import typer


def one_cotraining_suzuki():
    """Single cotraining task"""
    # baumgartner_reizman_one_experiments
    experiments = {
        f"baumgartner_suzuki_cotrain_reizman_suzuki_case_{i}": {
            "model_name": "baumgartner_suzuki",
            "benchmark": "data/baumgartner_suzuki/emulator",
            "dataset_1": f"data/reizman_suzuki/ord/reizman_suzuki_case_{i}.pb",
        }
        for i in range(1, 5)
    }

    # reizman_baumgartner_one_experiments
    experiments.update(
        {
            f"reizman_suzuki_{i}_cotrain_baumgartner_suzuki": {
                "model_name": f"reizman_suzuki_case_{i}",
                "benchmark": f"data/reizman_suzuki/emulator_case_{i}",
                "dataset_1": f"data/baumgartner_suzuki/ord/baumgartner_suzuki.pb",
            }
            for i in range(1, 5)
        }
    )

    # reizman_reizman_one_experiments
    experiments.update(
        {
            f"reizman_suzuki_case_{i}_cotrain_reizman_suzuki_case_{j}": {
                "model_name": f"reizman_suzuki_case_{i}",
                "benchmark": f"data/reizman_suzuki/emulator_case_{i}",
                "dataset_1": f"data/reizman_suzuki/ord/reizman_suzuki_case_{j}.pb",
            }
            for i in range(1, 5)
            for j in range(1, 5)
            if i != j
        }
    )

    return experiments


def one_cotraining_kinetic():
    """Single cotraining task"""
    # baumgartner_reizman_one_experiments
    parameters = {
        "case": [1,2,3,4,5],
        "ct_case": [1,2,3,4,5],
        "ct_strategy": ["STBO", "LHS"],
        "noise_level": [0.0, 1.0, 10.0],
        "ct_noise_level": [0.0, 1.0, 10.0],
        "num_initial_experiments": [0, 5],
        "ct_num_initial_experiments": [0, 5, 10],
        "max_ct_experiments": [20, 50],
        "acquisition_function": ["qNEI", "EI"]
    }
    experiments = {
        f"baumgartner_suzuki_cotrain_reizman_suzuki_case_{i}": {
            "model_name": "baumgartner_suzuki",
            "benchmark": "data/baumgartner_suzuki/emulator",
            "dataset_1": f"data/reizman_suzuki/ord/reizman_suzuki_case_{i}.pb",
        }
        for i in range(1, 5)
    }

    # reizman_reizman_one_experiments
    experiments.update(
        {
            f"reizman_suzuki_case_{i}_cotrain_reizman_suzuki_case_{j}": {
                "model_name": f"reizman_suzuki_case_{i}",
                "benchmark": f"data/reizman_suzuki/emulator_case_{i}",
                "dataset_1": f"data/reizman_suzuki/ord/reizman_suzuki_case_{j}.pb",
            }
            for i in range(1, 5)
            for j in range(1, 5)
            if i != j
        }
    )

    return experiments


def two_cotraining():
    pass


def fullfact(levels):
    """
    Copied from pyDOE2
    Create a general full-factorial design

    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.

    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels 0 to k-1 for a k-level factor

    Example
    -------
    ::

        >>> fullfact([2, 4, 3])
        array([[ 0.,  0.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 1.,  1.,  0.],
               [ 0.,  2.,  0.],
               [ 1.,  2.,  0.],
               [ 0.,  3.,  0.],
               [ 1.,  3.,  0.],
               [ 0.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 0.,  2.,  1.],
               [ 1.,  2.,  1.],
               [ 0.,  3.,  1.],
               [ 1.,  3.,  1.],
               [ 0.,  0.,  2.],
               [ 1.,  0.,  2.],
               [ 0.,  1.,  2.],
               [ 1.,  1.,  2.],
               [ 0.,  2.,  2.],
               [ 1.,  2.,  2.],
               [ 0.,  3.,  2.],
               [ 1.,  3.,  2.]])

    """
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))

    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j] * level_repeat
        rng = lvl * range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng

    return H


def main(params_file="params.yaml"):
    with open(params_file, "r") as f:
        params = load(f.read(), Loader=Loader)
    params.update({"suzuki_one_experiments": one_cotraining_suzuki()})
    with open(params_file, "w") as f:
        f.write(dump(params, Dumper=Dumper, default_flow_style=False))


if __name__ == "__main__":
    typer.run(main)
