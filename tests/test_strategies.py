import pytest

from summit import *
from multitask import *

import GPy
from fastprogress.fastprogress import progress_bar
import numpy as np
import os


@pytest.mark.parametrize(
    "max_num_exp, maximize, constraint",
    [
        # [50, True, True],
        # [50, False, True],
        [20, True, False],
        [20, False, False],
    ],
)
def test_stbo(
    max_num_exp,
    maximize,
    constraint,
    plot=False,
):

    hartmann3D = Hartmann3D(maximize=maximize, constraints=constraint)
    strategy = NewSTBO(domain=hartmann3D.domain, task=1)
    batch_size = 1

    hartmann3D.reset()
    r = Runner(
        strategy=strategy,
        experiment=hartmann3D,
        batch_size=batch_size,
        max_iterations=max_num_exp // batch_size,
    )
    r.run()

    objective = hartmann3D.domain.output_variables[0]
    data = hartmann3D.data
    if objective.maximize:
        fbest = data[objective.name].max()
    else:
        fbest = data[objective.name].min()

    fbest = np.around(fbest, decimals=2)
    print(f"Number of experiments: {data.shape[0]}")
    # Extrema of test function without constraint: glob_min = -3.86 at
    if maximize:
        assert fbest >= 3.5
    else:
        assert fbest <= -3.5

    # Test saving and loading
    strategy.save("stbo_test.json")
    strategy_2 = NewSTBO.load("stbo_test.json")
    os.remove("stbo_test.json")

    if plot:
        fig, ax = hartmann3D.plot()


@pytest.mark.parametrize(
    "max_num_exp, maximize, constraint",
    [
        # [50, True, True],
        # [50, False, True],
        [20, True, False],
        [20, False, False],
    ],
)
def test_mtbo(
    max_num_exp,
    maximize,
    constraint,
    plot=False,
    n_pretraining=50,
):

    hartmann3D = Hartmann3D(maximize=maximize, constraints=constraint)
    # Pretraining data
    random = Random(hartmann3D.domain)
    conditions = random.suggest_experiments(n_pretraining)
    results = hartmann3D.run_experiments(conditions)
    for v in hartmann3D.domain.output_variables:
        results[v.name] = 1.5 * results[v.name]
    results["task", "METADATA"] = 0
    strategy = NewMTBO(domain=hartmann3D.domain, pretraining_data=results, task=1)
    batch_size = 1

    hartmann3D.reset()
    r = Runner(
        strategy=strategy,
        experiment=hartmann3D,
        batch_size=batch_size,
        max_iterations=max_num_exp // batch_size,
    )
    r.run()

    objective = hartmann3D.domain.output_variables[0]
    data = hartmann3D.data
    if objective.maximize:
        fbest = data[objective.name].max()
    else:
        fbest = data[objective.name].min()

    fbest = np.around(fbest, decimals=2)
    print(f"Number of experiments: {data.shape[0]}")
    # Extrema of test function without constraint: glob_min = -3.86 at
    if maximize:
        assert fbest >= 3.5
    else:
        assert fbest <= -3.5

    # Test saving and loading
    strategy.save("mtbo_test.json")
    strategy_2 = NewMTBO.load("mtbo_test.json")
    os.remove("mtbo_test.json")

    if plot:
        fig, ax = hartmann3D.plot()
