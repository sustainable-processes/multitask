from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import typer


def one_cotraining():
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


def two_cotraining():
    pass


def main(params_file="params.yaml"):
    with open(params_file, "r") as f:
        params = load(f.read(), Loader=Loader)
    params.update({"suzuki_one_experiments": one_cotraining()})
    with open(params_file, "w") as f:
        f.write(dump(params, Dumper=Dumper, default_flow_style=False))


if __name__ == "__main__":
    typer.run(main)
