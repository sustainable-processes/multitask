from .suzuki_emulator import SuzukiEmulator
from .suzuki_benchmark_training import train_benchmark as train_suzuki_benchmark
import typer
from typing import Optional

app = typer.Typer()


@app.command()
def train_suzuki(
    data_path: str,
    save_path: str,
    figure_path: str,
    dataset_name: Optional[str] = None,
    include_reactant_concentrations: Optional[bool] = False,
    print_warnings: Optional[bool] = True,
    split_catalyst: Optional[bool] = True,
    max_epochs: Optional[int] = 1000,
    cv_folds: Optional[int] = 5,
    verbose: Optional[int] = 0,
) -> SuzukiEmulator:
    """Train a Suzuki benchmark"""
    return train_suzuki_benchmark(
        data_path=data_path,
        save_path=save_path,
        figure_path=figure_path,
        dataset_name=dataset_name,
        include_reactant_concentrations=include_reactant_concentrations,
        print_warnings=print_warnings,
        split_catalyst=split_catalyst,
        max_epochs=max_epochs,
        cv_folds=cv_folds,
        verbose=verbose,
    )


if __name__ == "__main__":
    app()
