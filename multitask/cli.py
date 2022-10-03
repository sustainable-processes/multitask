import typer
from multitask.etl import app as etl_cli
from multitask.benchmarks import app as benchmarks_cli
from multitask.optimization import app as optimization_cli
from multitask.visualization import app as visualization_cli
import logging

app = typer.Typer()
app.add_typer(
    etl_cli, name="etl", help="Extract, transform, and load data from spreadsheets"
)
app.add_typer(benchmarks_cli, name="benchmarks", help="Benchmark training")
app.add_typer(
    optimization_cli,
    name="optimization",
    help="Benchmarking of single-task and multi-task strategies",
)
app.add_typer(
    visualization_cli,
    name="viz",
    help="Visualize the results of the benchmarks",
)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler("multitask.log")
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # add the file handler to the logger
    logger.addHandler(handler)

    # Typer apps
    app()
