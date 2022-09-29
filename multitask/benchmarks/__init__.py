from .suzuki_emulator import SuzukiEmulator
from .suzuki_benchmark_training import train_benchmark as train_suzuki_benchmark
from .cn_benchmark_training import train_benchmark as train_cn_benchmark
import typer

app = typer.Typer()

app.command(name="train-suzuki")(train_suzuki_benchmark)
app.command(name="train-cn")(train_cn_benchmark)

if __name__ == "__main__":
    app()
