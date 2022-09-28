import typer
from . import suzuki_optimization

app = typer.Typer()

app.command()(suzuki_optimization.mtbo)
app.command()(suzuki_optimization.stbo)

if __name__ == "__main__":
    app()
