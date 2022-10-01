import typer
from . import optimization

app = typer.Typer()

app.command()(optimization.mtbo)
app.command()(optimization.stbo)

if __name__ == "__main__":
    app()
