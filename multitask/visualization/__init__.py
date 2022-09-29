from . import suzuki_figures
import typer

app = typer.Typer()
app.command()(suzuki_figures.baumgartner_suzuki_auxiliary_reizman_suzuki)
app.command()(suzuki_figures.reizman_suzuki_auxiliary_baumgartner_suzuki)
app.command()(suzuki_figures.reizman_suzuki_auxiliary_reizman_suzuki)

if __name__ == "__main__":
    app()
