import typer
from .etl_baumgartner_suzuki import main as etl_baumgartner_suzuki
from .etl_reizman_suzuki import main as etl_reizman_suzuki
from .etl_baumgartner_cn import main as etl_baumgartner_cn

app = typer.Typer()

app.command(name="baumgartner-suzuki")(etl_baumgartner_suzuki)
app.command(name="reizman-suzuki")(etl_reizman_suzuki)
app.command(name="baumgartner-cn")(etl_baumgartner_cn)

if __name__ == "__main__":
    app()
