from . import suzuki_figures
from . import cn_figures
import typer

app = typer.Typer()
app.command()(suzuki_figures.baumgartner_suzuki_auxiliary_reizman_suzuki)
app.command()(suzuki_figures.reizman_suzuki_auxiliary_baumgartner_suzuki)
app.command()(suzuki_figures.reizman_suzuki_auxiliary_reizman_suzuki)
app.command()(suzuki_figures.all_suzuki)
app.command()(cn_figures.baumgartner_cn_auxiliary_one_baumgartner_cn)
app.command()(cn_figures.baumgartner_cn_auxiliary_all_baumgartner_cn)
app.command()(cn_figures.all_cn)

if __name__ == "__main__":
    app()
