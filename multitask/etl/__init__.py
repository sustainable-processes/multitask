import typer
import wandb
from .etl_baumgartner_suzuki import main as etl_baumgartner_suzuki
from .etl_reizman_suzuki import main as etl_reizman_suzuki
from .etl_baumgartner_cn import main as etl_baumgartner_cn

app = typer.Typer()

app.command(name="baumgartner-suzuki")(etl_baumgartner_suzuki)
app.command(name="reizman-suzuki")(etl_reizman_suzuki)
app.command(name="baumgartner-cn")(etl_baumgartner_cn)


@app.command()
def all(
    upload_wandb: bool = True,
    wandb_entity: str = "ceb-sre",
    wandb_project: str = "multitask",
):
    """Extract all files and upload to wandb"""
    # Connect to wanbd
    if upload_wandb:
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            job_type="processing",
        )

    # Process Baumgartner C-N
    etl_baumgartner_cn(
        input_file="data/baumgartner_cn/op9b00236_si_002.xlsx",
        output_path="data/baumgartner_cn/ord/",
    )
    if upload_wandb:
        reizman_artifact = wandb.Artifact("baumgartner_cn", type="dataset")
        reizman_artifact.add_dir("data/baumgartner_cn/ord")
        run.log_artifact(reizman_artifact)

    # Process baumgartner Suzuki
    etl_baumgartner_suzuki(
        input_file="data/baumgartner_suzuki/c8re00032h2.xlsx",
        output_path="data/baumgartner_suzuki/ord",
    )
    if upload_wandb:
        baumgartner_artifact = wandb.Artifact("baumgartner_suzuki", type="dataset")
        baumgartner_artifact.add_dir("data/baumgartner_suzuki/ord")
        run.log_artifact(baumgartner_artifact)

    # Process baumgartner Suzuki
    etl_reizman_suzuki(
        input_path="data/reizman_suzuki/", output_path="data/reizman_suzuki/ord/"
    )
    if upload_wandb:
        reizman_artifact = wandb.Artifact("reizman_suzuki", type="dataset")
        reizman_artifact.add_dir("data/reizman_suzuki/ord")
        run.log_artifact(reizman_artifact)


if __name__ == "__main__":
    app()
