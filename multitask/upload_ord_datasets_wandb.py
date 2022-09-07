import wandb
from etl_baumgartner_suzuki import main as proceess_baugmartner_suzuki
from etl_reizman_suzuki import main as process_reizman_suzuki


# Connect to wanbd
WANDB_SETTINGS = {"wandb_entity": "ceb-sre", "wandb_project": "multitask"}
run = wandb.init(
    entity=WANDB_SETTINGS["wandb_entity"],
    project=WANDB_SETTINGS["wandb_project"],
    job_type="processing",
)

# Process baumgartner Suzuki
proceess_baugmartner_suzuki(
    input_file="data/baumgartner_suzuki/c8re00032h2.xlsx",
    output_path="data/baumgartner_suzuki/ord",
)
baumgartner_artifact = wandb.Artifact("baumgartner_suzuki", type="dataset")
baumgartner_artifact.add_dir("data/baumgartner_suzuki/ord")
run.log_artifact(baumgartner_artifact)

# Process reizman Suzuki
process_reizman_suzuki(
    input_path="data/reizman_suzuki/", output_path="data/reizman_suzuki/ord/"
)
reizman_artifact = wandb.Artifact("reizman_suzuki", type="dataset")
reizman_artifact.add_dir("data/reizman_suzuki/ord")
run.log_artifact(reizman_artifact)
