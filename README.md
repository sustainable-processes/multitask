# Multitask 

Optimizing reactions using multitask bayesian optimization. Case study with alcohol amination via borrowing hydrogen  plus large-scale in-silico comparison.

* Manuscript draft on [Overleaf](https://www.overleaf.com/project/608a83b48a501409d68c2f69)
* Kanban board on [Github](https://github.com/sustainable-processes/multitask/projects)

## Relevant References

- [Fast continuous alcohol amination employing a hydrogen borrowing protocol](https://pubs.rsc.org/en/content/articlelanding/2019/gc/c8gc03328e#!divAbstract)
- [A Survey of the Borrowing Hydrogen Approach to the Synthesis of some Pharmaceutically Relevant Intermediates](https://pubs.acs.org/doi/10.1021/acs.oprd.5b00199)
- [Multi-task Bayesian Optimization of Chemical Reactions](https://chemrxiv.org/articles/preprint/Multi-task_Bayesian_Optimization_of_Chemical_Reactions/13250216)
- Worth reading through the [peer review](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03213-y/MediaObjects/41586_2021_3213_MOESM2_ESM.pdf) of the shields Nature paper.

## In-Silico Comparison

Questions:
- [ ] What is the effect of the number of cotraining tasks?
- [ ] Can MTBO accelerate reaction optimization with unseen reagents (e.g., catalyst, bases)?
- [ ] Does MTBO work when only some of the variables are changed in the co-training dataset? (e.g., high throughput datasets)
- [ ] What is the effect of the difference in substrates?
- [ ] How many more experiments does each extra objective add?

Algorithms to compare
- MTBO
- STBO and SOBO 
- For multiobjective, TSEMO and EHVI/qEHVI
- Full factorial DoE

Quantification
- Percent of total combinations required to get quantitative yield?
### Data Sources for Benchmarks

* Suzuki-Miyura Reaction
    - [x] [Reizman](https://gosummit.readthedocs.io/en/latest/experiments_benchmarks/implemented_benchmarks.html#summit.benchmarks.ReizmanSuzukiEmulator): 4 cases with varying catalysts, temperature, residence time, catalyst loading and optimizing yield and TON
    - [x] [Baumgartner Suzuki](https://pubs.rsc.org/en/content/articlelanding/2018/RE/C8RE00032H#!divAbstract): 3 cases varying catalyst complex, catalyst loading, temperature, and residence time and optimizing yield and TON
    - [ ] [Perera](http://www.sciencemag.org/lookup/doi/10.1126/science.aap9112): HT screening 9 cases (3 electrophiles, 3 nucelophiles) varying ligand, base, solvent
    - [ ] [Christensen](https://pubs.rsc.org/en/content/articlelanding/2019/re/c9re00086k#!divAbstract): Kinetic profiling, only looking at different species over time, probably not the best for our application
    - [ ] [Christensen](https://chemrxiv.org/articles/preprint/Data-science_driven_autonomous_process_optimization/13146404): 1 case varying ligand, temperature, catalyst loading and ligand ratio to optimize formation of one product over two others

* C-N Cross Coupling Reaction
    - [] [Baumgartner C-N](https://pubs.acs.org/doi/10.1021/acs.oprd.9b00236): 4 cases with varying catalysts, bases, temperature, residence time, base equivalents and optimizing yield and TON
    - [x] [Ahneman](https://science.sciencemag.org/content/360/6385/186): HT screening 15 cases with varying additives, catalysts, bases
    - [x] [Buitrago-Santinilla](https://science.sciencemag.org/content/347/6217/49): HT Screening 5 cases varying  catalyst, catalyst loading, base, and base loading.
    - [] [Bédard](https://science.sciencemag.org/content/361/6408/1220.full) - 1 case varying temperature, mol% of catlayst, and flowrates of different pumps to optimize conversion to product (see page 71 of the [SI](https://science.sciencemag.org/content/sci/suppl/2018/09/19/361.6408.1220.DC1/aat0650_Bedard_SM.pdf))
    
* Asymmetric Hydrogenation
    - [Amar](https://pubs.rsc.org/en/content/articlehtml/2019/sc/c9sc01844a)
    
* Nucleophilic Aromatic Substitution
    - [Hone](https://gosummit.readthedocs.io/en/latest/experiments_benchmarks/implemented_benchmarks.html#snar-benchmark) - Mechanistic benchmark based on kinetic model
    - [Jorner](https://pubs.rsc.org/en/content/articlelanding/2021/SC/d0sc04896h#!divAbstract) - Large database of kinetic constants for SnAr reactions
    - [Schweidtmann](https://www.sciencedirect.com/science/article/pii/S1385894718312634) - 1 case varying residence time, equivalents, concetration of aromatic SM, and temperature to optimize E-factor and STY
    - [Zhou](https://www.sciencedirect.com/science/article/pii/S0009250913007550?via%3Dihub#bib6) Reaction constatns in various solvents
    - [Bédard](https://science.sciencemag.org/content/361/6408/1220.full) - 1 case varying temperature, and flowrates of different pumps to optimize conversion to product (see page 75 of the [SI](https://science.sciencemag.org/content/sci/suppl/2018/09/19/361.6408.1220.DC1/aat0650_Bedard_SM.pdf))
  
* Aldol Condensation
    -[Jeraal](https://chemistry-europe.onlinelibrary.wiley.com/doi/full/10.1002/cmtd.202000044)
  
* Reductive Amination
    - [Bédard](https://science.sciencemag.org/content/361/6408/1220.full) - 1 case varying temperatures of each step, and flowrates of different pumps to optimize conversion to product (see page 73 of the [SI](https://science.sciencemag.org/content/sci/suppl/2018/09/19/361.6408.1220.DC1/aat0650_Bedard_SM.pdf))
    - [Bray](https://www.sciencedirect.com/science/article/pii/0040403995009459) - Paper from the 90s with multiple conditions but on solid phase
    - [Krsieor](https://www.sciencedirect.com/science/article/pii/S0009250920307193) - Kinetic model
    - [Song](https://www.sciencedirect.com/science/article/pii/S2468823118302062) - Kinetic model in multple solvents
    - [Finnigan](https://pubs.acs.org/doi/full/10.1021/acs.oprd.0c00075): Enzyme catalysis
  
* Grignard Reaction
    - [Pedersen](https://pubs.acs.org/doi/full/10.1021/acs.iecr.8b00564): 1 case with a kinetic model
    - [Changi](https://pubs.acs.org/doi/full/10.1021/acs.oprd.5b00281): 1 case with a kinetic model

## Setup

Clone the project:

    git clone https://github.com/sustainable-processes/multitask.git

Install [poetry](https://python-poetry.org/docs/) and then run the following command to install dependencies:

    poetry install

Additionally, until this [issue](https://github.com/open-reaction-database/ord-schema/issues/600) is fixed, you will need to manually install the ord-schema dependencies:
    
    poetry run pip install -r ord-requirements.txt

You can run commands from inside the poetry virtual environment in one of two ways:

  - **Option 1**: Put `poetry run` in front of every command (e.g., `poetry run dvc repro`) as we did above to install extra requirements.
  - **Option 2**: Activate the virtual environement and run everything as normal:
      - Run `poetry show -v`. Copy the path specified after "Using virtualenv"
      - Activate the virtual environemnt at that path. So, if the path is `/Users/Kobi/Library/Caches/pypoetry/virtualenvs/multitask-z7ErTcQa-py3.7`, you would run `source /Users/Kobi/Library/Caches/pypoetry/virtualenvs/multitask-z7ErTcQa-py3.7/bin/activate`
  
[DVC](https://dvc.org/doc) is used for tracking data and keeping it in sync with git. When you first clone the project, you will need to download the existing data:

    dvc pull

This will ask you to authenticate with Google Drive. Make sure to use your Cambridge account (or an account you shared the [Multitask Bayesian Optimization shared drive](https://drive.google.com/drive/u/2/folders/0AGWGXkw78NfUUk9PVA) with).

### Pipelines

[Pipelines](https://dvc.org/doc/start/data-pipelines#get-started-data-pipelines) are a feature of DVC for keeping track of data workflows.  You can see our existing pipelines using the following command:

    dvc dag

You can rerun the pipeline using `dvc repro`. By default, DVC will only re-run the steps of the pipeline where code or data dependencies have changed.   You can override this with `dvc repro no-run-cache`.

Follow the steps [here](https://dvc.org/doc/start/data-pipelines#get-started-data-pipelines) to create a new pipeline on the command line or read through the [documentation](https://dvc.org/doc/user-guide/project-structure/pipelines-files) on `dvc.yaml` to see how to edit the pipeline configuration file directly. 

## Coding Guidelines

Please format your code using [black](https://github.com/psf/black). Jupyter notebooks, which are for exploration, don't need to be formatted.

**Directory Structure**
```
├── data/  # Store all data and results here (tracked mainly by DVC)
├── dvc.lock # Lock file for DVC
├── dvc.yaml # DVC pipelines configuration
├── figures/  # Store figures in this directory and track using dvc
├── multitask # python package with key functionality
├── nbs # Exploratory Jupyter notebooks
├── ord-requirements.txt # Dependencies for ORD schema
├── params.yaml # Parameters for DVC pipelines
├── poetry.lock  # Poetry lock file
└── pyproject.toml # Poetry cnfiguration file
```
* Put final code for pipelines in `multitask`, and add the code to pipelines as described in the previous section.
* [Typer](https://typer.tiangolo.com/tutorial/first-steps/) is great for turning a python function into a command line script. It's already installed, so you might as well use it.



