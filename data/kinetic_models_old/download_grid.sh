#!/bin/bash
ARGS=("$@")
grid artifacts "${ARGS[@]}"
rm -r data/kinetic_models/grid_artifacts/*/**/dist
rm -r data/kinetic_models/grid_artifacts/*/**/multitask.egg-info
