pip install .
ARGS=("$@")
python multitask/kinetic_optimization.py stbo "${ARGS[@]}"
