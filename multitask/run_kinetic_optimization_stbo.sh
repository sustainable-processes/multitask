pip install --use-feature=in-tree-build .
ARGS=("$@")
python multitask/kinetic_optimization.py stbo "${ARGS[@]}"
