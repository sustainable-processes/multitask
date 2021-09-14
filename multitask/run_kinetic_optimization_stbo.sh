python setup.py install
ARGS=("$@")
python multitask/kinetic_optimization.py stbo "${ARGS[@]}"
