# Source this file in the Pod shell. No host settings or permissions are changed.
export TMG_NATIVE_ROOT="${TMG_NATIVE_ROOT:-$HOME/.local/share/tmg-native}"
export TMG_DATA_ROOT="${TMG_DATA_ROOT:-$HOME/workspace/working/tmg_data}"
export PATH="$TMG_NATIVE_ROOT/bin:$TMG_NATIVE_ROOT/tools/usr/bin:$PATH"
export CLUSTER=pod
export VAL_PARALLEL="${VAL_PARALLEL:-1}"
export TEST_SHARDS="${TEST_SHARDS:-1}"
export SEED_PARALLEL="${SEED_PARALLEL:-2}"
export SEQUENTIAL_TASKS=1
export TRAIN_OVERRIDES="${TRAIN_OVERRIDES:-training.cache=hdf5 training.keep_training_states=last training.num_threads=4}"
