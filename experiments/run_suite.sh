#!/bin/zsh
# Full suite: results/<name>/<group>/<dataset>/, then SUMMARY.md and summary_all.png.
# Usage: experiments/run_suite.sh [out_dir]   (JOBS=2 by default)
# One dataset takes about 8 minutes alone. Each process needs about 1 GB, and running
# more than the machine can hold swaps so badly that runs take 40 times longer.
cd "$(dirname "$0")/.."
export R=${1:-results/11_06-tangent_safeguard}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=experiments:src
# A dataset is retried once: JAX 0.11's compiler has segfaulted here at random (seen once
# in nine datasets), which leaves no Python traceback. summarize_suite.py flags any that stay missing.
# wind has only 764 points. Its window must keep H*alpha_min below beta, or no ordinary
# allocation can meet the target budget: window 150 gives H*alpha_min = 0.08.
xargs -P "${JOBS:-2}" -L 1 sh -c 'g=$0 d=$1; shift; mkdir -p "$R/$g/$d"
  run() { .venv_local/bin/python -u experiments/run_1d.py --datasets "$d" --out "$R/$g/$d" "$@" > "$R/$g/$d/log.txt" 2>&1; }
  run "$@" || run "$@"
  echo "finished $g/$d (exit $?)"' <<'EOF'
energy wind --n_train 150 --window 150 --t_max 440
stocks amd
stocks nvidia
stocks amazon
stocks blackberry
stocks fanniemae
robotics drone_x
synthetic regime_shift
energy solar
EOF
.venv_local/bin/python experiments/summarize_suite.py "$R"
