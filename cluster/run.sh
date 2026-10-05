#!/bin/bash
# HTCondor job script: runs one experiment script on a cluster node.
# Submit from the repository root (see cluster/jobs.sub); REPO_DIR is passed by the submit file.
# PYTHON may name an interpreter that has the packages of requirements.txt (default: python3).
set -u
cd "${REPO_DIR:?REPO_DIR must name the repository root}" || exit 1
PY="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
script="$1"; shift
echo "script=$script version=$("$PY" --version 2>&1) start=$(date -Is)"
"$PY" -u "experiments/$script" "$@"
rc=$?
echo "rc=$rc end=$(date -Is)"
exit $rc
