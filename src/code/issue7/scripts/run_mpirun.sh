#!/bin/bash
# 用法: ./run_mpirun.sh [进程数] [其它参数...]
NP=${1:-4}
shift 1
mpirun -np $NP ./test_allreduce "$@"
