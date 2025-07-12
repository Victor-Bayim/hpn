#!/bin/bash
# 用法: ./run_srun.sh [进程数] [其它参数...]
NP=${1:-4}
shift 1
srun -n $NP ./test_allreduce "$@"
