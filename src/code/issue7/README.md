# Allreduce 项目使用说明

## 构建与安装

本项目依赖 NVIDIA 的 NVSHMEM 库和 CUDA 工具链。请确保已安装 CUDA (含 nvcc 编译器) 和 NVSHMEM 并正确设置环境变量。可以通过设置环境变量 `NVSHMEM_HOME` 来指向 NVSHMEM 的安装路径。

构建步骤：
1. 编辑 `Makefile`，根据实际安装位置修改 `NVSHMEM_HOME` 和 `CUDA_HOME`（如有必要）。
2. 在项目根目录下运行 `make`，将使用 NVCC 编译生成可执行文件 `test_allreduce`。

Makefile 默认会包含 NVSHMEM 的头文件和库（例如 `-I$(NVSHMEM_HOME)/include` 和 `-L$(NVSHMEM_HOME)/lib -lnvshmem`），并启用 CUDA C++11 标准和优化选项。

## 运行方法

编译成功后，可以使用以下两种方式启动测试程序：
- 使用 MPI 启动：运行脚本 `scripts/run_mpirun.sh`。例如：
  ```bash
  ./scripts/run_mpirun.sh 4            # 启动 4 个进程进行 Allreduce 测试
  ./scripts/run_mpirun.sh 4 --perf -n 1000000   # 4 个进程，运行性能测试，数据规模 100万
