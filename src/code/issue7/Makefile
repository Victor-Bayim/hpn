
## Makefile

Makefile 定义了项目的构建规则，包括 NVCC 编译选项和 NVSHMEM 链接设置。

```makefile
# 默认 NVSHMEM 和 CUDA 安装路径，可根据需要修改
NVSHMEM_HOME ?= /usr/local/nvshmem
CUDA_HOME    ?= /usr/local/cuda

NVCC        := $(CUDA_HOME)/bin/nvcc
CXXFLAGS    := -O2 -std=c++11 -I$(NVSHMEM_HOME)/include
LDFLAGS     := -L$(NVSHMEM_HOME)/lib -lnvshmem -lcuda

TARGET      := test_allreduce
SRC_FILES   := src/allreduce.cu src/allreduce_utils.cu src/ibgda_detect.cu tests/test_allreduce.cpp

all: $(TARGET)

$(TARGET): $(SRC_FILES) include/allreduce.h
	$(NVCC) $(CXXFLAGS) $(SRC_FILES) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
