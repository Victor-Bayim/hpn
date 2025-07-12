NVSHMEM_HOME ?= /usr/local/nvshmem
CUDA_HOME    ?= /usr/local/cuda

NVCC     := $(CUDA_HOME)/bin/nvcc
CXXFLAGS := -O2 -std=c++11 -I$(NVSHMEM_HOME)/include -Iinclude
LDFLAGS  := -L$(NVSHMEM_HOME)/lib -lnvshmem -lcuda

TARGET    := test_allreduce
SRC_FILES := src/allreduce.cu src/allreduce_utils.cu src/ibgda_detect.cu tests/test_allreduce.cpp

all: $(TARGET)

$(TARGET): $(SRC_FILES) include/allreduce.h include/allreduce_utils.h
	$(NVCC) $(CXXFLAGS) $(SRC_FILES) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
