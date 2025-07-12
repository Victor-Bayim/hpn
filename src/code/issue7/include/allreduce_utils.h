#ifndef ALLREDUCE_UTILS_H
#define ALLREDUCE_UTILS_H

#include "allreduce.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// CUDA API 错误检查
#define CUDA_CHECK(stmt) \
    do { \
        cudaError_t result = (stmt); \
        if (result != cudaSuccess) { \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(result)); \
            exit(-1); \
        } \
    } while (0)

// NVSHMEM 调用错误检查
#define NVSHMEM_CHECK(stmt) \
    do { \
        int rc = (stmt); \
        if (rc != 0) { \
            fprintf(stderr, "[NVSHMEM ERROR] %s:%d: return code %d\n", __FILE__, __LINE__, rc); \
            exit(-1); \
        } \
    } while (0)

void print_allreduce_config(const AllreduceConfig *config);

#endif // ALLREDUCE_UTILS_H
