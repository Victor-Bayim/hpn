#include "allreduce.h"

// 检查 CUDA API 调用结果的宏。如果有错误则输出并退出
#define CUDA_CHECK(stmt)                                                        \
    do {                                                                        \
        cudaError_t result = (stmt);                                            \
        if (cudaSuccess != result) {                                            \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(result));                                \
            exit(-1);                                                           \
        }                                                                       \
    } while (0)

// 检查 NVSHMEM 调用的宏（假设返回 0 为成功）
#define NVSHMEM_CHECK(stmt)                                                    \
    do {                                                                       \
        int rc = (stmt);                                                       \
        if (rc != 0) {                                                         \
            fprintf(stderr, "[NVSHMEM ERROR] %s:%d: return code %d\n",         \
                    __FILE__, __LINE__, rc);                                   \
            exit(-1);                                                          \
        }                                                                      \
    } while (0)

// （可选）打印 Allreduce 配置的函数
void print_allreduce_config(const AllreduceConfig *config) {
    printf("PE %d/%d: IBGDA %s, Algorithm = %s\n",
           config->mype, config->npes,
           config->use_ibgda ? "Enabled" : "Disabled",
           (config->algo == ALLREDUCE_RING ? "Ring" : "Unknown"));
}
