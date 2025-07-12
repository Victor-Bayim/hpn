#ifndef ALLREDUCE_H
#define ALLREDUCE_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// Allreduce 算法类型（目前仅实现 Ring 算法）
enum AllreduceAlgo {
    ALLREDUCE_RING = 0
};

// Allreduce 配置结构体
struct AllreduceConfig {
    int npes;        // 全局参与进程数 (PE 数量)
    int mype;        // 本进程的 PE 编号
    bool use_ibgda;  // 是否启用 IBGDA (由检测模块设置)
    AllreduceAlgo algo; // 所使用的 Allreduce 算法
};

// 初始化 NVSHMEM 并进行 Allreduce 全局配置
int allreduce_init(int *argc, char ***argv, AllreduceConfig *config);

// 释放 NVSHMEM 资源
int allreduce_finalize();

// 执行整数加法 Allreduce（环形算法实现）
int allreduce_sum_int(int *data, size_t count, AllreduceConfig *config);

#endif // ALLREDUCE_H
