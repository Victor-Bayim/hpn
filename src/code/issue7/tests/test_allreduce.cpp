#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include "allreduce.h"

int main(int argc, char *argv[]) {
    AllreduceConfig config;
    bool perf_mode = false;
    size_t count = 1024;  // 默认元素数量
    // 简单解析命令行参数
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--perf") == 0 || strcmp(argv[i], "-p") == 0) {
            perf_mode = true;
        } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--count") == 0) && i + 1 < argc) {
            count = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            if (config.mype == 0) {
                std::cout << "Usage: " << argv[0] << " [-n <count>] [--perf]\n";
            }
            return 0;
        }
    }

    // 初始化 Allreduce 环境
    allreduce_init(&argc, &argv, &config);

    // 分配对称内存用于 Allreduce 数据（每个进程都分配相同大小）
    int *data = (int *) nvshmem_malloc(count * sizeof(int));
    if (!data) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed for %zu ints\n", config.mype, count);
        nvshmem_global_exit(1);
    }

    if (!perf_mode) {
        // 正确性验证模式：初始化数据，执行 Allreduce，检查结果正确性
        // 将每个元素初始化为本进程 ID
        CUDA_CHECK(cudaDeviceSynchronize());
        for (size_t i = 0; i < count; ++i) {
            data[i] = config.mype;
        }
        // 执行 Allreduce (整数求和)
        allreduce_sum_int(data, count, &config);
        nvshmem_barrier_all();  // 确保所有进程都完成
        // 验证结果：每个元素应等于所有 PE 值之和
        long long expected = 0;
        for (int pe = 0; pe < config.npes; ++pe) {
            expected += pe;
        }
        // 拷贝部分数据回主机检查
        std::vector<int> host_data(count);
        CUDA_CHECK(cudaMemcpy(host_data.data(), data, sizeof(int) * count, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (size_t i = 0; i < count; ++i) {
            if (host_data[i] != expected) {
                fprintf(stderr, "Error: data[%zu]=%d, expected=%lld\n",
                        i, host_data[i], expected);
                ok = false;
                break;
            }
        }
        if (config.mype == 0) {
            if (ok) {
                std::cout << "Allreduce correctness PASSED (expected sum = " << expected << ")" << std::endl;
            } else {
                std::cout << "Allreduce correctness FAILED" << std::endl;
            }
        }
    } else {
        // 性能基准模式：多次执行 Allreduce 并统计平均时间
        size_t warmup = 10;
        size_t iterations = 100;
        // 初始化随机数据
        CUDA_CHECK(cudaDeviceSynchronize());
        for (size_t i = 0; i < count; ++i) {
            data[i] = rand() % 100;
        }
        nvshmem_barrier_all();
        // 预热若干次
        for (size_t i = 0; i < warmup; ++i) {
            allreduce_sum_int(data, count, &config);
        }
        nvshmem_barrier_all();
        // 正式计时
        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            allreduce_sum_int(data, count, &config);
        }
        nvshmem_barrier_all();
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_sec = std::chrono::duration<double>(end_time - start_time).count();
        double avg_time = total_sec / iterations;
        // 计算每次 Allreduce 平均吞吐量 (GB/s)，每个进程通信量约为2*count*sizeof(int)
        double bytes = (double)count * sizeof(int);
        double bandwidth = (2 * bytes) / avg_time;  // 每秒处理的字节数
        if (config.mype == 0) {
            std::cout << "Allreduce perf: data count=" << count 
                      << ", avg time=" << (avg_time * 1000) << " ms"
                      << ", throughput=" << (bandwidth / 1e9) << " GB/s per PE" << std::endl;
        }
    }

    // 释放资源
    nvshmem_free(data);
    allreduce_finalize();
    return 0;
}
