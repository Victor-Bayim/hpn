#include "allreduce.h"
#include "allreduce_utils.cu"  // 引入辅助函数和宏
#include "ibgda_detect.cu"     // 引入 IBGDA 检测函数

// CUDA 内核：将源数组 `src[offset:offset+len]` 加到目标数组 `dest[offset:offset+len]`
__global__ void add_kernel(int *dest, const int *src, size_t offset, size_t len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        dest[offset + idx] += src[offset + idx];
    }
}

int allreduce_init(int *argc, char ***argv, AllreduceConfig *config) {
    // 初始化 NVSHMEM 库（使用默认启动方式，比如通过 MPI 或 PMI 启动）
    nvshmem_init();
    // 获取全局 PE 数量和本地 PE ID
    config->npes = nvshmem_n_pes();
    config->mype = nvshmem_my_pe();
    config->algo = ALLREDUCE_RING;
    // 为当前进程选择对应的 GPU 设备（按照节点内 PE ID 进行映射）
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(mype_node));
    // 检测 IBGDA 支持情况并设置配置
    config->use_ibgda = false;
    if (check_ibgda_env()) {
        // NVSHMEM 版本支持且环境变量启用了 IBGDA
        config->use_ibgda = true;
    }
    // 输出配置信息（仅主进程打印）
    if (config->mype == 0) {
        printf("Allreduce init: npes=%d, use_ibgda=%s, algorithm=Ring\n",
               config->npes, config->use_ibgda ? "true" : "false");
        fflush(stdout);
    }
    return 0;
}

int allreduce_finalize() {
    nvshmem_finalize();
    return 0;
}

int allreduce_sum_int(int *data, size_t count, AllreduceConfig *config) {
    int npes = config->npes;
    int mype = config->mype;
    // 若只有一个进程，直接返回（数据已是全局和）
    if (npes == 1) return 0;

    // 计算每个分块的大小（假设 count 能被 npes 整除，以简化实现）
    size_t seg_size = count / npes;
    size_t remainder = count % npes;
    if (remainder != 0) {
        // 为简化处理，不支持不能整除的情况
        if (mype == 0) {
            fprintf(stderr, "Warning: data count %zu not divisible by npes %d, remainder will be ignored.\n",
                    count, npes);
        }
    }

    // 获取相邻 PE 编号（左邻居prev，右邻居next）
    int prev_pe = (mype - 1 + npes) % npes;
    int next_pe = (mype + 1) % npes;

    // 分配对等通信的对称内存缓冲（用于接收数据块）
    // 注意：data 指向的内存需通过 nvshmem_malloc 分配，此处假定 data 已是对称内存
    static __shared__ int dummy; // 占位符，无需额外分配

    // 进入 Scatter-Reduce 阶段：逐步累加来自其他进程的数据块
    for (int step = 0; step < npes - 1; ++step) {
        // 计算本步要发送和接收的数据块索引
        int send_index = (mype - step + npes) % npes;
        int recv_index = (mype - step - 1 + npes) % npes;
        size_t offset_send = send_index * seg_size;
        size_t offset_recv = recv_index * seg_size;
        size_t elems = seg_size;
        if (send_index == npes - 1 && remainder != 0) {
            // 如果有剩余元素，将最后一个块的元素数调整为 seg_size + remainder
            elems = seg_size + remainder;
        }
        if (recv_index == npes - 1 && remainder != 0) {
            // 接收块为最后一块时，同样调整元素数量
            elems = seg_size + remainder;
        }
        // 将本进程对应发送块的数据发送到下一个进程的 recv 缓冲区相同偏移处
        nvshmem_int_put_nbi(&data[offset_send], &data[offset_send], elems, next_pe);
        // 等待所有进程完成发送（同步点），确保数据已送达
        nvshmem_barrier_all();
        // 将收到的数据块与本地对应块累加
        if (elems > 0) {
            int threads = 256;
            int blocks = (int)((elems + threads - 1) / threads);
            add_kernel<<<blocks, threads>>>(data, data, offset_recv, elems);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // 此时每个进程 data 数组中仅有一个块（索引 (mype + 1) mod npes）的数据为全局和结果
    // 进入 Allgather 阶段：将各进程持有的结果块广播给所有进程
    // 计算初始持有的结果块索引
    int cur_index = (mype + 1) % npes;
    for (int step = 0; step < npes - 1; ++step) {
        // 当前步要发送和接收的块索引
        int send_index = (mype + 1 - step + npes) % npes;
        int recv_index = (mype - step + npes) % npes;
        size_t offset_send = send_index * seg_size;
        size_t offset_recv = recv_index * seg_size;
        size_t elems = seg_size;
        if (send_index == npes - 1 && remainder != 0) {
            elems = seg_size + remainder;
        }
        if (recv_index == npes - 1 && remainder != 0) {
            elems = seg_size + remainder;
        }
        // 将当前持有的结果块发送给下一个进程
        nvshmem_int_put_nbi(&data[offset_send], &data[offset_send], elems, next_pe);
        nvshmem_barrier_all();
        // 将接收到的块直接放入本地 data 数组相应位置（覆盖原部分）
        // 由于 nvshmem_put 直接写入了 data 指定位置，因此此处无需额外累加，只要同步即可
        // （假设 NVSHMEM 实现确保 put 后 barrier 返回时数据已写入本地内存）
    }

    return 0;
}
