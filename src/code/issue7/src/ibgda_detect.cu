#include "allreduce.h"

// 检查环境变量 NVSHMEM_IB_ENABLE_IBGDA 是否启用
static bool check_ibgda_env_flag() {
    const char *env = getenv("NVSHMEM_IB_ENABLE_IBGDA");
    if (env && atoi(env) == 1) {
        return true;
    }
    return false;
}

// 检测 NVSHMEM IBGDA 支持 (检查版本和环境变量)
// 返回 true 表示 IBGDA 已启用
bool check_ibgda_env() {
    int major = 0, minor = 0, patch = 0;
    nvshmemx_vendor_get_version_info(&major, &minor, &patch);
    // NVSHMEM 2.6.0+ 支持 IBGDA
    bool supported = (major > 2 || (major == 2 && minor >= 6));
    bool enabled = false;
    if (supported && check_ibgda_env_flag()) {
        enabled = true;
    }
    if (nvshmem_my_pe() == 0) {
        if (supported) {
            printf("NVSHMEM v%d.%d.%d supports IBGDA: %s\n",
                   major, minor, patch,
                   enabled ? "ENABLED" : "disabled (env not set)");
        } else {
            printf("NVSHMEM v%d.%d.%d does not support IBGDA (requires >= 2.6.0)\n",
                   major, minor, patch);
        }
        fflush(stdout);
    }
    return (supported && enabled);
}
