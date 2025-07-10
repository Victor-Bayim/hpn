# 基于NVSHMEM, 实现多机 Global Allreduce
## 如何参与ISSUE
1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」  
2、Fork 到个人的仓库下  
3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue7/ 目录
4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成  
5、如有任何疑问可以在评论区留言或者邮件至联络人  

## 环境准备

1. 你可以在 在NV官网下载 NVSHMEM 的代码: https://developer.nvidia.com/nvshmem-archive
2. 你可以先学习一下 NVSHMEM 中IBGDA和IBRC的基础概念: https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/
3. 你可以基于 DeepEP的代码框架, 增加一个 Allreduce 的接口, DeepEP 代码地址: https://github.com/deepseek-ai/DeepEP
4. 你可以基于 IBGDA 实现多QP连接; 当前 IBGDA 天然支持多 QP 实现; 你也可以基于 IBRC 实现多QP连接, 可以参考: https://github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP


## 验收要求

1. 基于 NVSHMEM 的 Allreduce Kernel 代码, 包括中间层调用代码
2. 参考 DeepEP 的 test_internode.py 的 Allreduce 测试代码
