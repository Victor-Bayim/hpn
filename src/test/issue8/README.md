# 研究基于IBGDA和IBRC的Alltoall在各个msg的理论性能上限
## 如何参与ISSUE
1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」  
2、Fork 到个人的仓库下  
3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue7/ 目录
4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成  
5、如有任何疑问可以在评论区留言或者邮件至联络人  

## 环境准备

1. 你可以在 在NV官网下载 NVSHMEM 的代码: https://developer.nvidia.com/nvshmem-archive
2. 你可以先学习一下 NVSHMEM 中IBGDA和IBRC的基础概念: https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/


## 验收要求

1. 输出alltoall各个msg下，ibrc 和 ibgda 的理论耗时
2. 输出alltoall 的 cuda kernel中，各个独立子模块的理论耗时计算公式
