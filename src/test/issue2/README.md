# GPU内通信方式（LSU、TMA、CE）的最佳通信区间

## 如何参与ISSUE
1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」  
2、Fork 到个人的仓库下  
3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue7/ 目录
4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成  
5、如有任何疑问可以在评论区留言或者邮件至联络人  

## 环境准备

1. 你可以下载 NCCL 源代码: NCCL: https://github.com/NVIDIA/nccl
2. 你可以下载 nccltest 源代码: https://github.com/NVIDIA/nccl-tests

## 验收要求

1. 给出LSU、TMA、CE等机内通信方式的使用范例，要求可以充分利用硬件性能
2. 给出各个机内通信方式在各个 message 上的通信性能评测结果
