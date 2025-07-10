# NIXL不同后端notif性能差异分析及优化

## 如何参与ISSUE

1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」  
2、Fork 到个人的仓库下  
3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue7/ 目录
4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成  
5、如有任何疑问可以在评论区留言或者邮件至联络人  

## 环境准备

1. 从 github 上下载 NIXL 的代码: <https://github.com/ai-dynamo/nixl.git>
2. 了解 nixlbenchmark 测试环境的搭建方案，可以参考源码中相关readme文件及源码文件 `benchmark/nixlbench/contrib/build.sh`
3. 在两台GPU服务器上搭建上述benchmark测试环境，并跑通ucx和mooncake两个backend的基本测试用例
4. 分析两个backend的测试结果，并分析性能差异的原因
5. 从 github 上下载 Mooncake 项目源码：<https://github.com/kvcache-ai/Mooncake.git>
6. 优化mooncake backend 中 notif 功能，将目前的tcp方案替换为rdma方案

## 验收要求

1. nixl 通信组件 ucx 和 mooncake 两个 backend 的性能测试结果报告以及原因分析报告
2. mooncake backend notif 接口的rdma实现方案源码
