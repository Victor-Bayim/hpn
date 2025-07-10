# 提高 DeepEP Low-Latency Kernel 在专家负载不均时的通信性能
## 如何参与ISSUE
1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」  
2、Fork 到个人的仓库下  
3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue4/ 目录  
4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成  
5、如有任何疑问可以在评论区留言或者邮件至联络人  

## Issue 目标

1. 最小化集合通信完成时间

## 环境准备

1. 你可以在 GitHub 下载 DeepEP 开源代码：https://github.com/deepseek-ai/DeepEP
2. 你可以参考 `README.md` 中的安装说明和 `tests/` 目录下的各个测试脚本， 学习 DeepEP 的部署和使用方式
3. 你需要在两台机器上部署 DeepEP，并运行 low-latency 测试，达到性能基线
4. 你需要修改 `test_low_latency.py` 中生成 TopK 专家（`topk_idx`）的代码，模拟专家负载不均的情况，满足 `最大负载 / 平均负载 = 2`
5. 你需要对 low-latency kernel 的底层实现进行优化，提升 DeepEP 在这种专家负载不均情况下的通信性能

## 验收要求

1. 对 `test_low_latency.py` 进行修改，模拟 `最大负载 / 平均负载 = 2` 的负载不均情形
2. 修改后的 `test_low_latency.py` 能正常执行完毕，且在同样的负载不均情况下，修改后的 DeepEP low-latency kernel 与原版相比，集合通信完成时间有所降低，且 SM 使用不高于原版
