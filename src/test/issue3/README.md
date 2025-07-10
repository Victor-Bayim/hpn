# 使用 GPU 的 Copy Engine 实现 DeepEP Internode 中的 NVLink 域数据转发
## 如何参与ISSUE
1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」  
2、Fork 到个人的仓库下  
3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue4/ 目录  
4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成  
5、如有任何疑问可以在评论区留言或者邮件至联络人  

## Issue 目标

1. 降低 DeepEP Internode 场景 SM 使用，降低推理成本

## 环境准备

1. 你可以在 GitHub 下载 DeepEP 开源代码：https://github.com/deepseek-ai/DeepEP
2. 你可以参考 `README.md` 中的安装说明和 `tests/` 目录下的各个测试脚本， 学习 DeepEP 的部署和使用方式
3. 在两台机器上部署 DeepEP，并运行 internode 测试，达到性能基线
4. 修改 `csrc/kernels/internode.cu` 等相关文件，将 NVLink 域转发方式替换为使用 Copy Engine（CE）
5. 分析 SM 数和 Warp 数对转发性能的影响，并通过减少 channel、重新安排 warp 职责等方式尽可能压缩 GPU 计算资源占用

注：当前 DeepEP 最新版本已经在 Internode kernel 中使用 TMA 来进行 NVLink 转发，你可以选择一个仍然使用 LD/ST 指令进行转发的的稍旧版本来作为性能基线

## 验收要求

1. 正确运行 DeepEP `test_internode.py`
2. 在上述前提下，与原版 DeepEP 相比，通信完成时间不增加，且 NVLink 转发所使用的 SM 或 Warp 数目减少 50% 以上
