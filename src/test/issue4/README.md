# 探究层内不同overlap方法的最佳使用范式
## 如何参与ISSUE
1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」  
2、Fork 到个人的仓库下  
3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue4/ 目录  
4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成  
5、如有任何疑问可以在评论区留言或者邮件至联络人  

## 环境准备

1. 你可以在 github 下载代码
    - 基于融合的方式: https://github.com/bytedance/flux、https://github.com/ByteDance-Seed/Triton-distributed
    - 基于分解的方式: https://github.com/NVIDIA/TransformerEngine
    - 基于信号的方式: https://github.com/infinigence/FlashOverlap
2. 你可以在开源代码主页了解这些overlap方法的安装方法，并在example中学习如何使用和测试
3. 你可以选择一种计算/通信模式，编写或者复用已有测试，验证各种overlap方式的实际运行效果
4. 你可以基于从粒度控制、硬件资源和软件流程等维度分析性能影响要素，明确其优劣势与适用场景


## 验收要求

1. 各种Overlap方式在统一可对比测试环境中的测试脚本/代码
2. 各种Overlap方式优劣势与适用场景的分析报告
