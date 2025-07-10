# MOE+Dispatch实现更细粒度算子融合

## 如何参与ISSUE
1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」  
2、Fork 到个人的仓库下  
3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue6/ 目录  
4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成  
5、如有任何疑问可以在评论区留言或者邮件至联络人  

## Issue 目标

1. 在ByteDance Flux库的moe_ag_scatter算子基础上实现更细粒度的算子融合
2. 在参数配置 `<Group,M,N,K> = <4,256, 4096, 7168>` 下达到80%的通信Overlap水平
3. 优化MOE（Mixture of Experts）模型中的计算-通信重叠性能

## 环境准备

1. 你可以在 GitHub 下载 ByteDance Flux 开源代码：https://github.com/bytedance/flux
2. 你可以参考项目 `README.md` 中的安装说明，按照以下步骤安装 Flux：
   ```bash
   git clone --recursive https://github.com/bytedance/flux.git
   ```
3. 你需要熟悉 Flux 的 MoE 相关算子，特别是 `moe_ag_scatter` 算子的实现
4. 你可以参考 `test/python/moe_ag_scatter/test_moe_ag.py` 了解基础用法
5. 你需要配置测试环境以支持多GPU通信测试

## 技术背景

Flux 是MOE模型中用于高性能计算-通信重叠的通信库，支持稠密/MoE模型在GPU上的各种并行方式。其中 `moe_ag_scatter` 算子用于实现 MoE 模型中的 all-gather 与 grouped GEMM 的融合操作。

当前需要在以下参数配置下优化性能：
- Group: 4 (专家组数)
- M: 256 (矩阵维度)
- N: 4096 (矩阵维度)  
- K: 7168 (矩阵维度)

## 验收要求

1. **性能基线测试**：使用原始的 `moe_ag_scatter` 算子在给定参数下进行性能测试，建立基线
2. **算子融合优化**：实现更细粒度的MOE+Dispatch算子融合，提升计算-通信重叠效率
3. **通信Overlap目标**：在参数 `<Group,M,N,K> = <4,256, 4096, 7168>` 下实现至少80%的通信Overlap水平
4. **性能验证**：提供完整的性能测试脚本和结果对比

## 评估指标

- **通信Overlap比例**：目标达到80%以上
- **端到端延迟**：相比原始实现应有显著改善
- **内存使用**：不应显著增加GPU内存占用
- **吞吐量**：在保证Overlap的前提下提升整体吞吐量

## 参考资料

- [Flux 项目文档](https://github.com/bytedance/flux)
- [FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion](https://arxiv.org/abs/2406.06858)
- [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/abs/2502.19811) 