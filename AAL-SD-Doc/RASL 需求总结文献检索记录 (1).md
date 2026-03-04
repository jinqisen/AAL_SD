# RASL 需求总结文献检索记录

## 1. 传统 AL 在语义分割中的局限性 & 动态策略需求

**Mittal et al. (2023) "Best Practices in Active Learning for Semantic Segmentation" (被引 37)**
- 关键发现：数据分布冗余度、是否结合半监督学习、标注预算大小，这三个因素对"最佳 AL 策略的选择"是决定性的
- 意味着：不存在一个"万能"的固定 AL 策略，策略选择必须根据具体条件动态调整

**Doucet et al. (2024) "Bridging Diversity and Uncertainty in AL with Self-Supervised Pre-Training" (ICLR 2024 Workshop)**
- 关键发现：提出 TCM 启发式方法——先用 TypiClust（diversity）再切换到 Margin（uncertainty）
- 直接证据：diversity-based 策略在早期（低数据量）更好，uncertainty-based 在后期（高数据量）更好
- 局限：切换时机是固定的启发式规则，而非自适应的

**Cebron (被引 87) "Active Learning for Object Classification: From Exploration to Exploitation"**
- 关键发现：在分类迭代过程中，exploration 的影响自然递减，exploitation 的影响自然递增
- 理论支撑：AL 中确实存在"阶段性"的策略需求变化

**Tuia et al. (2009, 2011, 2012) 遥感主动学习系列 (被引 638+694+80)**
- 遥感 AL 的经典综述和方法论
- 指出遥感数据的特殊性（空间相关性、混合像素、类别不平衡）对 AL 策略选择的影响

**Yang et al. (2015) "Multi-class AL by Uncertainty Sampling with Diversity Maximization" (被引 620)**
- 提出同时考虑 uncertainty 和 diversity 的框架
- 但权重是固定的，不随学习阶段动态调整

## 2. AAL-SD 定位相关

**Xia et al. (2025) "From Selection to Generation: A Survey of LLM-based Active Learning"**
- 最新综述，分类了 LLM 在 AL 中的角色
- LLM 作为"元策略控制器"的角色定位在综述中未被覆盖——AAL-SD 的定位确实是新颖的

## 3. RL 后评价/过拟合/收敛问题的经典解决方案

**Arjona-Medina et al. (2019) "RUDDER: Return Decomposition for Delayed Rewards" (NeurIPS 2019)**
- 核心思想：将延迟奖励分解为即时奖励，通过 LSTM 的贡献分析将 return 重新分配到各时间步
- 与 RASL 的关联：RASL 中 Agent 的奖励（mIoU 变化）是延迟的（需要完成一轮训练才能获得），RUDDER 的思想可以用于将"轮级奖励"分解为"epoch级即时信号"

**Harutyunyan et al. (2019) "Hindsight Credit Assignment" (NeurIPS 2019, 被引 122)**
- 核心思想：利用事后信息（hindsight）来高效地进行信用分配
- 与 RASL 的关联：Agent 可以在获得最终 mIoU 后，回溯分析哪些采样决策贡献最大

**Han et al. (2022) "Off-Policy RL with Delayed Rewards" (ICML 2022)**
- 研究了深度 RL 在延迟奖励下的算法设计

**Xie et al. (2022) "Adaptive Deep RL for Non-Stationary Environments"**
- 研究了 RL 在非平稳环境中的适应性问题，与 AL 中模型状态持续变化的场景高度相关

## 4. LLM In-Context RL：替代参数化 RL 的理论依据

**Monea et al. (2024) "LLMs Are In-Context Bandit Reinforcement Learners" (COLM 2025, 被引 15)**
- 关键发现：LLM 能够在上下文中进行在线学习，从外部奖励信号中学习，而非仅从监督数据中学习
- 直接支撑 RASL：证明了 LLM 具备 in-context RL 能力，可以替代传统参数化 RL 策略网络
- 局限：发现 LLM 在隐式推理错误方面存在根本性限制

**Song et al. (2025) "Reward Is Enough: LLMs Are In-Context Reinforcement Learners" (arXiv 2506.06303, 2026.01 v4)**
- 关键发现：RL 在 LLM 推理时自然涌现（in-context RL, ICRL）
- 提出 ICRL prompting 框架：每轮响应后给予数值奖励，下一轮将所有历史响应和奖励拼接到上下文中
- 在 Game of 24、创意写作、ScienceWorld 等任务上显著优于 Self-Refine 和 Reflexion
- 直接支撑 RASL：即使奖励信号由同一个 LLM 生成，ICRL 仍能提升性能

**Dherin et al. (2025) "The Implicit Dynamics of In-Context Learning" (被引 19)**
- 理论分析：ICL 的隐式动力学，LLM 在推理时如何"学习"

## 5. 主动学习策略的动态选择与组合

**Hsu & Lin (2015) "Active Learning by Learning" (AAAI 2015, 被引 252)**
- 核心思想：将 AL 策略选择建模为多臂老虎机（MAB）问题
- 设计 ALBL 算法：自适应地从一组给定策略中学习最优选择
- 与 RASL 的关联：ALBL 是 RASL 的直接前驱工作，但它使用的是 MAB 而非 LLM

**Hacohen et al. (2023) "How to Select Which Active Learning Strategy is Best Suited for Your Task" (NeurIPS 2023, 被引 22)**
- 关键发现：不同的查询策略在不同条件和预算约束下表现差异显著
- 直接支撑 RASL 的动机：需要一个能够根据条件动态选择策略的机制

**Dynamic Ensemble Active Learning (DEAL, 2018)**
- 将动态 AL 策略选择建模为非平稳老虎机问题
- 使用 expert advice 算法来动态组合多个 AL 策略

**Fang et al. (2017) "Learning How to Active Learn" (EMNLP 2017, 被引 418)**
- 首次将 AL 策略学习建模为 RL 问题
- 使用 DQN 学习数据选择策略
- 与 RASL 的关联：RASL 的直接前驱，但 Fang 使用参数化 RL，RASL 使用 LLM in-context learning

**Pang et al. (2018) "Meta-Learning Transferable AL Policies by Deep RL" (被引 106)**
- 提出跨数据集的 AL 策略迁移
- 使用 dataset embedding 实现数据集无关的 AL 策略训练

## 6. LLM Agent 经验学习的核心文献

**Zhao et al. (2024) "ExpeL: LLM Agents Are Experiential Learners" (AAAI 2024, 被引 446)**
- 核心思想：Agent 自主收集经验并用自然语言提取知识，存储在记忆中用于跨任务推理
- 与 RASL 的关联：ExpeL 证明了"经验 → 知识 → 决策"这一路径的可行性，RASL 将其应用于 AL 策略调度

**Shinn et al. (2023) "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023, 被引 3338)**
- 核心思想：通过语言反馈而非权重更新来强化 Agent，Agent 将反思文本存储在情景记忆中
- 与 RASL 的关联：Reflexion 的"verbal RL"范式是 RASL 的重要理论基础

**Luo et al. (2026) "From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms"**
- 最新综述，系统梳理了 LLM Agent 记忆机制的演进

## 7. 超参数对训练梯度的影响（理论基础）

**Chen et al. (2018) "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (ICML 2018, 被引 2083)**
- 核心思想：通过梯度归一化自动平衡多任务学习中的损失权重
- 直接支撑 RASL 理论：证明了损失权重（类似于 λ）对梯度方向和训练速率的决定性影响
- 关键公式：通过调节 loss weight 使得不同任务的梯度范数保持平衡
