# RASL 方法论创新性查新记录

## 查新点 1：LLM Agent 作为主动学习策略选择器

### 已有工作

**1. Xia et al. (2025) "From Selection to Generation: A Survey of LLM-based Active Learning"**
- 这是一篇全面的综述，分类了 LLM 在 AL 中的角色
- LLM 在 AL 中的角色主要是：(a) 直接选择样本 (b) 生成新样本 (c) 标注样本
- **关键发现**：综述中没有提到 LLM 作为"元策略控制器"（即动态选择使用哪种 AL 策略）的工作
- LLM 的角色是"替代传统 AL 策略"，而不是"在多种策略间动态切换"

**2. ActiveLLM (Bayer & Reuter, 2024/2026, TACL, 被引 26)**
- 用 GPT-4/Llama/Mistral 直接选择"最有信息量的样本"
- LLM 看到未标注样本的文本内容，直接判断哪些值得标注
- **本质**：LLM 替代了 uncertainty/diversity 等传统 AL 策略，成为一个"端到端的选择器"
- **与 RASL 的区别**：
  - ActiveLLM：LLM 直接看数据内容做选择（仅适用于 NLP 文本任务）
  - RASL：LLM 不看数据内容，而是看"学习状态"来决定采样策略的超参数 λ
  - ActiveLLM 是"LLM 做 AL"，RASL 是"LLM 控制 AL 的策略"
  - ActiveLLM 无法处理图像/遥感数据，RASL 可以

**3. Expert-Integrated Active Learning for Optimizing LLM Agents (Li et al., OpenReview)**
- 结合 RL post-training 和 AL 来优化 LLM Agent
- 方向是用 AL 来优化 LLM，而非用 LLM 来优化 AL

### 查新结论
- **"LLM 作为 AL 的元策略控制器（meta-strategy controller）"这一具体角色定位，目前没有发现直接相同的工作。**
- 已有工作中 LLM 在 AL 中的角色是"直接选择样本"或"标注样本"，而非"动态调度采样策略的超参数"。
- 但需注意：这个创新点的"增量"可能被质疑为"工程层面的区别"，需要从理论上论证为什么"控制策略"比"直接选择"更优。

## 查新点 2：RAG 增强 LLM 决策 / 经验检索引导策略

### 已有工作（高度相关，构成直接威胁）

**1. ExpeL (Zhao et al., AAAI 2024, 被引 446)**
- LLM Agent 从训练任务中自主收集经验，提取自然语言形式的知识
- 在新任务中检索相关经验来指导决策
- **与 RASL 的相似度：极高**。ExpeL 的核心思路——"收集经验 → 提取规则 → 检索应用"——与 RASL 的"构建策略记忆库 → 检索相似经验 → 注入 LLM 上下文"几乎同构
- **关键区别**：ExpeL 面向的是通用 LLM Agent 任务（如 ALFWorld, WebShop），而非主动学习的超参数调度

**2. Reflexion (Shinn et al., NeurIPS 2023, 被引 3362)**
- LLM Agent 通过"语言反思"（verbal reinforcement）从失败中学习
- 将反思结果存入记忆，在下一轮决策中作为上下文
- **与 RASL 的相似度：中高**。Reflexion 的"从历史中学习"机制与 RASL 类似，但 Reflexion 是单任务内的自我反思，不涉及跨任务经验迁移

**3. Retrieval-Augmented Reinforcement Learning (Goyal et al., ICML 2022)**
- 用检索过程增强 RL Agent，从经验数据集中检索有用信息
- 训练一个神经网络来做检索，而非用 LLM
- **与 RASL 的相似度：中**。概念上相似（检索增强决策），但技术路径完全不同（参数化检索 vs LLM in-context）

**4. Meta-Policy Reflexion (Wu et al., 2025)**
- 维护一个"元策略记忆"，包含从过去轨迹中提取的规则
- 在每步决策时检索相关规则子集
- **与 RASL 的相似度：高**。"元策略记忆 + 检索"与 RASL 的"策略记忆库 + 检索"非常接近

**5. Cross-Task Experiential Learning (Li et al., 2025)**
- 跨任务的经验学习，收集和量化 Agent 交互轨迹作为经验
- **与 RASL 的相似度：高**。直接涉及"跨任务经验迁移"

**6. Self-generated In-Context Examples (Sarukkai et al., 2025, 被引 11)**
- Agent 自动生成 in-context examples 来改善序列决策
- 在每个决策点检索相关的先前经验
- **与 RASL 的相似度：高**。"检索先前经验作为 in-context examples"与 RASL 核心机制一致

### 查新结论
- **"经验检索增强 LLM Agent 决策"这一范式已经是一个成熟的研究方向**，ExpeL (446引用) 和 Reflexion (3362引用) 是代表性工作。
- RASL 如果仅仅是"检索历史经验 → 注入 LLM 上下文 → 做决策"，那么**方法论层面的创新性不足**。
- **RASL 的潜在差异化**在于：(a) 应用场景是主动学习的超参数调度，而非通用任务 (b) 经验的形式是"学习动态特征 + 采样策略 + 性能结果"，而非通用的任务轨迹。但这更像是"应用层面的创新"而非"方法论层面的创新"。

## 查新点 3：主动学习中的跨任务策略迁移

### 已有工作（高度相关，构成直接竞争）

**1. Konyushkova et al. (2018) "Meta-Learning Transferable Active Learning Policies by Deep RL"**
- 用 RL 训练一个可迁移的 AL 策略网络
- 通过多任务训练 + dataset embedding，学习跨数据集通用的 AL 策略
- **与 RASL 的相似度：极高**。核心目标完全一致——"学习一个可跨任务迁移的 AL 策略"
- **关键区别**：用参数化 RL 网络 vs RASL 用 LLM in-context learning

**2. Hu et al. (NeurIPS 2020, 被引 96) "Graph Policy Network for Transferable Active Learning"**
- 用 GNN 策略网络 + RL 学习可迁移的 AL 策略
- 在源图上联合训练，直接泛化到未标注的目标图
- **与 RASL 的相似度：高**。同样是"跨任务迁移 AL 策略"

**3. Martins et al. (2023, Pattern Recognition, 被引 41) "Meta-learning for Dynamic Tuning of AL"**
- 用元学习动态调节 Uncertainty Sampling 的阈值
- 利用统计元特征来推荐合适的阈值
- **与 RASL 的相似度：极高且直接**。这几乎就是 RASL 要做的事情——"动态调节 AL 的超参数"，只是用元学习而非 LLM

**4. Cross-Task and Cross-Model Active Learning with Meta Features**
- 定义了专门用于 AL 的元特征 MF(x_t, L_t, U_t, f_t)
- 用这些元特征实现跨任务和跨模型的 AL 策略迁移

### 查新结论
- **"跨任务迁移 AL 策略"已经是一个有大量工作的成熟方向**。
- 特别是 Konyushkova et al. (2018) 和 Martins et al. (2023) 的工作，与 RASL 的目标高度重叠。
- **RASL 的差异化**在于用 LLM 替代参数化网络来做策略推断，但这个差异化的学术价值需要被论证——为什么 LLM 比参数化元学习网络更好？
- 可能的论证角度：(a) LLM 不需要针对新任务重新训练 (b) LLM 可以利用自然语言形式的领域知识 (c) LLM 的 in-context learning 天然支持 few-shot 策略适应

## 查新点 4：主动学习中的动态超参数/策略调度

### 已有工作

**1. Martins et al. (2023, Pattern Recognition, 被引 41) "Meta-learning for Dynamic Tuning of AL on Stream Classification"**
- 用元学习动态调节 Uncertainty Sampling 的阈值
- **与 RASL 的相似度：极高**。本质上就是"动态调节 AL 超参数"

**2. D2ADA (Wu et al., ECCV 2022, 被引 25) "Dynamic Density-aware Active Domain Adaptation"**
- 设计了动态调度策略来调整 model uncertainty 和 domain exploration 之间的预算分配
- **与 RASL 的相似度：高**。同样是动态调节 uncertainty 和 diversity 的权重

**3. Bridging Diversity and Uncertainty in AL with Self-Supervised Pre-training (2024)**
- 研究了何时从 diversity-based (TypiClust) 切换到 uncertainty-based (Margin) 策略
- **与 RASL 的相似度：中高**。研究了 U-K 切换的时机，但用的是启发式规则而非学习

**4. DWBA-ADA (2024, IEEE) "Dynamic Weighting and Boundary-Aware Active Domain Adaptation"**
- 动态加权的主动域适应方法，用于语义分割
- **与 RASL 的相似度：中**。在语义分割场景下动态调权

**5. RIPU (Xie et al., CVPR 2022) "Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation"**
- 结合区域不纯度和预测不确定性进行主动学习
- 权重系数 α1, α2 是固定的（0.1 和 1.0），不是动态调节的

### 查新结论
- "动态调节 AL 中 uncertainty 和 diversity 的权重"这一思路已有多项工作，但大多数使用的是：
  (a) 固定权重 (b) 启发式规则 (c) 参数化元学习网络
- **用 LLM 来做这个动态调节**确实是一个新的技术路径
- 但学术上需要论证：LLM 做这个调节比参数化方法有什么本质优势？
