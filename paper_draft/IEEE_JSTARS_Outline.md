# IEEE JSTARS 论文详细大纲

## 论文标题
**LLM Agent-Driven Adaptive Active Learning for Landslide Semantic Segmentation from Remote Sensing Imagery**

## 摘要 (Abstract, ~250 words)
- 研究问题：深度学习滑坡检测依赖大量标注数据，标注成本高昂
- 现有方法局限：传统主动学习策略依赖单一数学度量，难以平衡"探索"与"利用"
- 本文方法：提出AAL-SD框架，将LLM Agent引入主动学习决策环；设计AD-KUCS动态混合查询算法
- 核心贡献：Score(x) = (1-λ_t)×U(x) + λ_t×K(x)，λ_t基于Sigmoid动态调整
- 实验结果：在Landslide4Sense数据集上，ALC达到0.76，mIoU达到0.76

## 关键词
Active Learning; Landslide Detection; Semantic Segmentation; LLM Agent; Remote Sensing; Deep Learning

---

## 1. 引言 (Introduction, ~1500 words)

### 1.1 研究背景与问题 (400 words)
- 滑坡灾害的严重性和早期识别的意义
- 深度学习在滑坡检测中的成功应用
- 核心问题：标注成本高昂，数据依赖性强
- 主动学习作为解决方案的潜力

### 1.2 主动学习在遥感中的研究现状 (500 words)
- 传统主动学习方法（随机、不确定性、多样性采样）
- 现有方法的局限性：
  * 单一度量难以平衡探索与利用
  * 缺乏可解释性
  * 固定策略无法适应学习进度
- LLM Agent在机器学习中的新兴应用

### 1.3 本文贡献 (600 words)
- 首次将LLM Agent引入遥感主动学习决策环
- 提出AD-KUCS动态混合查询策略
- 构建完整的AAL-SD实验框架
- 在Landslide4Sense数据集上验证有效性

---

## 2. 相关工作 (Related Work, ~1200 words)

### 2.1 主动学习在语义分割中的应用 (400 words)
- 像素级不确定性度量（熵、MC Dropout、BALD）
- 多样性采样策略（Core-Set、DBAL）
- 现有方法在遥感图像分割中的挑战

### 2.2 滑坡检测的深度学习方法 (400 words)
- U-Net、DeepLabV3+等语义分割网络
- Landslide4Sense数据集介绍
- 标注数据稀缺的现状

### 2.3 大型语言模型在机器学习中的应用 (400 words)
- LLM作为推理和决策代理
- Chain-of-Thought推理能力
- LLM在主动学习中的初步探索

---

## 3. 方法论 (Methodology, ~2000 words)

### 3.1 问题定义与框架概述 (400 words)
- 主动学习形式化定义
- AAL-SD框架整体架构
- 决策环中的LLM Agent角色

### 3.2 AD-KUCS查询策略 (600 words)
- 不确定性度量U(x)：基于像素级熵
- 知识增益度量K(x)：基于Core-Set覆盖度
- 动态权重λ_t：Sigmoid函数控制
- 自适应调整机制

### 3.3 LLM Agent决策系统 (600 words)
- Agent架构设计
- Prompt模板设计（包含Chain-of-Thought）
- 工具箱定义（get_system_status, get_top_k_samples等）
- 决策可解释性机制

### 3.4 风险控制与回滚机制 (400 words)
- 过拟合风险检测
- CI置信区间监控
- 自适应回滚策略

---

## 4. 实验设置 (Experimental Setup, ~1000 words)

### 4.1 数据集 (300 words)
- Landslide4Sense数据集介绍
- 数据划分：初始标注集5%，测试集20%
- 预处理流程

### 4.2 基线方法 (300 words)
- Random Sampling
- Entropy Sampling
- Core-Set Sampling
- BALD Sampling
- DIAL-style / Wang-style

### 4.3 评估指标 (200 words)
- Area Under Curve (ALC)
- Mean Intersection over Union (mIoU)
- F1-Score

### 4.4 实现细节 (200 words)
- DeepLabV3+模型配置
- 训练参数设置
- 主动学习轮次与预算

---

## 5. 实验结果 (Experimental Results, ~1500 words)

### 5.1 整体性能对比 (500 words)
- ALC性能对比表
- 最终mIoU对比
- 收敛曲线分析

### 5.2 消融实验 (500 words)
- 有/无Agent对比
- 固定λ vs 动态λ
- 不确定性-only vs 知识增益-only

### 5.3 案例分析 (500 words)
- λ_t动态变化轨迹
- Agent决策解释示例
- 样本选择可视化

---

## 6. 讨论 (Discussion, ~800 words)

### 6.1 方法优势分析 (300 words)
- LLM Agent带来的推理能力
- 动态策略的自适应性
- 可解释性价值

### 6.2 局限性与未来工作 (300 words)
- 计算开销
- LLM API依赖
- 跨数据集泛化性

### 6.3 实际应用价值 (200 words)
- 降低标注成本
- 提升滑坡检测效率

---

## 7. 结论 (Conclusion, ~300 words)
- 研究工作总结
- 主要贡献回顾
- 未来研究方向

---

## 参考文献 (~40篇)
主要引用：
- Landslide4Sense原始论文
- BALD, Core-Set相关工作
- DeepLabV3+论文
- LLM Agent相关文献（ReAct, Chain-of-Thought）
- 主动学习综述论文

---

## 附录（如有）
- 额外实验数据
- 代码链接
