# AAL-SD框架Vibe Coding实施方案 v2.0 (最终版)

**面向AI辅助编码工具的详细技术文档**

---

**项目名称**: AAL-SD (Agent-based Active Learning for Scientific Discovery)  
**应用领域**: 小样本滑坡遥感图像识别  
**目标期刊**: SCI四区  
**文档版本**: v2.0 (最终版)  
**创建日期**: 2026年1月29日  

---

## 目录

- [第一部分：项目整体结构与模块划分](#第一部分项目整体结构与模块划分)
  - [1.1 核心目标](#11-核心目标)
  - [1.2 项目根目录结构 (已修订)](#12-项目根目录结构-已修订)
  - [1.3 模块职责详解 (已修订)](#13-模块职责详解-已修订)
- [第二部分：模块接口定义与数据流](#第二部分模块接口定义与数据流)
  - [2.1 核心类接口定义 (已修订)](#21-核心类接口定义-已修订)
  - [2.2 数据流图](#22-数据流图)
- [第三部分：实验设计、基线与评估体系 (新增)](#第三部分实验设计基线与评估体系-新增)
  - [3.1 数据预处理与划分](#31-数据预处理与划分)
  - [3.2 基线方法实现](#32-基线方法实现)
  - [3.3 消融实验配置](#33-消融实验配置)
  - [3.4 评估指标与方法](#34-评估指标与方法)
- [第四部分：Agent Prompt模板与工具定义](#第四部分agent-prompt模板与工具定义)
  - [4.1 System Prompt模板 (已修订)](#41-system-prompt模板-已修订)
  - [4.2 工具箱详细实现 (已修订)](#42-工具箱详细实现-已修订)
  - [4.3 Agent管理器完整实现 (已修订)](#43-agent管理器完整实现-已修订)
- [第五部分：开发顺序、测试验证与AI编码提示](#第五部分开发顺序测试验证与ai编码提示)
  - [5.1 开发任务优先级清单 (已修订)](#51-开发任务优先级清单-已修订)
  - [5.2 每个模块的验证标准 (已修订)](#52-每个模块的验证标准-已修订)
  - [5.3 针对每个任务的AI编码提示 (已修订)](#53-针对每个任务的ai编码提示-已修订)

---

## 第一部分：项目整体结构与模块划分

### 1.1 核心目标

构建一个完整、可用于学术研究的主动学习闭环系统，该系统包含以下核心能力：
1.  使用深度学习模型（DeepLabV3+）对遥感影像进行语义分割。
2.  基于AD-KUCS算法，从一个大的未标注数据池中选择“最具信息价值”的样本。
3.  集成一个名为“GeoMind”的LLM Agent，使其能够参与并指导样本选择的决策过程。
4.  实现并对比多种基线（Random, Entropy, Core-Set）和消融实验。
5.  管理数据集（已标注、未标注），并能够增量式地微调模型。
6.  使用mIoU, F1-Score, ALC等指标全面评估算法性能。

### 1.2 项目根目录结构 (已修订)

```
landslide_aal/
├── main.py
├── config.py
├── data/
│   ├── landslide4sense/  # 原始数据集
│   └── pools/            # 数据池CSV文件
├── core/
│   ├── __init__.py
│   ├── data_preprocessing.py # <-- 新增
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   └── sampler.py
├── agent/
│   ├── __init__.py
│   ├── agent_manager.py
│   ├── prompt_template.py
│   └── toolbox.py
├── baselines/                # <-- 新增
│   ├── __init__.py
│   ├── random_sampler.py
│   ├── entropy_sampler.py
│   └── coreset_sampler.py
├── experiments/              # <-- 新增
│   ├── __init__.py
│   └── ablation_config.py
└── utils/
    ├── __init__.py
    ├── logger.py
    └── evaluation.py         # <-- 新增
```

### 1.3 模块职责详解 (已修订)

| 模块路径 | 核心职责 |
|:---|:---|
| `main.py` | **业务流程编排器**。实现主动学习的主循环逻辑，支持不同采样策略的切换。 |
| `config.py` | **全局配置中心**。管理所有路径、超参数、API密钥、损失函数选项等。 |
| `core/data_preprocessing.py` | **数据预处理器**。负责执行数据集的划分，创建初始标注集、未标注池和测试集。 |
| `core/dataset.py` | **数据加载器**。实现PyTorch Dataset，负责加载图像和掩码，并进行数据增强。 |
| `core/model.py` | **模型定义**。封装DeepLabV3+，提供加载预训练权重、修改分类头等功能。 |
| `core/trainer.py` | **模型训练器**。封装训练、评估（mIoU, F1）、特征提取的完整逻辑。 |
| `core/sampler.py` | **核心算法模块**。实现AD-KUCS算法，包括归一化处理。 |
| `agent/agent_manager.py` | **智能体中枢**。实现`AgentManager`类，负责管理ReAct交互循环和错误处理。 |
| `agent/prompt_template.py` | **Prompt模板库**。存储“GeoMind” Agent的System Prompt模板。 |
| `agent/toolbox.py` | **Agent的工具箱**。实现所有供Agent调用的Python函数，包括`set_hyperparameter`。 |
| `baselines/*.py` | **基线采样器**。实现各种用于对比实验的传统主动学习采样策略。 |
| `experiments/ablation_config.py` | **消融实验配置**。定义不同消融实验的参数组合。 |
| `utils/evaluation.py` | **评估工具**。提供计算ALC（学习曲线下面积）等高级评估指标的函数。 |
| `utils/logger.py` | **日志工具**。提供一个配置好的全局logger。 |

---

## 第二部分：模块接口定义与数据流

### 2.1 核心类接口定义 (已修订)

#### `core/data_preprocessing.py`
```python
class DataPreprocessor:
    def __init__(self, config: object):
        pass
    def create_data_pools(self):
        pass
```

#### `core/trainer.py`
```python
class Trainer:
    # ...
    def evaluate(self, val_loader: DataLoader) -> dict: # 返回值变为dict
        pass
    def extract_features(self, data_loader: DataLoader) -> dict:
        pass
```

#### `core/sampler.py`
```python
class ADKUCSSampler:
    # ...
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        pass
```

#### `agent/toolbox.py`
```python
class Toolbox:
    # ...
    def set_hyperparameter(self, alpha: float) -> str:
        pass
```

### 2.2 数据流图
(与v1.0一致，此处省略)

---

## 第三部分：实验设计、基线与评估体系 (新增)

### 3.1 数据预处理与划分
- **实现**: `core/data_preprocessing.py`
- **逻辑**: 按照《研究方案》中的比例（例如：1%初始标注，20%测试）划分数据集，并保存为CSV文件。

### 3.2 基线方法实现
- **实现**: `baselines/` 目录
- **方法**: `RandomSampler`, `EntropySampler`, `CoresetSampler`。每个类都应有`rank_samples`方法。

### 3.3 消融实验配置
- **实现**: `experiments/ablation_config.py`
- **配置**: 定义`full_model`, `no_agent`, `uncertainty_only`, `knowledge_only`等实验设置，便于`main.py`调用。

### 3.4 评估指标与方法
- **实现**: `utils/evaluation.py`
- **指标**: mIoU, F1-Score, ALC (Area under Learning Curve)。
- **方法**: `calculate_alc` 函数使用`sklearn.metrics.auc`计算学习曲线下面积。

---

## 第四部分：Agent Prompt模板与工具定义

### 4.1 System Prompt模板 (已修订)

**Vibe Coding指令**: 打开 `agent/prompt_template.py`，更新`SYSTEM_PROMPT`字符串。
```python
SYSTEM_PROMPT = """
你是一个世界顶级的遥感图像分析专家和主动学习策略师。你的名字叫“GeoMind”。
你的核心目标是辅助一个主动学习系统，以最少的标注成本，训练一个高精度的滑坡语义分割模型。

## 情境感知 (Context)
当前主动学习的状态如下:
- 总迭代轮数 (T_max): {total_iterations}
- 当前迭代轮数 (t): {current_iteration}
- 当前模型性能 (mIoU): {last_miou:.4f}
- 当前自适应权重 (λ_t): {lambda_t:.4f}

你必须根据当前所处的学习阶段（初期、中期、后期）来调整你的决策策略。初期应侧重不确定性，后期应侧重知识增益。

## 你的工作流程 (ReAct)
...
"""
```

### 4.2 工具箱详细实现 (已修订)

**Vibe Coding指令**: 打开 `agent/toolbox.py`，在`Toolbox`类中增加`set_hyperparameter`方法。
```python
class Toolbox:
    # ... (其他工具不变) ...

    def set_hyperparameter(self, alpha: float) -> str:
        """
        (高级功能) 调整AD-KUCS算法的超参数，如陡峭因子α。
        :param alpha: float, 新的陡峭因子α值。
        :return: str, 一个确认修改成功的JSON字符串。
        """
        try:
            # 此处需要一个回调机制来更新主程序的Config
            # self.config_callback('ALPHA', alpha)
            return json.dumps({"status": "success", "message": f"ALPHA set to {alpha}"})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
```

### 4.3 Agent管理器完整实现 (已修订)

**Vibe Coding指令**: 打开 `agent/agent_manager.py`，在`AgentManager`类中，重写`_execute_tool`方法以包含完整的错误处理协议。
```python
class AgentManager:
    # ... (其他方法不变) ...

    def _execute_tool(self, tool_name: str, args: list) -> str:
        """执行指定的工具并返回结果，包含错误处理。"""
        try:
            if not hasattr(self.toolbox, tool_name):
                raise ValueError(f"Unknown tool: {tool_name}")
            
            method = getattr(self.toolbox, tool_name)
            result = method(*args)
            # 假设工具本身返回的是dict或list，这里统一转为JSON字符串
            return json.dumps({"status": "success", "result": result}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error_type": type(e).__name__,
                "message": str(e)
            }, ensure_ascii=False)
```

---

## 第五部分：开发顺序、测试验证与AI编码提示

### 5.1 开发任务优先级清单 (已修订)

| 优先级 | 任务ID | 模块 | 任务描述 |
|:---:|:---:|:---|:---|
| P0 | T1 | 项目初始化 | 创建更新后的项目目录结构 |
| P0 | T2 | `config.py` | 实现全局配置类，增加损失函数选项 |
| P1 | T3 | `core/data_preprocessing.py` | 实现数据划分逻辑 |
| P1 | T4 | `core/dataset.py` | 实现PyTorch数据集类，支持数据增强 |
| P1 | T5 | `core/model.py` | 实现DeepLabV3+模型加载 |
| P1 | T6 | `core/trainer.py` | 实现训练器，支持F1评估和特征提取 |
| P2 | T7 | `core/sampler.py` | 实现AD-KUCS算法，包含归一化 |
| P2 | T8 | `agent/prompt_template.py` | 实现“GeoMind”的Prompt模板 |
| P2 | T9 | `agent/toolbox.py` | 实现Agent工具箱，包含所有工具 |
| P2 | T10 | `agent/agent_manager.py` | 实现ReAct循环管理器，包含错误处理 |
| P3 | T11 | `baselines/` | 实现所有基线采样方法 |
| P3 | T12 | `experiments/` | 定义消融实验配置 |
| P3 | T13 | `utils/evaluation.py` | 实现ALC计算 |
| P3 | T14 | `main.py` | 实现主循环，支持不同实验配置 |

### 5.2 每个模块的验证标准 (已修订)

| 任务ID | 模块 | 验证标准 |
|:---:|:---|:---|
| T3 | `data_preprocessing.py` | 生成的CSV文件行数符合预期 |
| T6 | `trainer.py` | `evaluate`方法返回包含`mIoU`和`f1_score`的字典 |
| T7 | `sampler.py` | 单元测试验证归一化后的得分在[0,1]区间 |
| T11 | `baselines/` | 每个基线采样器都能返回一个排序后的样本列表 |
| T13 | `evaluation.py` | `calculate_alc`能根据给定的性能历史计算出数值 |

### 5.3 针对每个任务的AI编码提示 (已修订)

#### **Prompt for T6: 实现`core/trainer.py` (修订版)**
```
请修订 `core/trainer.py` 文件中的 `Trainer` 类。

要求：
1. `__init__` 方法：允许通过config选择损失函数（如CrossEntropyLoss, FocalLoss）。
2. `evaluate` 方法：
   - 返回一个字典 `{"mIoU": ..., "f1_score": ...}`。
   - 使用 `sklearn.metrics` 中的 `jaccard_score` 和 `f1_score` 进行计算。
3. 新增 `extract_features(self, data_loader)` 方法：
   - 使用PyTorch的Hook机制，注册一个前向钩子到`model.backbone.layer4`。
   - 在钩子函数中，获取该层的输出特征图。
   - 对特征图进行全局平均池化（`F.adaptive_avg_pool2d`），得到每个样本的特征向量。
   - 返回一个字典 `{sample_id: feature_vector}`。
```

#### **Prompt for T7: 实现`core/sampler.py` (修订版)**
```
请修订 `core/sampler.py` 文件中的 `ADKUCSSampler` 类。

要求：
1. 新增 `_normalize_scores(self, scores: np.ndarray) -> np.ndarray` 方法：
   - 使用最大-最小归一化将得分缩放到[0, 1]区间。
   - 处理分母为0的边缘情况。
2. 修改 `rank_samples` 方法：
   - 在计算最终得分前，分别对U(x)得分列表和K(x)得分列表调用 `_normalize_scores` 方法。
   - 使用归一化后的得分计算最终的AD-KUCS得分。
```

#### **Prompt for T14: 实现`main.py` (修订版)**
```
请实现 `main.py` 文件，这是整个主动学习系统的入口点，并支持实验配置切换。

要求：
1. 使用 `argparse` 库接收命令行参数，如 `--experiment_name`，用于从`experiments/ablation_config.py`中选择实验配置。
2. 在 `main()` 函数中：
   - 根据实验配置，动态选择要使用的采样器（AD-KUCS或某个基线）。
   - 根据实验配置，决定是否启用Agent (`use_agent` 标志)。
   - 在主动学习循环结束后，调用 `utils.evaluation.calculate_alc` 计算并记录ALC值。
3. 主循环逻辑应保持清晰，每个阶段（预测、排序、决策、更新、微调、评估）都有明确的注释和函数调用。
```


---

## 附录A：完整代码实现规范

### A.1 `core/data_preprocessing.py` 完整实现

```python
# core/data_preprocessing.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """
    数据预处理器，负责执行数据集的划分。
    """

    def __init__(self, config):
        """
        初始化数据预处理器。
        :param config: 配置对象，包含DATA_DIR, POOLS_DIR, INITIAL_LABELED_SIZE, TEST_SIZE。
        """
        self.config = config
        self.raw_data_dir = config.DATA_DIR
        self.pools_dir = config.POOLS_DIR

    def create_data_pools(self):
        """
        执行完整的数据划分流程，创建初始标注集、未标注池和固定测试集。
        1. 扫描原始数据目录，获取所有图像和掩码的路径。
        2. 按照配置的比例进行划分。
        3. 将划分结果保存为三个CSV文件：`labeled_pool.csv`, `unlabeled_pool.csv`, `test_pool.csv`。
        """
        if os.path.exists(os.path.join(self.pools_dir, 'labeled_pool.csv')):
            print("Data pools already exist. Skipping creation.")
            return

        image_dir = os.path.join(self.raw_data_dir, 'images')
        mask_dir = os.path.join(self.raw_data_dir, 'masks')
        
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.tif', '.jpg'))])
        
        df = pd.DataFrame({
            'sample_id': [os.path.splitext(f)[0] for f in image_files],
            'image_path': [os.path.join(image_dir, f) for f in image_files],
            'mask_path': [os.path.join(mask_dir, f) for f in image_files]
        })

        # 划分测试集
        train_val_df, test_df = train_test_split(
            df, test_size=self.config.TEST_SIZE, random_state=42
        )
        
        # 划分初始标注集和未标注池
        labeled_df, unlabeled_df = train_test_split(
            train_val_df, train_size=self.config.INITIAL_LABELED_SIZE, random_state=42
        )

        # 保存文件
        os.makedirs(self.pools_dir, exist_ok=True)
        labeled_df.to_csv(os.path.join(self.pools_dir, 'labeled_pool.csv'), index=False)
        unlabeled_df.to_csv(os.path.join(self.pools_dir, 'unlabeled_pool.csv'), index=False)
        test_df.to_csv(os.path.join(self.pools_dir, 'test_pool.csv'), index=False)

        print(f"Created data pools:")
        print(f"- Labeled pool: {len(labeled_df)} samples")
        print(f"- Unlabeled pool: {len(unlabeled_df)} samples")
        print(f"- Test pool: {len(test_df)} samples")
```

### A.2 `baselines/` 目录完整实现

#### `baselines/random_sampler.py`
```python
# baselines/random_sampler.py
import numpy as np

class RandomSampler:
    """随机采样基线"""
    
    def __init__(self, config):
        pass

    def rank_samples(self, unlabeled_info: dict, **kwargs) -> list:
        """对所有未标注样本进行随机排序。"""
        sample_ids = list(unlabeled_info.keys())
        np.random.shuffle(sample_ids)
        return [(sample_id, np.random.rand()) for sample_id in sample_ids]
```

#### `baselines/entropy_sampler.py`
```python
# baselines/entropy_sampler.py
import numpy as np

class EntropySampler:
    """基于熵的不确定性采样基线"""
    
    def __init__(self, config):
        pass

    def _calculate_entropy(self, prob_map: np.ndarray) -> float:
        """计算像素级熵的平均值"""
        eps = 1e-10
        entropy = -np.sum(prob_map * np.log2(prob_map + eps), axis=0)
        return np.mean(entropy)

    def rank_samples(self, unlabeled_info: dict, **kwargs) -> list:
        """根据不确定性（熵）对样本进行排序。"""
        scores = []
        for sample_id, info in unlabeled_info.items():
            uncertainty = self._calculate_entropy(info['prob_map'])
            scores.append((sample_id, uncertainty))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
```

#### `baselines/coreset_sampler.py`
```python
# baselines/coreset_sampler.py
import numpy as np
from scipy.spatial.distance import cdist

class CoresetSampler:
    """Core-Set采样基线"""
    
    def __init__(self, config):
        pass

    def rank_samples(self, unlabeled_info: dict, labeled_features: np.ndarray, **kwargs) -> list:
        """根据到已标注集的最小距离对样本进行排序。"""
        scores = []
        for sample_id, info in unlabeled_info.items():
            feature = info['feature'].reshape(1, -1)
            distances = cdist(feature, labeled_features, metric='euclidean')
            min_dist = np.min(distances)
            scores.append((sample_id, min_dist))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
```

### A.3 `experiments/ablation_config.py` 完整实现

```python
# experiments/ablation_config.py

ABLATION_SETTINGS = {
    "full_model": {
        "description": "完整的AD-KUCS + Agent模型",
        "use_agent": True,
        "sampler_type": "ad_kucs",
        "lambda_override": None  # 使用动态λ
    },
    "no_agent": {
        "description": "移除Agent，直接使用argmax(Score)",
        "use_agent": False,
        "sampler_type": "ad_kucs",
        "lambda_override": None
    },
    "uncertainty_only": {
        "description": "固定λ=0，仅使用不确定性",
        "use_agent": False,
        "sampler_type": "ad_kucs",
        "lambda_override": 0.0
    },
    "knowledge_only": {
        "description": "固定λ=1，仅使用知识增益",
        "use_agent": False,
        "sampler_type": "ad_kucs",
        "lambda_override": 1.0
    },
    "fixed_lambda": {
        "description": "固定λ=0.5",
        "use_agent": False,
        "sampler_type": "ad_kucs",
        "lambda_override": 0.5
    },
    "baseline_random": {
        "description": "随机采样基线",
        "use_agent": False,
        "sampler_type": "random",
        "lambda_override": None
    },
    "baseline_entropy": {
        "description": "熵采样基线",
        "use_agent": False,
        "sampler_type": "entropy",
        "lambda_override": None
    },
    "baseline_coreset": {
        "description": "Core-Set采样基线",
        "use_agent": False,
        "sampler_type": "coreset",
        "lambda_override": None
    }
}
```

### A.4 `utils/evaluation.py` 完整实现

```python
# utils/evaluation.py
import numpy as np
from sklearn.metrics import auc

def calculate_alc(performance_history: list, budget_history: list) -> float:
    """
    计算学习曲线下面积 (Area under Learning Curve)。
    :param performance_history: 包含每轮mIoU的列表。
    :param budget_history: 包含每轮标注数量的列表。
    :return: ALC值。
    """
    sorted_indices = np.argsort(budget_history)
    sorted_budget = np.array(budget_history)[sorted_indices]
    sorted_perf = np.array(performance_history)[sorted_indices]
    
    # 归一化budget到[0, 1]
    normalized_budget = (sorted_budget - sorted_budget.min()) / (sorted_budget.max() - sorted_budget.min())
    
    return auc(normalized_budget, sorted_perf)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2) -> dict:
    """
    计算分割任务的完整评估指标。
    :return: 包含mIoU, F1-Score等指标的字典。
    """
    from sklearn.metrics import jaccard_score, f1_score
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    miou = jaccard_score(y_true_flat, y_pred_flat, average='macro')
    f1 = f1_score(y_true_flat, y_pred_flat, average='macro')
    
    return {"mIoU": miou, "f1_score": f1}
```

### A.5 `core/sampler.py` 完整实现 (修订版)

```python
# core/sampler.py
import numpy as np
from scipy.spatial.distance import cdist

class ADKUCSSampler:
    """
    AD-KUCS (自适应知识与不确定性驱动采样) 算法的核心实现。
    """

    def __init__(self, config):
        self.config = config
        self.alpha = config.ALPHA

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """将得分归一化到[0, 1]区间。"""
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val < 1e-10:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def _calculate_uncertainty(self, prob_map: np.ndarray) -> float:
        """计算单个样本的不确定性U(x)，基于像素级熵。"""
        eps = 1e-10
        entropy = -np.sum(prob_map * np.log2(prob_map + eps), axis=0)
        return np.mean(entropy)

    def _calculate_knowledge_gain(self, sample_feature: np.ndarray, labeled_features: np.ndarray) -> float:
        """计算单个样本的知识增益K(x)，基于到已标注集的最小距离。"""
        if labeled_features.shape[0] == 0:
            return 1.0
        feature = sample_feature.reshape(1, -1)
        distances = cdist(feature, labeled_features, metric='euclidean')
        return np.min(distances)

    def _get_adaptive_weight(self, current_iteration: int, total_iterations: int, override: float = None) -> float:
        """计算当前迭代的自适应权重λ_t。"""
        if override is not None:
            return override
        t_normalized = current_iteration / total_iterations
        lambda_t = 1 / (1 + np.exp(-self.alpha * (t_normalized - 0.5)))
        return lambda_t

    def rank_samples(self, unlabeled_info: dict, labeled_features: np.ndarray, 
                     current_iteration: int, lambda_override: float = None) -> list:
        """
        对所有未标注样本进行排序。
        """
        u_scores = []
        k_scores = []
        sample_ids = list(unlabeled_info.keys())

        for sample_id in sample_ids:
            info = unlabeled_info[sample_id]
            u_scores.append(self._calculate_uncertainty(info['prob_map']))
            k_scores.append(self._calculate_knowledge_gain(info['feature'], labeled_features))

        # 归一化
        u_scores_norm = self._normalize_scores(np.array(u_scores))
        k_scores_norm = self._normalize_scores(np.array(k_scores))

        # 计算自适应权重
        lambda_t = self._get_adaptive_weight(
            current_iteration, self.config.TOTAL_ITERATIONS, lambda_override
        )

        # 计算最终得分
        final_scores = (1 - lambda_t) * u_scores_norm + lambda_t * k_scores_norm

        # 组合并排序
        ranked_list = sorted(zip(sample_ids, final_scores, u_scores_norm, k_scores_norm), 
                             key=lambda x: x[1], reverse=True)
        
        # 返回包含详细信息的列表
        return [
            {
                "sample_id": item[0], 
                "final_score": item[1], 
                "uncertainty": item[2], 
                "knowledge_gain": item[3]
            } 
            for item in ranked_list
        ]
```

---

## 附录B：完整的Vibe Coding Prompts汇总

### B.1 项目初始化 (T1)
```
请在当前目录下创建一个名为 `landslide_aal` 的Python项目，包含以下目录结构：

landslide_aal/
├── main.py
├── config.py
├── data/
├── core/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   └── sampler.py
├── agent/
│   ├── __init__.py
│   ├── agent_manager.py
│   ├── prompt_template.py
│   └── toolbox.py
├── baselines/
│   ├── __init__.py
│   ├── random_sampler.py
│   ├── entropy_sampler.py
│   └── coreset_sampler.py
├── experiments/
│   ├── __init__.py
│   └── ablation_config.py
└── utils/
    ├── __init__.py
    ├── logger.py
    └── evaluation.py

所有Python文件暂时只需要包含一个占位注释。
```

### B.2 数据预处理 (T3)
```
请实现 `core/data_preprocessing.py` 文件中的 `DataPreprocessor` 类。

要求：
1. `__init__` 方法接收 `config` 对象。
2. `create_data_pools` 方法：
   - 扫描 `config.DATA_DIR` 下的 `images` 和 `masks` 目录。
   - 使用 `sklearn.model_selection.train_test_split` 进行数据划分。
   - 首先划分出测试集（`config.TEST_SIZE` 个样本）。
   - 然后从剩余数据中划分出初始标注集（`config.INITIAL_LABELED_SIZE` 个样本）。
   - 将三个数据集分别保存为 `labeled_pool.csv`, `unlabeled_pool.csv`, `test_pool.csv`。
   - CSV文件应包含 `sample_id`, `image_path`, `mask_path` 三列。
```

### B.3 基线方法 (T11)
```
请在 `baselines/` 目录下实现三个基线采样器类：

1. `random_sampler.py` 中的 `RandomSampler`:
   - `rank_samples` 方法随机打乱样本顺序并返回。

2. `entropy_sampler.py` 中的 `EntropySampler`:
   - `rank_samples` 方法根据像素级熵（不确定性）对样本进行降序排序。

3. `coreset_sampler.py` 中的 `CoresetSampler`:
   - `rank_samples` 方法根据样本到已标注集的最小欧氏距离进行降序排序。

所有类都应有统一的 `rank_samples(self, unlabeled_info: dict, **kwargs) -> list` 接口。
```

### B.4 消融实验配置 (T12)
```
请实现 `experiments/ablation_config.py` 文件。

要求：
1. 定义一个名为 `ABLATION_SETTINGS` 的字典。
2. 字典的键是实验名称（如 "full_model", "no_agent", "uncertainty_only"）。
3. 字典的值是包含以下字段的字典：
   - `description`: 实验描述
   - `use_agent`: 是否使用Agent (bool)
   - `sampler_type`: 采样器类型 ("ad_kucs", "random", "entropy", "coreset")
   - `lambda_override`: 固定的λ值，如果为None则使用动态λ
4. 至少包含以下实验配置：
   - full_model, no_agent, uncertainty_only, knowledge_only, fixed_lambda
   - baseline_random, baseline_entropy, baseline_coreset
```

### B.5 评估工具 (T13)
```
请实现 `utils/evaluation.py` 文件。

要求：
1. `calculate_alc(performance_history, budget_history)` 函数：
   - 计算学习曲线下面积 (Area under Learning Curve)。
   - 使用 `sklearn.metrics.auc` 进行计算。
   - 对budget进行归一化处理。

2. `calculate_metrics(y_true, y_pred, num_classes)` 函数：
   - 计算mIoU和F1-Score。
   - 使用 `sklearn.metrics` 中的 `jaccard_score` 和 `f1_score`。
   - 返回包含所有指标的字典。
```

---

## 附录C：文档修订记录

| 版本 | 日期 | 修订内容 |
|:---|:---|:---|
| v1.0 | 2026-01-29 | 初始版本，覆盖核心算法和Agent架构 |
| v2.0 | 2026-01-29 | 新增数据预处理、基线方法、消融实验、评估指标等完整实验体系 |

---

**文档结束**
