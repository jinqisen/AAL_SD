import numpy as np

class LLMRandomSampler:
    """
    LLM-RS 基线：仅基于随机分数（Random Score）进行决策
    
    目的：验证AD-KUCS中不确定性-知识混合策略的有效性。
    通过完全随机选择样本，作为最弱基线，对比其他方法的效果。
    
    实现说明：此基线直接随机选择样本，不涉及任何不确定性或知识增益计算。
    这与RandomSampler类似，但作为LLM对照组的一部分，用于对比完整AD-KUCS的效果。
    """
    
    def __init__(self, config=None):
        self.config = config
        np.random.seed(getattr(config, 'RANDOM_SEED', 42) if config else 42)
    
    def rank_samples(self, unlabeled_info: dict, **kwargs) -> list:
        """
        随机对样本进行排序
        
        Args:
            unlabeled_info: {sample_id: info}
        
        Returns:
            随机排序后的样本列表
        """
        sample_ids = list(unlabeled_info.keys())
        random_scores = np.random.rand(len(sample_ids))
        
        scores = []
        for sample_id, random_score in zip(sample_ids, random_scores):
            scores.append({
                "sample_id": sample_id,
                "final_score": float(random_score)
            })
        
        scores.sort(key=lambda x: x["final_score"], reverse=True)
        return scores
