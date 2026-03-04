import numpy as np

class LLMUncertaintySampler:
    """
    LLM-US 基线：仅基于不确定性分数（Uncertainty Score）进行决策
    
    目的：剥离和验证Agent的复杂推理能力是否带来了超越其基础语言能力的额外增益。
    通过限制LLM只能看到不确定性分数，验证AD-KUCS中知识增益模块和混合策略的必要性。
    
    实现说明：此基线直接使用不确定性分数进行样本选择，不涉及LLM推理。
    这与EntropySampler类似，但作为LLM对照组的一部分，用于对比完整AD-KUCS的效果。
    """
    
    def __init__(self, config=None):
        self.config = config
    
    def rank_samples(self, unlabeled_info: dict, **kwargs) -> list:
        """
        根据不确定性分数对样本进行排序
        
        Args:
            unlabeled_info: {sample_id: info}
                info需要包含 'uncertainty_score' (scalar) 或 'prob_map' (tensor/array)
        
        Returns:
            排序后的样本列表，按不确定性分数降序排列
        Raises:
            ValueError: 如果大部分样本缺少不确定性信息
        """
        scores = []
        missing_count = 0
        total_count = len(unlabeled_info)
        
        if total_count == 0:
            return []
            
        for sample_id, info in unlabeled_info.items():
            uncertainty = info.get('uncertainty_score') or info.get('U')
            
            # 如果没有预计算的分数，尝试从 prob_map 计算熵
            if uncertainty is None:
                prob_map = info.get('prob_map')
                if prob_map is not None:
                    # 计算熵: -sum(p * log2(p))
                    eps = 1e-10
                    # prob_map: (C, H, W)
                    if isinstance(prob_map, np.ndarray):
                        entropy = -np.sum(prob_map * np.log2(prob_map + eps), axis=0)
                        uncertainty = float(np.mean(entropy))
                    else:
                        # 假设是 Tensor
                        import torch
                        if isinstance(prob_map, torch.Tensor):
                             p = prob_map.cpu().numpy()
                             entropy = -np.sum(p * np.log2(p + eps), axis=0)
                             uncertainty = float(np.mean(entropy))
            
            if uncertainty is None:
                missing_count += 1
                final_val = 0.0
            elif isinstance(uncertainty, np.ndarray):
                final_val = float(np.mean(uncertainty))
            elif isinstance(uncertainty, (int, float)):
                final_val = float(uncertainty)
            else:
                missing_count += 1
                final_val = 0.0
            
            scores.append({
                "sample_id": sample_id,
                "final_score": final_val
            })
            
        # 验证有效性
        if missing_count == total_count:
             raise ValueError(
                 "LLM-US Sampler: 所有样本均缺少不确定性信息 (uncertainty_score/U 或 prob_map)。"
                 "无法计算不确定性基线。"
             )
        
        if missing_count > 0:
             pass
        
        scores.sort(key=lambda x: x["final_score"], reverse=True)
        return scores
