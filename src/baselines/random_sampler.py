
import numpy as np

class RandomSampler:
    """随机采样基线"""
    
    def __init__(self, config=None):
        pass

    def rank_samples(self, unlabeled_info: dict, **kwargs) -> list:
        """对所有未标注样本进行随机排序。"""
        # unlabeled_info is a dict {sample_id: info}
        # We just need keys
        sample_ids = list(unlabeled_info.keys())
        np.random.shuffle(sample_ids)
        return [
            {"sample_id": sample_id, "final_score": float(np.random.rand())}
            for sample_id in sample_ids
        ]
