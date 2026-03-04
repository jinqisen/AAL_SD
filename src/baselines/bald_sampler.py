import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any


class BALDSampler:
    """
    BALD (Bayesian Active Learning by Disagreement) 基线

    严格遵循 Gal et al. (2017) 和 Houlsby et al. (2011) 的原始论文实现。

    核心公式:
        MI[y, θ | x] = H[E[p(y|x,θ)]] - E[H[p(y|x,θ)]]

    其中:
    - H[E[p(y|x,θ)]]: 预测熵 (Predictive Entropy) - 总不确定性
    - E[H[p(y|x,θ)]]: 期望熵 (Expected Entropy) - 数据不确定性
    - MI: 互信息 (Mutual Information) - 模型不确定性

    **学术严谨性说明**:
    本实现不支持降级到熵计算。BALD的核心贡献在于通过MC Dropout
    采样模型参数分布，计算预测与参数之间的互信息。这与简单的不确定性
    采样（Entropy）在数学本质上是不同的算法。

    参考文献:
    - Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation.
    - Houlsby, N., et al. (2011). Bayesian active learning for classification.
    """

    def __init__(self, config=None):
        self.n_samples = getattr(config, "N_MC_SAMPLES", 10) if config else 10
        self.device = getattr(config, "DEVICE", "cpu") if config else "cpu"
        self._validate_config()

    def _validate_config(self):
        """验证配置参数的有效性"""
        if self.n_samples < 1:
            raise ValueError(f"N_MC_SAMPLES must be >= 1, got {self.n_samples}")

    def _calculate_mutual_information(self, mc_predictions: np.ndarray) -> float:
        """
        计算互信息（BALD分数）

        实现公式: MI = H[E[p]] - E[H[p]]

        Args:
            mc_predictions: (n_samples, C, H, W) 多次MC Dropout预测的概率分布

        Returns:
            互信息值（标量），范围通常在 [0, log(C)] 之间

        Raises:
            ValueError: 输入概率分布不是有效的概率分布
        """
        eps = 1e-10

        if mc_predictions.ndim != 4:
            raise ValueError(
                f"mc_predictions must be 4D (n_samples, C, H, W), "
                f"got shape {mc_predictions.shape}"
            )

        mean_pred = np.mean(mc_predictions, axis=0)

        if np.any(mean_pred < 0) or np.any(mean_pred > 1):
            raise ValueError("Predictions must be valid probabilities in [0, 1]")

        if np.any(np.abs(np.sum(mean_pred, axis=0) - 1.0) > 1e-5):
            raise ValueError("Predictions must sum to 1 along class dimension")

        predictive_entropy = -np.sum(mean_pred * np.log2(mean_pred + eps), axis=0)

        sample_entropies = -np.sum(
            mc_predictions * np.log2(mc_predictions + eps), axis=1
        )

        expected_entropy = np.mean(sample_entropies, axis=0)

        mutual_info = predictive_entropy - expected_entropy

        return float(np.mean(mutual_info))

    def _enable_mc_dropout(self, model: torch.nn.Module) -> None:
        """
        启用模型的MC Dropout模式

        将所有Dropout和Dropout2d层设置为训练模式，但不更新BN等参数
        """
        model.eval()
        for module in model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
                module.train()

    def _get_mc_predictions(
        self, model: torch.nn.Module, images: torch.Tensor
    ) -> np.ndarray:
        """
        通过MC Dropout获取多次预测

        Args:
            model: 已训练好的PyTorch模型
            images: 输入图像 (B, C, H, W)

        Returns:
            (n_samples, B, C, H, W) 多次预测的概率分布

        Raises:
            RuntimeError: 模型不支持MC Dropout
        """
        self._enable_mc_dropout(model)

        images = images.to(self.device)
        mc_preds = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                mc_preds.append(probs.cpu().numpy())

        return np.stack(mc_preds, axis=0)

    def rank_samples(
        self,
        unlabeled_info: dict,
        model: Optional[torch.nn.Module] = None,
        data_loader: Optional[Any] = None,
        **kwargs,
    ) -> list:
        """
        根据BALD分数（互信息）对样本进行排序

        **学术严谨性**: 本方法要求必须提供模型或预计算的MC predictions。
        不支持降级到单次预测熵计算。

        Args:
            unlabeled_info: {sample_id: info}
            model: PyTorch模型（必须支持MC Dropout）
            data_loader: 数据加载器
            **kwargs:
                - mc_predictions: (n_samples, C, H, W) 预计算的MC预测
                - sample_indices: 对应的样本索引列表

        Returns:
            排序后的样本列表，按BALD分数降序排列

        Raises:
            ValueError: 未提供有效的MC Dropout输入
        """
        precomputed_mc = kwargs.get("mc_predictions")
        sample_indices = kwargs.get("sample_indices", list(unlabeled_info.keys()))

        if precomputed_mc is not None:
            return self._rank_with_precomputed(
                unlabeled_info, precomputed_mc, sample_indices
            )
        elif model is not None and data_loader is not None:
            return self._rank_with_mc_inference(
                unlabeled_info, model, data_loader, sample_indices
            )
        else:
            raise ValueError(
                "BALD requires either:\n"
                "1. Pre-computed mc_predictions via kwargs\n"
                "2. Both model and data_loader for MC Dropout inference\n"
                "降级到熵计算是不允许的，因为这会混淆BALD与Entropy采样算法。"
            )

    def _rank_with_precomputed(
        self,
        unlabeled_info: dict,
        mc_predictions: np.ndarray,
        sample_indices: list,
    ) -> list:
        """使用预计算的MC predictions进行排序"""
        scores = []

        for i, sample_id in enumerate(sample_indices):
            if i < len(mc_predictions):
                sample_mc = mc_predictions[i]
                bald_score = self._calculate_mutual_information(sample_mc)
            else:
                raise ValueError(f"Not enough MC predictions for sample {sample_id}")

            scores.append(
                {
                    "sample_id": sample_id,
                    "final_score": float(bald_score),
                    "method": "BALD",
                    "n_mc_samples": self.n_samples,
                }
            )

        scores.sort(key=lambda x: x["final_score"], reverse=True)
        return scores

    def _rank_with_mc_inference(
        self,
        unlabeled_info: dict,
        model: torch.nn.Module,
        data_loader: Any,
        sample_indices: list,
    ) -> list:
        """通过MC Dropout推理进行排序"""
        scores = []
        sample_idx = 0

        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            batch_size = images.shape[0]

            try:
                mc_predictions = self._get_mc_predictions(model, images)
            except AttributeError as e:
                raise RuntimeError(
                    f"Model does not support MC Dropout. "
                    f"Ensure model has Dropout layers. Error: {e}"
                )

            for i in range(batch_size):
                if sample_idx >= len(sample_indices):
                    break

                sample_id = sample_indices[sample_idx]
                sample_mc = mc_predictions[:, i, :, :, :]
                bald_score = self._calculate_mutual_information(sample_mc)

                scores.append(
                    {
                        "sample_id": sample_id,
                        "final_score": float(bald_score),
                        "method": "BALD",
                        "n_mc_samples": self.n_samples,
                    }
                )

                sample_idx += 1

        scores.sort(key=lambda x: x["final_score"], reverse=True)
        return scores
