import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.config import AgentThresholds


class ADKUCSSampler:
    def __init__(
        self,
        device="cpu",
        alpha=5.0,
        score_normalization: bool = True,
        feature_num_workers: int = 0,
        feature_persistent_workers: bool = False,
        feature_prefetch_factor: int = 2,
        feature_pin_memory: bool = False,
    ):
        self.device = device
        self.alpha = alpha
        self.score_normalization = bool(score_normalization)
        self.current_round = None
        self.uncertainty_calibration = None
        self.uncertainty_method = "entropy"
        self.n_mc_samples = 10
        self.uncertainty_aggregation = "mean"
        self.entropy_threshold = 0.5
        self.feature_num_workers = int(feature_num_workers or 0)
        self.feature_persistent_workers = bool(feature_persistent_workers)
        self.feature_prefetch_factor = int(feature_prefetch_factor or 2)
        self.feature_pin_memory = bool(feature_pin_memory)

    def configure_from_exp(self, exp_config: dict | None) -> None:
        if isinstance(exp_config, dict):
            cfg = exp_config.get("uncertainty_calibration")
            self.uncertainty_calibration = dict(cfg) if isinstance(cfg, dict) else None
            um = exp_config.get("uncertainty_method")
            if um is None:
                self.uncertainty_method = "entropy"
            else:
                self.uncertainty_method = str(um).strip().lower() or "entropy"
            protocol = exp_config.get("acquisition_protocol")
            protocol = protocol if isinstance(protocol, dict) else {}
            self.uncertainty_aggregation = (
                str(
                    protocol.get(
                        "uncertainty_aggregation", self.uncertainty_aggregation
                    )
                    or self.uncertainty_aggregation
                )
                .strip()
                .lower()
            )
            try:
                self.entropy_threshold = float(
                    protocol.get("entropy_threshold", self.entropy_threshold)
                    or self.entropy_threshold
                )
            except Exception:
                self.entropy_threshold = float(self.entropy_threshold or 0.5)
            try:
                self.n_mc_samples = int(exp_config.get("n_mc_samples") or 10)
            except Exception:
                self.n_mc_samples = 10
        else:
            self.uncertainty_calibration = None
            self.uncertainty_method = "entropy"
            self.n_mc_samples = 10
            self.uncertainty_aggregation = "mean"
            self.entropy_threshold = 0.5

    def set_round(self, round_idx: int | None) -> None:
        self.current_round = int(round_idx) if round_idx is not None else None

    def _unpack_images(self, batch):
        if isinstance(batch, dict):
            return batch.get("image")
        if isinstance(batch, (list, tuple)):
            return batch[0] if len(batch) >= 1 else None
        return batch

    def _should_calibrate_uncertainty(self) -> bool:
        cfg = self.uncertainty_calibration
        if not isinstance(cfg, dict):
            return False
        rounds = cfg.get("update_rounds")
        if rounds is None:
            return True
        try:
            current = (
                int(self.current_round) if self.current_round is not None else None
            )
            return current in set(int(x) for x in rounds)
        except Exception:
            return False

    def _calibrate_uncertainty_scores(self, scores: np.ndarray) -> np.ndarray:
        cfg = self.uncertainty_calibration
        if not self._should_calibrate_uncertainty():
            return scores
        if not isinstance(cfg, dict):
            return scores
        mode = str(cfg.get("mode", "")).strip().lower()
        arr = np.asarray(scores, dtype=np.float32)
        if arr.size == 0:
            return arr
        if mode == "temperature":
            tau = float(cfg.get("tau", 1.0))
            if not np.isfinite(tau) or tau <= 0.0:
                return arr
            power = 1.0 / float(tau)
            return np.power(np.maximum(arr, 0.0), power)
        if mode == "quantile":
            q_low = float(cfg.get("q_low", cfg.get("quantile_low", 0.05)))
            q_high = float(cfg.get("q_high", cfg.get("quantile_high", 0.95)))
            q_low = min(max(q_low, 0.0), 1.0)
            q_high = min(max(q_high, 0.0), 1.0)
            if q_low > q_high:
                q_low, q_high = q_high, q_low
            lo = float(np.quantile(arr, q_low))
            hi = float(np.quantile(arr, q_high))
            if not np.isfinite(lo) or not np.isfinite(hi) or float(hi - lo) < 1e-10:
                return arr
            return np.clip(arr, lo, hi)
        return arr

    def _max_pairwise_distance(self, features: np.ndarray) -> float:
        x = np.asarray(features)
        if x.ndim != 2 or len(x) < 2:
            return 0.0
        x = x.astype(np.float64, copy=False)
        try:
            from scipy.spatial.distance import pdist

            distances = pdist(x, metric="euclidean")
            max_dist = float(np.max(distances)) if len(distances) > 0 else 0.0
            return max_dist if np.isfinite(max_dist) and max_dist > 0.0 else 0.0
        except (ImportError, Exception):
            n = int(len(x))
            norms2 = np.einsum("ij,ij->i", x, x)
            max_dist2 = 0.0
            chunk = int(min(256, n))
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                gram_chunk = x[start:end] @ x.T
                dist2_chunk = norms2[start:end, None] + norms2[None, :]
                dist2_chunk = dist2_chunk - 2.0 * gram_chunk
                dist2_chunk = np.maximum(dist2_chunk, 0.0)
                cur = float(np.max(dist2_chunk))
                if cur > max_dist2:
                    max_dist2 = cur
            if not np.isfinite(max_dist2) or max_dist2 <= 0.0:
                return 0.0
            return float(np.sqrt(max_dist2))

    def _coreset_to_labeled_scores(
        self, features: np.ndarray, labeled_features: np.ndarray
    ) -> np.ndarray:
        x = np.asarray(features)
        lf = np.asarray(labeled_features)

        if x.ndim != 2 or len(x) == 0:
            return np.zeros((0,), dtype=np.float64)
        if lf.ndim != 2 or len(lf) == 0:
            raise RuntimeError("coreset-to-labeled requires non-empty labeled_features")
        if x.shape[1] != lf.shape[1]:
            raise RuntimeError(
                f"coreset-to-labeled feature dim mismatch: unlabeled_dim={x.shape[1]} labeled_dim={lf.shape[1]}"
            )

        x = x.astype(np.float64, copy=False)
        lf = lf.astype(np.float64, copy=False)

        max_dist = self._max_pairwise_distance(lf)
        if max_dist < 1e-10:
            return np.zeros((len(x),), dtype=np.float64)

        lf_norms2 = np.einsum("ij,ij->i", lf, lf)
        x_norms2 = np.einsum("ij,ij->i", x, x)
        min_dist2 = np.full((len(x),), np.inf, dtype=np.float64)
        lf_n = int(len(lf))
        chunk = int(min(256, lf_n))
        for start in range(0, lf_n, chunk):
            end = min(start + chunk, lf_n)
            dot = lf[start:end] @ x.T
            dist2 = lf_norms2[start:end, None] + x_norms2[None, :]
            dist2 = dist2 - 2.0 * dot
            dist2 = np.maximum(dist2, 0.0)
            min_dist2 = np.minimum(min_dist2, np.min(dist2, axis=0))
        min_dist = np.sqrt(min_dist2)
        scores = min_dist / float(max_dist)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores.astype(np.float64, copy=False)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores)
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val < 1e-10:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def _aggregate_uncertainty_map(self, u_map: np.ndarray) -> float:
        arr = np.asarray(u_map, dtype=np.float32)
        if arr.size == 0:
            return 0.0
        arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)
        mode = (
            str(getattr(self, "uncertainty_aggregation", "mean") or "mean")
            .strip()
            .lower()
        )
        if mode in ("mean", "full_mean", "none", ""):
            return float(np.mean(arr))
        tau = float(getattr(self, "entropy_threshold", 0.5) or 0.5)
        mask = arr > float(tau)
        if int(np.sum(mask)) <= 0:
            return float(np.mean(arr))
        return float(np.mean(arr[mask]))

    def _calculate_uncertainty(self, prob_map: np.ndarray) -> float:
        eps = 1e-10
        entropy = -np.sum(prob_map * np.log2(prob_map + eps), axis=0)
        return self._aggregate_uncertainty_map(entropy)

    def _calculate_bald_score(self, mc_predictions: np.ndarray) -> float:
        """
        计算BALD分数（互信息）

        Args:
            mc_predictions: (n_samples, C, H, W) MC Dropout预测

        Returns:
            互信息值（标量）
        """
        eps = 1e-10

        mean_pred = np.mean(mc_predictions, axis=0)
        predictive_entropy = -np.sum(mean_pred * np.log2(mean_pred + eps), axis=0)
        sample_entropies = -np.sum(
            mc_predictions * np.log2(mc_predictions + eps), axis=1
        )
        expected_entropy = np.mean(sample_entropies, axis=0)
        mutual_info = predictive_entropy - expected_entropy
        return self._aggregate_uncertainty_map(mutual_info)

    def _enable_mc_dropout(self, model: torch.nn.Module) -> None:
        model.eval()
        for module in model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
                module.train()

    def calculate_bald_scores(
        self, model, dataset, unlabeled_indices, n_mc_samples: int = 10
    ) -> tuple:
        """
        计算所有未标注样本的BALD分数

        **学术严谨性**:
        本方法通过MC Dropout采样计算BALD分数，不支持降级。

        Args:
            model: PyTorch模型（必须支持MC Dropout）
            dataset: 数据集
            unlabeled_indices: 未标注样本索引
            n_mc_samples: MC Dropout采样次数

        Returns:
            (bald_scores_list, sample_ids)
            - bald_scores_list: 每个样本的BALD分数
            - sample_ids: 对应的样本ID

        Raises:
            RuntimeError: MC Dropout采样失败
        """
        from torch.utils.data import DataLoader, Subset

        subset_u = Subset(dataset, unlabeled_indices)
        loader_kwargs = {
            "batch_size": 8,
            "shuffle": False,
            "num_workers": self.feature_num_workers,
            "persistent_workers": bool(
                self.feature_num_workers > 0 and self.feature_persistent_workers
            ),
            "pin_memory": self.feature_pin_memory,
        }
        if self.feature_num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.feature_prefetch_factor
        loader_u = DataLoader(subset_u, **loader_kwargs)
        eps = 1e-10
        bald_scores: list[float] = []
        sample_ids: list[int] = []

        model.eval()
        self._enable_mc_dropout(model)
        cursor = 0
        with torch.no_grad():
            for batch in tqdm(loader_u, desc="MC Dropout Sampling"):
                images = self._unpack_images(batch)
                if images is None:
                    continue
                images = images.to(self.device)
                batch_size = int(images.shape[0])
                sum_probs = None
                sum_ent = torch.zeros((batch_size,), device=self.device)
                for _ in range(int(n_mc_samples or 10)):
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                    sum_probs = probs if sum_probs is None else (sum_probs + probs)
                    ent = -torch.sum(probs * torch.log2(probs + eps), dim=1)
                    mode = (
                        str(getattr(self, "uncertainty_aggregation", "mean") or "mean")
                        .strip()
                        .lower()
                    )
                    if mode in ("mean", "full_mean", "none", ""):
                        sum_ent = sum_ent + ent.mean(dim=(1, 2))
                    else:
                        tau = float(getattr(self, "entropy_threshold", 0.5) or 0.5)
                        ent_mask = ent > tau
                        ent_sum_masked = (ent * ent_mask).sum(dim=(1, 2))
                        mask_count = ent_mask.sum(dim=(1, 2)) + 1e-10
                        sum_ent = sum_ent + (ent_sum_masked / mask_count)

                mean_probs = sum_probs / float(int(n_mc_samples or 10))
                pred_ent = -torch.sum(mean_probs * torch.log2(mean_probs + eps), dim=1)
                mode = (
                    str(getattr(self, "uncertainty_aggregation", "mean") or "mean")
                    .strip()
                    .lower()
                )
                if mode in ("mean", "full_mean", "none", ""):
                    pred_ent_scalar = pred_ent.mean(dim=(1, 2))
                else:
                    tau = float(getattr(self, "entropy_threshold", 0.5) or 0.5)
                    pred_ent_mask = pred_ent > tau
                    pred_ent_scalar = (pred_ent * pred_ent_mask).sum(dim=(1, 2)) / (
                        pred_ent_mask.sum(dim=(1, 2)) + 1e-10
                    )
                mi = pred_ent_scalar - (sum_ent / float(int(n_mc_samples or 10)))
                bald_scores.extend(mi.detach().cpu().tolist())
                batch_ids = unlabeled_indices[cursor : cursor + batch_size]
                sample_ids.extend(batch_ids)
                cursor += batch_size

        model.eval()
        if int(len(bald_scores)) != int(len(sample_ids)) or int(len(sample_ids)) != int(
            len(unlabeled_indices)
        ):
            raise RuntimeError("BALD score extraction failed: output size mismatch")
        return bald_scores, sample_ids

    def _cluster_features(self, features: np.ndarray, n_clusters: int = 10) -> tuple:
        if len(features) < n_clusters:
            n_clusters = max(2, len(features) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        try:
            import joblib
        except Exception:
            joblib = None
        if joblib is not None:
            with joblib.parallel_backend("threading"):
                cluster_labels = kmeans.fit_predict(features)
        else:
            cluster_labels = kmeans.fit_predict(features)
        cluster_centers = kmeans.cluster_centers_
        return cluster_labels, cluster_centers

    def _calculate_knowledge_gain_clustering(
        self,
        sample_feature: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_centers: np.ndarray,
    ) -> float:
        if cluster_centers is None or len(cluster_centers) == 0:
            return 0.0
        dists = np.linalg.norm(cluster_centers - sample_feature, axis=1)
        assigned = int(np.argmin(dists))
        distance_to_center = float(dists[assigned])
        if len(cluster_centers) == 1:
            return 0.0
        max_distance = self._max_pairwise_distance(cluster_centers)
        if max_distance < 1e-10:
            return 0.0
        return float(distance_to_center / max_distance)

    def _resolve_feature_layer(self, model):
        layer = None
        if hasattr(model, "backbone"):
            layer = getattr(model.backbone, "layer4", None)
        if (
            layer is None
            and hasattr(model, "model")
            and hasattr(model.model, "encoder")
        ):
            layer = getattr(model.model, "encoder", None)
            if layer is not None:
                layer = getattr(layer, "layer4", None)
        if layer is None:
            candidates = ["features", "avgpool", "layer4", "layer3", "mixed_7c"]
            for name in candidates:
                if hasattr(model, name):
                    layer = getattr(model, name)
                    break
                if hasattr(model, "module") and hasattr(model.module, name):
                    layer = getattr(model.module, name)
                    break
        return layer

    def _calculate_knowledge_gain(
        self, sample_feature: np.ndarray, labeled_features: np.ndarray
    ) -> float:
        if labeled_features is None or len(labeled_features) == 0:
            return 1.0
        labeled_features = np.asarray(labeled_features)
        if labeled_features.ndim != 2:
            return 1.0
        sample_feature = np.asarray(sample_feature)
        if (
            sample_feature.ndim != 1
            or labeled_features.shape[1] != sample_feature.shape[0]
        ):
            return 1.0
        dists = np.linalg.norm(labeled_features - sample_feature, axis=1)
        min_dist = float(np.min(dists))
        max_distance = self._max_pairwise_distance(labeled_features)
        if max_distance < 1e-10 or not np.isfinite(min_dist):
            return 0.0
        return float(min_dist / max_distance)

    def _get_adaptive_weight(
        self, current_iteration: int, total_iterations: int, override: float = None
    ) -> float:
        if override is not None:
            return float(override)
        total_iterations = max(int(total_iterations), 1)
        progress = float(current_iteration) / total_iterations
        return AgentThresholds.calculate_lambda_t(progress, alpha=self.alpha)

    def get_predictions_and_features(
        self, model, data_loader, mc_dropout: bool = False, n_mc_samples: int = 10
    ):
        """
        获取模型预测和特征

        Args:
            model: PyTorch模型
            data_loader: 数据加载器
            mc_dropout: 是否启用MC Dropout（用于BALD等贝叶斯方法）
            n_mc_samples: MC Dropout采样次数

        Returns:
            如果mc_dropout=False: (probs_tensor, features_tensor)
            如果mc_dropout=True: (mc_probs_dict, features_tensor)
                mc_probs_dict: {sample_idx: (n_mc_samples, C, H, W)}
        """
        model.eval()
        probs_list: list[torch.Tensor] = []
        probs_tensor = None
        total_len = None
        try:
            total_len = int(len(getattr(data_loader, "dataset", [])))
        except Exception:
            total_len = None
        write_offset = 0

        if mc_dropout:
            raise RuntimeError(
                "get_predictions_and_features(mc_dropout=True) is disabled; use get_uncertainty_and_features(method='bald') instead."
            )

        features_list: list[torch.Tensor] = []

        def hook_fn(module, input, output):
            gap = F.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
            features_list.append(gap.detach().cpu())

        layer = self._resolve_feature_layer(model)

        handle = None
        if layer is not None:
            handle = layer.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Querying Samples"):
                    images = self._unpack_images(batch)
                    if images is None:
                        continue

                    images = images.to(self.device)
                    logits = model(images)
                    probs = F.softmax(logits, dim=1)
                    probs_cpu = probs.detach().cpu()
                    batch_size = int(probs_cpu.shape[0])
                    if probs_tensor is None and total_len is not None:
                        probs_tensor = torch.empty(
                            (total_len, *probs_cpu.shape[1:]), dtype=probs_cpu.dtype
                        )
                    if probs_tensor is not None:
                        probs_tensor[write_offset : write_offset + batch_size] = (
                            probs_cpu
                        )
                    else:
                        probs_list.append(probs_cpu)
                    write_offset += batch_size
        finally:
            if handle is not None:
                handle.remove()

        features_tensor = torch.cat(features_list) if features_list else None
        if probs_tensor is not None:
            probs_tensor = probs_tensor[:write_offset]
            return probs_tensor, features_tensor
        if not probs_list:
            return None, features_tensor
        probs_tensor = torch.cat(probs_list)
        return probs_tensor, features_tensor

    def get_uncertainty_and_features(
        self,
        model,
        data_loader,
        pos_class: int | None = None,
        pos_threshold: float = 0.5,
    ):
        method = (
            str(getattr(self, "uncertainty_method", "entropy") or "entropy")
            .strip()
            .lower()
        )
        if method == "bald":
            eps = 1e-10
            uncertainties: list[float] = []
            pos_areas: list[float] | None = [] if pos_class is not None else None
            features_list: list[torch.Tensor] = []

            capture_first = {"on": True}

            def hook_fn(module, input, output):
                if not capture_first["on"]:
                    return
                gap = F.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
                features_list.append(gap.detach().cpu())
                capture_first["on"] = False

            layer = self._resolve_feature_layer(model)
            handle = layer.register_forward_hook(hook_fn) if layer is not None else None
            model.eval()
            self._enable_mc_dropout(model)
            try:
                with torch.no_grad():
                    for batch in tqdm(data_loader, desc="MC Dropout Sampling"):
                        images = self._unpack_images(batch)
                        if images is None:
                            continue

                        images = images.to(self.device)
                        batch_size = int(images.shape[0])
                        sum_probs = None
                        sum_ent = torch.zeros((batch_size,), device=self.device)
                        capture_first["on"] = True
                        for mc_idx in range(
                            int(getattr(self, "n_mc_samples", 10) or 10)
                        ):
                            logits = model(images)
                            if mc_idx == 0:
                                capture_first["on"] = False
                            probs = torch.softmax(logits, dim=1)
                            if sum_probs is None:
                                sum_probs = probs
                            else:
                                sum_probs = sum_probs + probs
                            ent = -torch.sum(probs * torch.log2(probs + eps), dim=1)
                            mode = (
                                str(
                                    getattr(self, "uncertainty_aggregation", "mean")
                                    or "mean"
                                )
                                .strip()
                                .lower()
                            )
                            if mode in ("mean", "full_mean", "none", ""):
                                sum_ent = sum_ent + ent.mean(dim=(1, 2))
                            else:
                                tau = float(
                                    getattr(self, "entropy_threshold", 0.5) or 0.5
                                )
                                ent_mask = ent > tau
                                ent_sum_masked = (ent * ent_mask).sum(dim=(1, 2))
                                mask_count = ent_mask.sum(dim=(1, 2)) + 1e-10
                                sum_ent = sum_ent + (ent_sum_masked / mask_count)

                        mean_probs = sum_probs / float(
                            int(getattr(self, "n_mc_samples", 10) or 10)
                        )
                        pred_ent = -torch.sum(
                            mean_probs * torch.log2(mean_probs + eps), dim=1
                        )
                        mode = (
                            str(
                                getattr(self, "uncertainty_aggregation", "mean")
                                or "mean"
                            )
                            .strip()
                            .lower()
                        )
                        if mode in ("mean", "full_mean", "none", ""):
                            pred_ent_scalar = pred_ent.mean(dim=(1, 2))
                        else:
                            tau = float(getattr(self, "entropy_threshold", 0.5) or 0.5)
                            pred_ent_mask = pred_ent > tau
                            pred_ent_scalar = (pred_ent * pred_ent_mask).sum(
                                dim=(1, 2)
                            ) / (pred_ent_mask.sum(dim=(1, 2)) + 1e-10)
                        mi = pred_ent_scalar - (
                            sum_ent
                            / float(int(getattr(self, "n_mc_samples", 10) or 10))
                        )
                        uncertainties.extend(mi.detach().cpu().tolist())
                        if pos_areas is not None:
                            channel = mean_probs[:, int(pos_class), :, :]
                            area = (
                                (channel > float(pos_threshold))
                                .float()
                                .mean(dim=(1, 2))
                            )
                            pos_areas.extend(area.detach().cpu().tolist())
            finally:
                model.eval()
                if handle is not None:
                    handle.remove()
            features_tensor = torch.cat(features_list) if features_list else None
            unc_arr = np.asarray(uncertainties, dtype=np.float32)
            pos_arr = (
                np.asarray(pos_areas, dtype=np.float32)
                if pos_areas is not None
                else None
            )
            return unc_arr, features_tensor, pos_arr

        model.eval()
        uncertainties: list[float] = []
        pos_areas: list[float] | None = [] if pos_class is not None else None
        features_list: list[torch.Tensor] = []

        def hook_fn(module, input, output):
            gap = F.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
            features_list.append(gap.detach().cpu())

        layer = self._resolve_feature_layer(model)

        handle = None
        if layer is not None:
            handle = layer.register_forward_hook(hook_fn)

        eps = 1e-10
        try:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Querying Samples"):
                    images = self._unpack_images(batch)
                    if images is None:
                        continue

                    images = images.to(self.device)
                    logits = model(images)
                    probs = F.softmax(logits, dim=1)

                    ent = -torch.sum(probs * torch.log2(probs + eps), dim=1)
                    mode = (
                        str(getattr(self, "uncertainty_aggregation", "mean") or "mean")
                        .strip()
                        .lower()
                    )
                    if mode in ("mean", "full_mean", "none", ""):
                        ent_mean = ent.mean(dim=(1, 2))
                    else:
                        tau = float(getattr(self, "entropy_threshold", 0.5) or 0.5)
                        ent_mask = ent > tau
                        ent_mean = (ent * ent_mask).sum(dim=(1, 2)) / (
                            ent_mask.sum(dim=(1, 2)) + 1e-10
                        )
                    uncertainties.extend(ent_mean.detach().cpu().tolist())

                    if pos_areas is not None:
                        channel = probs[:, int(pos_class), :, :]
                        area = (channel > float(pos_threshold)).float().mean(dim=(1, 2))
                        pos_areas.extend(area.detach().cpu().tolist())
        finally:
            if handle is not None:
                handle.remove()

        features_tensor = torch.cat(features_list) if features_list else None
        unc_arr = np.asarray(uncertainties, dtype=np.float32)
        pos_arr = (
            np.asarray(pos_areas, dtype=np.float32) if pos_areas is not None else None
        )
        return unc_arr, features_tensor, pos_arr

    def get_features_only(self, model, data_loader):
        model.eval()
        features_list: list[torch.Tensor] = []

        def hook_fn(module, input, output):
            gap = F.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
            features_list.append(gap.detach().cpu())

        layer = self._resolve_feature_layer(model)

        handle = None
        if layer is not None:
            handle = layer.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Querying Samples"):
                    images = self._unpack_images(batch)
                    if images is None:
                        continue

                    images = images.to(self.device)
                    _ = model(images)
        finally:
            if handle is not None:
                handle.remove()

        return torch.cat(features_list) if features_list else None

    def calculate_scores(self, model, dataset, unlabeled_indices, labeled_indices=None):
        from torch.utils.data import DataLoader, Subset

        subset_u = Subset(dataset, unlabeled_indices)
        loader_kwargs = {
            "batch_size": 8,
            "shuffle": False,
            "num_workers": self.feature_num_workers,
            "persistent_workers": bool(
                self.feature_num_workers > 0 and self.feature_persistent_workers
            ),
            "pin_memory": self.feature_pin_memory,
        }
        if self.feature_num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.feature_prefetch_factor
        loader_u = DataLoader(subset_u, **loader_kwargs)
        u_scores_arr, features_u, _ = self.get_uncertainty_and_features(model, loader_u)
        if features_u is None:
            raise RuntimeError("Feature extraction failed: missing features")
        features_u_np = features_u.numpy()

        # 1. Generate clustering on Unlabeled Features
        n_clusters = min(88, len(features_u_np))  # Query size is 88 by default
        cluster_labels, cluster_centers = self._cluster_features(
            features_u_np, n_clusters=n_clusters
        )

        # 2. Calculate Cluster-based Representativeness (K score)
        # We compute the distance to the assigned cluster center.
        # Smaller distance means MORE representative. We want higher K score for more representative samples.
        # Vectorized: compute all distances at once instead of looping
        diffs = features_u_np - cluster_centers[cluster_labels]
        k_scores_dist = np.linalg.norm(diffs, axis=1).astype(np.float32)
        max_dist = (
            k_scores_dist.max()
            if len(k_scores_dist) > 0 and k_scores_dist.max() > 0
            else 1.0
        )

        # Invert normalized distance so 1.0 is exactly at cluster center (most representative)
        # and 0.0 is the furthest outlier in any cluster
        k_scores = (1.0 - (k_scores_dist / max_dist)).tolist()

        u_scores_arr = self._calibrate_uncertainty_scores(
            np.asarray(u_scores_arr, dtype=np.float32)
        )
        u_norm = self._normalize_scores(u_scores_arr)
        k_norm = self._normalize_scores(np.array(k_scores))
        return u_norm, k_norm

    def rank_samples(
        self,
        unlabeled_info: dict,
        labeled_features: np.ndarray,
        current_iteration: int,
        total_iterations: int,
        lambda_override: float = None,
    ) -> list:
        if not isinstance(unlabeled_info, dict) or not unlabeled_info:
            return []
        sample_ids = list(unlabeled_info.keys())

        u_scores = []
        features_list = []
        for sample_id in sample_ids:
            info = unlabeled_info[sample_id]
            if isinstance(info, dict) and info.get("uncertainty_score") is not None:
                u_scores.append(float(info.get("uncertainty_score") or 0.0))
            else:
                prob_map = info.get("prob_map") if isinstance(info, dict) else None
                if prob_map is None:
                    u_scores.append(0.0)
                else:
                    u_scores.append(self._calculate_uncertainty(prob_map))
            features_list.append(info["feature"])

        features_array = np.array(features_list, dtype=np.float32)

        # 1. Generate clustering on Unlabeled Features
        n_clusters = min(88, len(features_array))  # Query size is 88 by default
        cluster_labels, cluster_centers = self._cluster_features(
            features_array, n_clusters=n_clusters
        )

        # Calculate cluster saturation for Cluster-Balanced K'
        unlabeled_counts = np.bincount(cluster_labels, minlength=n_clusters)
        labeled_counts = np.zeros(n_clusters, dtype=int)
        if labeled_features is not None and len(labeled_features) > 0:
            from scipy.spatial.distance import cdist
            # Assign labeled features to the nearest unlabeled cluster center
            dists = cdist(labeled_features, cluster_centers)
            labeled_cluster_assignments = np.argmin(dists, axis=1)
            labeled_counts = np.bincount(labeled_cluster_assignments, minlength=n_clusters)

        pool_counts = unlabeled_counts + labeled_counts
        
        # Laplace smoothing: (labeled + eps) / (pool + eps) to prevent numerical instability
        eps = 1.0  # Changed to 1.0 as per AAL-SD-Doc methodology
        sat = (labeled_counts + eps) / (pool_counts + eps)
        
        # Calculate cluster weights: w(c) = (1 - sat(c))^gamma
        gamma = 1.0  # Linear penalty
        
        # Apply noise cluster filtering (min_cluster_size)
        min_cluster_size = 5
        w_c = np.zeros(n_clusters, dtype=np.float32)
        for c in range(n_clusters):
            if pool_counts[c] < min_cluster_size:
                w_c[c] = 0.0  # Filter out noise clusters
            else:
                w_c[c] = np.power(max(1.0 - sat[c], 0.0), gamma)
                
        # Apply ratio floor to prevent winner-takes-all
        ratio_floor = 0.1
        w_max = np.max(w_c)
        if w_max > 0:
            for c in range(n_clusters):
                if w_c[c] > 0:
                    w_c[c] = max(w_c[c], w_max * ratio_floor)
        
        w_c = w_c / (np.sum(w_c) + 1e-10)  # Normalize weights
        
        # 2. Calculate Cluster-based Representativeness (K score)
        diffs = features_array - cluster_centers[cluster_labels]
        k_scores_dist = np.linalg.norm(diffs, axis=1).astype(np.float32)
        max_dist = (
            k_scores_dist.max()
            if len(k_scores_dist) > 0 and k_scores_dist.max() > 0
            else 1.0
        )

        # Invert normalized distance so 1.0 is exactly at cluster center (most representative)
        # and 0.0 is the furthest outlier.
        k_scores_raw = 1.0 - (k_scores_dist / max_dist)
        
        # Apply cluster weights to K score (Cluster-Balanced K')
        sample_weights = w_c[cluster_labels]
        k_scores = (k_scores_raw * sample_weights).tolist()

        u_scores_arr = self._calibrate_uncertainty_scores(
            np.array(u_scores, dtype=np.float32)
        )
        if self.score_normalization:
            u_scores_norm = self._normalize_scores(u_scores_arr)
            k_scores_norm = self._normalize_scores(np.array(k_scores))
        else:
            u_scores_norm = u_scores_arr
            k_scores_norm = np.array(k_scores, dtype=np.float32)
        lambda_t = self._get_adaptive_weight(
            current_iteration, total_iterations, lambda_override
        )
        final_scores = (1 - lambda_t) * u_scores_norm + lambda_t * k_scores_norm
        ranked_list = sorted(
            zip(sample_ids, final_scores, u_scores_norm, k_scores_norm),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {
                "sample_id": item[0],
                "final_score": float(item[1]),
                "uncertainty": float(item[2]),
                "knowledge_gain": float(item[3]),
                "lambda_t": float(lambda_t),
            }
            for item in ranked_list
        ]
