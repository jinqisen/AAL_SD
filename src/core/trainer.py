import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math


class Trainer:
    def __init__(self, model, config, device, training_state=None):
        self.model = model
        self.config = config
        self.device = device
        self.criterion = self._get_loss_function(config.LOSS_TYPE)
        self.optimizer = optim.Adam(model.parameters(), lr=config.LR)
        self._amp_enabled = bool(
            self.device == "cuda"
            and torch.cuda.is_available()
            and bool(getattr(self.config, "AMP_ENABLED", False))
        )
        self._amp_dtype = (
            str(getattr(self.config, "AMP_DTYPE", "float16") or "float16")
            .strip()
            .lower()
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)
        self.model.to(device)
        if (
            self.device == "cuda"
            and torch.cuda.is_available()
            and bool(getattr(self.config, "TORCH_COMPILE", False))
            and hasattr(torch, "compile")
        ):
            mode = str(
                getattr(self.config, "TORCH_COMPILE_MODE", "default") or "default"
            ).strip()
            try:
                self.model = torch.compile(self.model, mode=mode)
            except Exception:
                pass
        self._grad_probe_param_names = self._select_grad_probe_param_names()
        if isinstance(training_state, dict):
            self.training_state = training_state
        else:
            self.training_state = {
                "train_u_median_history": [],
                "train_k_median_history": [],
                "max_history_length": 5,
            }
        if "train_u_median_history" not in self.training_state:
            self.training_state["train_u_median_history"] = []
        if "train_k_median_history" not in self.training_state:
            self.training_state["train_k_median_history"] = []
        if "max_history_length" not in self.training_state:
            self.training_state["max_history_length"] = 5

    def cleanup(self):
        """显式释放资源，防止内存泄漏"""
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "criterion"):
            del self.criterion
        if hasattr(self, "scaler"):
            del self.scaler
        if hasattr(self, "model"):
            self.model.to("cpu")
            del self.model
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _update_uncertainty_history(self, round_num: int):
        u_median = self.training_state.get("train_u_median_selected")
        k_median = self.training_state.get("train_k_median_selected")
        max_len = int(self.training_state.get("max_history_length", 5) or 5)
        max_len = max(int(max_len), 0)
        if u_median is not None:
            self.training_state["train_u_median_history"].append(
                (int(round_num), float(u_median))
            )
            if (
                max_len > 0
                and len(self.training_state["train_u_median_history"]) > max_len
            ):
                self.training_state["train_u_median_history"].pop(0)
        if k_median is not None:
            self.training_state["train_k_median_history"].append(
                (int(round_num), float(k_median))
            )
            if (
                max_len > 0
                and len(self.training_state["train_k_median_history"]) > max_len
            ):
                self.training_state["train_k_median_history"].pop(0)

    def _select_grad_probe_param_names(self) -> list:
        candidates = []
        for name, p in self.model.named_parameters():
            if not getattr(p, "requires_grad", False):
                continue
            low = str(name).lower()
            if (
                ("segmentation_head" in low)
                or ("classifier" in low)
                or (low.endswith(".weight") and "head" in low)
            ):
                candidates.append(name)
        if candidates:
            return candidates[:8]
        names = [
            name
            for name, p in self.model.named_parameters()
            if getattr(p, "requires_grad", False)
        ]
        return names[-4:] if len(names) >= 4 else names

    def _split_named_params(self) -> dict:
        groups = {"backbone": [], "head": [], "other": []}
        for name, p in self.model.named_parameters():
            if not getattr(p, "requires_grad", False):
                continue
            low = str(name).lower()
            if ("encoder" in low) or ("backbone" in low):
                groups["backbone"].append(p)
            elif (
                ("decoder" in low)
                or ("segmentation_head" in low)
                or ("classifier" in low)
                or ("head" in low)
            ):
                groups["head"].append(p)
            else:
                groups["other"].append(p)
        return groups

    def _grad_global_norm(self, params: list) -> float:
        acc = 0.0
        for p in params:
            g = getattr(p, "grad", None)
            if g is None:
                continue
            v = float(g.detach().float().pow(2).sum().item())
            acc += v
        return float(math.sqrt(acc)) if acc > 0.0 else 0.0

    def _grad_probe_vector(self) -> np.ndarray | None:
        max_elems = int(getattr(self.config, "GRAD_LOG_PARAM_MAX_ELEMENTS", 0) or 0)
        chunks = []
        total = 0
        name_to_param = dict(self.model.named_parameters())
        for name in self._grad_probe_param_names:
            p = name_to_param.get(name)
            if p is None:
                continue
            g = getattr(p, "grad", None)
            if g is None:
                continue
            flat = g.detach().float().reshape(-1).cpu().numpy()
            if max_elems > 0 and (total + flat.size) > max_elems:
                remain = max(0, max_elems - total)
                if remain <= 0:
                    break
                flat = flat[:remain]
            chunks.append(flat)
            total += int(flat.size)
            if max_elems > 0 and total >= max_elems:
                break
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)

    def _cosine(self, a: np.ndarray | None, b: np.ndarray | None) -> float | None:
        if a is None or b is None:
            return None
        n = int(min(a.size, b.size))
        if n <= 0:
            return None
        aa = a[:n].astype(np.float64, copy=False)
        bb = b[:n].astype(np.float64, copy=False)
        na = float(np.linalg.norm(aa))
        nb = float(np.linalg.norm(bb))
        if na <= 0.0 or nb <= 0.0:
            return None
        return float(np.dot(aa, bb) / (na * nb))

    def _summarize(self, xs: list) -> dict:
        arr = np.asarray([float(x) for x in xs if x is not None], dtype=float)
        if arr.size == 0:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(arr.size),
        }

    def _unpack_batch(self, batch, require_mask: bool):
        images = None
        masks = None
        if isinstance(batch, dict):
            images = batch.get("image")
            masks = batch.get("mask")
        elif isinstance(batch, (list, tuple)):
            if len(batch) >= 1:
                images = batch[0]
            if len(batch) >= 2:
                masks = batch[1]
        else:
            images = batch

        if isinstance(masks, str):
            masks = None
        if isinstance(masks, list) and masks and isinstance(masks[0], str):
            masks = None
        if masks is not None and hasattr(masks, "numel") and int(masks.numel()) == 0:
            masks = None
        if require_mask and masks is None:
            raise RuntimeError("Batch is missing masks")
        return images, masks

    def _get_loss_function(self, loss_type):
        if loss_type == "FocalLoss":
            return FocalLoss()
        return nn.CrossEntropyLoss()

    def train_one_epoch(self, loader, grad_probe_loader=None):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc="Training")
        grad_enabled = bool(getattr(self.config, "GRAD_LOGGING", False))
        max_batches = int(getattr(self.config, "GRAD_LOG_MAX_BATCHES", 0) or 0)
        param_groups = self._split_named_params() if grad_enabled else None
        per_batch_total = []
        per_batch_backbone = []
        per_batch_head = []
        probe_vecs = []
        probe_cos_consecutive = []
        probe_iter = None
        if (
            grad_enabled
            and grad_probe_loader is not None
            and bool(getattr(self.config, "GRAD_LOG_VAL_ALIGNMENT", False))
        ):
            probe_iter = iter(grad_probe_loader)

        for batch in pbar:
            images, masks = self._unpack_batch(batch, require_mask=True)
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            if self._amp_enabled:
                amp_dtype = torch.float16
                if self._amp_dtype in ("bf16", "bfloat16"):
                    amp_dtype = torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()

            if grad_enabled:
                if self._amp_enabled:
                    try:
                        self.scaler.unscale_(self.optimizer)
                    except Exception:
                        pass
                try:
                    total_norm = self._grad_global_norm(
                        param_groups["backbone"]
                        + param_groups["head"]
                        + param_groups["other"]
                    )
                    per_batch_total.append(total_norm)
                    per_batch_backbone.append(
                        self._grad_global_norm(param_groups["backbone"])
                    )
                    per_batch_head.append(self._grad_global_norm(param_groups["head"]))
                except Exception:
                    pass

                if max_batches <= 0 or len(probe_vecs) < max_batches:
                    vec = None
                    try:
                        vec = self._grad_probe_vector()
                    except Exception:
                        vec = None
                    if vec is not None:
                        if probe_vecs:
                            probe_cos_consecutive.append(
                                self._cosine(probe_vecs[-1], vec)
                            )
                        probe_vecs.append(vec)
            if self._amp_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

        grad_payload = None
        if grad_enabled:
            mean_vec = None
            if probe_vecs:
                min_len = int(min(v.size for v in probe_vecs if v is not None))
                if min_len > 0:
                    stack = np.stack([v[:min_len] for v in probe_vecs], axis=0).astype(
                        np.float64, copy=False
                    )
                    mean_vec = np.mean(stack, axis=0)
            cos_to_mean = []
            if mean_vec is not None and probe_vecs:
                probe_stack = np.stack(probe_vecs, axis=0)
                norms_probe = np.linalg.norm(probe_stack, axis=1, keepdims=True)
                norm_mean = np.linalg.norm(mean_vec)
                if norm_mean > 0 and np.all(norms_probe > 0):
                    cos_to_mean = (
                        probe_stack @ mean_vec / (norms_probe.flatten() * norm_mean)
                    ).tolist()
                else:
                    cos_to_mean = [self._cosine(v, mean_vec) for v in probe_vecs]

            train_val_cos = None
            if probe_iter is not None and mean_vec is not None:
                try:
                    vb = next(probe_iter)
                    v_images, v_masks = self._unpack_batch(vb, require_mask=True)
                    v_images = v_images.to(self.device)
                    v_masks = v_masks.to(self.device)
                    was_training = self.model.training
                    self.model.eval()
                    self.optimizer.zero_grad()
                    if self._amp_enabled:
                        with torch.autocast(device_type="cuda", enabled=False):
                            v_out = self.model(v_images)
                            v_loss = self.criterion(v_out, v_masks)
                        v_loss.backward()
                    else:
                        v_out = self.model(v_images)
                        v_loss = self.criterion(v_out, v_masks)
                        v_loss.backward()
                    v_vec = self._grad_probe_vector()
                    train_val_cos = self._cosine(mean_vec, v_vec)
                    self.optimizer.zero_grad()
                    if was_training:
                        self.model.train()
                except Exception:
                    train_val_cos = None

            grad_payload = {
                "total_norm": self._summarize(per_batch_total),
                "backbone_norm": self._summarize(per_batch_backbone),
                "head_norm": self._summarize(per_batch_head),
                "cos_consecutive": self._summarize(probe_cos_consecutive),
                "cos_to_mean": self._summarize(cos_to_mean),
                "train_val_cos": train_val_cos,
                "n_probe_batches": int(len(probe_vecs)),
            }

        avg_loss = running_loss / len(loader)
        return avg_loss, grad_payload

    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        num_classes = int(getattr(self.config, "NUM_CLASSES", 2) or 2)
        k = int(num_classes)
        conf = np.zeros((k, k), dtype=np.int64) if k > 0 else None

        with torch.no_grad():
            pbar = tqdm(loader, desc="Validating")
            for batch in pbar:
                images, masks = self._unpack_batch(batch, require_mask=True)
                images = images.to(self.device)
                masks = masks.to(self.device)

                if self._amp_enabled:
                    amp_dtype = torch.float16
                    if self._amp_dtype in ("bf16", "bfloat16"):
                        amp_dtype = torch.bfloat16
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                targets = masks.detach().cpu().numpy()
                if conf is not None:
                    y_pred_flat = np.asarray(preds).reshape(-1)
                    y_true_flat = np.asarray(targets).reshape(-1)
                    y_true_flat = np.clip(
                        y_true_flat.astype(np.int64, copy=False), 0, k - 1
                    )
                    y_pred_flat = np.clip(
                        y_pred_flat.astype(np.int64, copy=False), 0, k - 1
                    )
                    indices = y_true_flat * k + y_pred_flat
                    conf += (
                        np.bincount(indices, minlength=k * k)
                        .reshape(k, k)
                        .astype(np.int64)
                    )

        if conf is None or int(conf.size) == 0:
            mean_iou = 0.0
            mean_f1 = 0.0
        else:
            tp = np.diag(conf).astype(np.float64, copy=False)
            fp = conf.sum(axis=0).astype(np.float64, copy=False) - tp
            fn = conf.sum(axis=1).astype(np.float64, copy=False) - tp
            iou_denom = tp + fp + fn
            f1_denom = 2.0 * tp + fp + fn
            iou = np.divide(
                tp,
                iou_denom,
                out=np.full_like(tp, np.nan, dtype=np.float64),
                where=iou_denom > 0.0,
            )
            f1 = np.divide(
                2.0 * tp,
                f1_denom,
                out=np.full_like(tp, np.nan, dtype=np.float64),
                where=f1_denom > 0.0,
            )
            mean_iou = float(np.nanmean(iou)) if bool(np.any(np.isfinite(iou))) else 0.0
            mean_f1 = float(np.nanmean(f1)) if bool(np.any(np.isfinite(f1))) else 0.0

        return {
            "loss": running_loss / len(loader),
            "mIoU": mean_iou,
            "f1_score": mean_f1,
            "per_class_iou": iou.tolist() if conf is not None and int(conf.size) > 0 else [],
        }

    def extract_features(self, data_loader):
        self.model.eval()
        features_dict = {}
        features_list = []

        def hook_fn(module, input, output):
            gap = F.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
            features_list.append(gap.detach().cpu())

        layer = None
        if hasattr(self.model, "backbone"):
            layer = getattr(self.model.backbone, "layer4", None)
        if (
            layer is None
            and hasattr(self.model, "model")
            and hasattr(self.model.model, "encoder")
        ):
            layer = getattr(self.model.model.encoder, "layer4", None)
        if layer is None:
            return {}
        handle = layer.register_forward_hook(hook_fn)
        ordered_indices = self._get_ordered_indices(data_loader)
        offset = 0
        try:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Extracting Features"):
                    images, _ = self._unpack_batch(batch, require_mask=False)
                    features_list.clear()
                    images = images.to(self.device)
                    if self._amp_enabled:
                        amp_dtype = torch.float16
                        if self._amp_dtype in ("bf16", "bfloat16"):
                            amp_dtype = torch.bfloat16
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            _ = self.model(images)
                    else:
                        _ = self.model(images)
                    if not features_list:
                        continue
                    batch_features = features_list[-1]
                    batch_size = batch_features.size(0)
                    if ordered_indices is None:
                        batch_indices = list(range(offset, offset + batch_size))
                    else:
                        batch_indices = ordered_indices[offset : offset + batch_size]
                    for i, dataset_index in enumerate(batch_indices):
                        sample_id = self._resolve_sample_id(
                            data_loader.dataset, dataset_index
                        )
                        features_dict[str(sample_id)] = batch_features[i].numpy()
                    offset += batch_size
        finally:
            handle.remove()
        return features_dict

    def predict_probs(self, data_loader):
        """
        辅助函数：获取预测概率 (N, C, H, W) 和 特征 (N, D)
        """
        self.model.eval()
        probs_list = []
        features_list = []

        # Hook for features
        def hook_fn(module, input, output):
            gap = F.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
            features_list.append(gap.detach().cpu())

        handle = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
            layer = getattr(self.model.model.encoder, "layer4", None)
            if layer:
                handle = layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Predicting"):
                    images, _ = self._unpack_batch(batch, require_mask=False)

                    images = images.to(self.device)
                    if self._amp_enabled:
                        amp_dtype = torch.float16
                        if self._amp_dtype in ("bf16", "bfloat16"):
                            amp_dtype = torch.bfloat16
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            outputs = self.model(images)  # Logits
                        probs = F.softmax(outputs.float(), dim=1)
                    else:
                        outputs = self.model(images)  # Logits
                        probs = F.softmax(outputs, dim=1)
                    probs_list.append(probs.cpu())
        finally:
            if handle:
                handle.remove()

        return torch.cat(probs_list), torch.cat(
            features_list
        ) if features_list else None

    def _get_ordered_indices(self, data_loader):
        dataset = getattr(data_loader, "dataset", None)
        if dataset is None:
            return None
        if hasattr(dataset, "indices"):
            return list(dataset.indices)
        return list(range(len(dataset)))

    def _resolve_sample_id(self, dataset, dataset_index):
        base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
        if hasattr(base_dataset, "images") and dataset_index < len(base_dataset.images):
            name = base_dataset.images[dataset_index]
            return os.path.splitext(name)[0]
        return str(dataset_index)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            alpha_tensor = torch.as_tensor(
                self.alpha, device=inputs.device, dtype=inputs.dtype
            )
            if alpha_tensor.numel() == 1:
                log_pt = log_pt * alpha_tensor
            else:
                log_pt = log_pt * alpha_tensor.gather(0, targets)
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
