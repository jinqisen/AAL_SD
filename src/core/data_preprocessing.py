
import os
import json
import hashlib
import time
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import h5py
import numpy as np

class DataPreprocessor:
    """
    数据预处理器，负责执行数据集的划分。
    """

    def __init__(self, config, experiment_name=None):
        """
        初始化数据预处理器。
        :param config: 配置对象，包含DATA_DIR, POOLS_DIR, INITIAL_LABELED_SIZE 等。
        :param experiment_name: 实验名称，用于创建实验特定的数据池目录。
        """
        self.config = config
        self.raw_data_dir = config.DATA_DIR
        self.experiment_name = experiment_name
        
        if experiment_name:
            self.pools_dir = os.path.join(config.POOLS_DIR, experiment_name)
        else:
            self.pools_dir = config.POOLS_DIR

    def create_data_pools(self, force=False):
        """
        执行数据池生成流程，创建初始标注集与未标注池。
        1. 扫描原始数据目录，获取所有图像和掩码的路径。
        2. 按照配置的比例进行 labeled/unlabeled 划分。
        3. 将划分结果保存为两个CSV文件：`labeled_pool.csv`, `unlabeled_pool.csv`。
        """
        labeled_path = os.path.join(self.pools_dir, 'labeled_pool.csv')
        unlabeled_path = os.path.join(self.pools_dir, 'unlabeled_pool.csv')
        manifest_path = os.path.join(self.pools_dir, "pools_manifest.json")
        if (not force) and os.path.exists(labeled_path) and os.path.exists(unlabeled_path):
            if not os.path.exists(manifest_path):
                raise RuntimeError(
                    f"Pool manifest is missing in {self.pools_dir}. "
                    f"Delete existing pools or rerun with force=True."
                )
            print(f"Data pools already exist for {self.experiment_name or 'default'}. Skipping creation.")
            return
        if (not force) and (os.path.exists(labeled_path) != os.path.exists(unlabeled_path)):
            raise RuntimeError(
                f"Partial pool files detected in {self.pools_dir}. "
                f"Delete them or rerun with force=True."
            )

        image_dir = os.path.join(self.raw_data_dir, 'TrainData', 'img')
        mask_dir = os.path.join(self.raw_data_dir, 'TrainData', 'mask')
        if not os.path.isdir(image_dir):
            raise RuntimeError(f"Missing required directory: {image_dir}")
        if not os.path.isdir(mask_dir):
            raise RuntimeError(f"Missing required directory: {mask_dir}")
        val_image_dir = os.path.join(self.raw_data_dir, "ValidData", "img")
        val_mask_dir = os.path.join(self.raw_data_dir, "ValidData", "mask")
        if not os.path.isdir(val_image_dir):
            raise RuntimeError(f"Missing required directory: {val_image_dir}")
        if not os.path.isdir(val_mask_dir):
            raise RuntimeError(f"Missing required directory: {val_mask_dir}")
        
        image_files = []
        valid_ext = ('.h5',)
        names = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)]
        names.sort()
        image_files.extend([os.path.join(image_dir, f) for f in names])
            
        if not image_files:
            raise RuntimeError(f"No training image files found in: {image_dir}")

        data = []
        missing_masks = []
        for img_path in image_files:
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            mask_filename = filename
            if filename.startswith('image_'):
                mask_filename = filename.replace('image_', 'mask_')
            
            mask_path = os.path.join(mask_dir, mask_filename)
            if os.path.exists(mask_path):
                has_positive = None
                try:
                    with h5py.File(mask_path, "r") as f:
                        if "mask" not in f:
                            raise RuntimeError(
                                f"Missing dataset key 'mask' in {mask_path} (keys={list(f.keys())})"
                            )
                        mask = f["mask"][()]
                    has_positive = bool(np.any(np.asarray(mask) > 0))
                except Exception as e:
                    raise RuntimeError(f"Failed to read mask for stratify: {mask_path} ({e})") from e
                data.append({
                    'sample_id': name_without_ext,
                    'image_path': img_path,
                    'mask_path': mask_path,
                    "has_positive": bool(has_positive),
                })
            else:
                missing_masks.append({"image_path": img_path, "mask_path": mask_path})
        
        df = pd.DataFrame(data)
        if missing_masks:
            raise RuntimeError(
                f"Missing masks for {len(missing_masks)} images (example={missing_masks[0]})"
            )
        if df.empty:
            raise RuntimeError("No valid image/mask pairs found in dataset directory")

        seed = int(getattr(self.config, "RANDOM_SEED", 42) or 42)
        
        train_df = df

        pos_count = int(train_df["has_positive"].sum())
        neg_count = int(len(train_df) - pos_count)
        if pos_count <= 0 or neg_count <= 0:
            raise RuntimeError(
                f"Stratified split requires both positive and negative samples; "
                f"got pos={pos_count} neg={neg_count} in {image_dir}"
            )
        train_size = float(self.config.INITIAL_LABELED_SIZE)
        n_total = int(len(train_df))
        n_labeled = int(np.floor(float(n_total) * float(train_size))) if train_size < 1.0 else int(train_size)
        n_unlabeled = int(n_total - n_labeled)
        if n_labeled <= 0 or n_unlabeled <= 0:
            raise RuntimeError(
                f"Invalid initial labeled size: train_size={train_size} "
                f"n_total={n_total} n_labeled={n_labeled} n_unlabeled={n_unlabeled}"
            )
        if n_labeled < 2 or n_unlabeled < 2:
            raise RuntimeError(
                f"Stratified split requires at least 2 samples per split; "
                f"n_labeled={n_labeled} n_unlabeled={n_unlabeled}"
            )

        labeled_df, unlabeled_df = train_test_split(
            train_df,
            train_size=self.config.INITIAL_LABELED_SIZE,
            random_state=seed,
            stratify=train_df["has_positive"],
        )

        # 保存文件
        os.makedirs(self.pools_dir, exist_ok=True)
        suffix = f".{os.getpid()}.{int(time.time() * 1_000_000)}"
        labeled_tmp = labeled_path + suffix + ".tmp"
        unlabeled_tmp = unlabeled_path + suffix + ".tmp"
        labeled_df.to_csv(labeled_tmp, index=False)
        unlabeled_df.to_csv(unlabeled_tmp, index=False)
        os.replace(labeled_tmp, labeled_path)
        os.replace(unlabeled_tmp, unlabeled_path)

        def _dir_fingerprint(img_dir_path: str, mask_dir_path: str | None) -> dict:
            h = hashlib.sha256()
            img_names = sorted(
                [f for f in os.listdir(img_dir_path) if str(f).lower().endswith(".h5")]
            )
            for name in img_names:
                h.update(str(name).encode("utf-8", errors="ignore"))
                h.update(b"\n")
            img_sha = h.hexdigest()
            mh = hashlib.sha256()
            mask_names = []
            if mask_dir_path is not None:
                mask_names = sorted(
                    [f for f in os.listdir(mask_dir_path) if str(f).lower().endswith(".h5")]
                )
                for name in mask_names:
                    mh.update(str(name).encode("utf-8", errors="ignore"))
                    mh.update(b"\n")
            return {
                "images": {"count": int(len(img_names)), "sha256": str(img_sha)},
                "masks": {
                    "count": int(len(mask_names)),
                    "sha256": str(mh.hexdigest()),
                }
                if mask_dir_path is not None
                else None,
            }

        test_image_dir = os.path.join(self.raw_data_dir, "TestData", "img")
        test_mask_dir = os.path.join(self.raw_data_dir, "TestData", "mask")
        test_present = os.path.isdir(test_image_dir) and os.path.isdir(test_mask_dir)

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now().isoformat(),
            "data_root": str(self.raw_data_dir),
            "splits": {
                "train": _dir_fingerprint(image_dir, mask_dir),
                "val": _dir_fingerprint(val_image_dir, val_mask_dir),
                "test": _dir_fingerprint(test_image_dir, test_mask_dir) if test_present else None,
            },
            "split_policy": {
                "name": "initial_labeled_stratified_by_has_positive",
                "initial_labeled_size": float(self.config.INITIAL_LABELED_SIZE),
                "random_seed": int(seed),
                "stratify_key": "has_positive",
                "train_counts": {"pos": int(pos_count), "neg": int(neg_count), "total": int(n_total)},
            },
            "pools": {
                "labeled": int(len(labeled_df)),
                "unlabeled": int(len(unlabeled_df)),
                "files": {
                    "labeled_pool": os.path.basename(labeled_path),
                    "unlabeled_pool": os.path.basename(unlabeled_path),
                },
            },
        }
        manifest_tmp = manifest_path + suffix + ".tmp"
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        os.replace(manifest_tmp, manifest_path)

        print(f"Created data pools:")
        print(f"- Labeled pool: {len(labeled_df)} samples")
        print(f"- Unlabeled pool: {len(unlabeled_df)} samples")
