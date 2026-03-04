
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from .dataset import Landslide4SenseDataset

class DataPreprocessor:
    """
    数据预处理器，负责执行数据集的划分。
    """

    def __init__(self, config, experiment_name=None):
        """
        初始化数据预处理器。
        :param config: 配置对象，包含DATA_DIR, POOLS_DIR, INITIAL_LABELED_SIZE, TEST_SIZE。
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
        执行完整的数据划分流程，创建初始标注集、未标注池和固定测试集。
        1. 扫描原始数据目录，获取所有图像和掩码的路径。
        2. 按照配置的比例进行划分。
        3. 将划分结果保存为三个CSV文件：`labeled_pool.csv`, `unlabeled_pool.csv`, `test_pool.csv`。
        """
        labeled_path = os.path.join(self.pools_dir, 'labeled_pool.csv')
        unlabeled_path = os.path.join(self.pools_dir, 'unlabeled_pool.csv')
        test_path = os.path.join(self.pools_dir, 'test_pool.csv')
        if (not force) and os.path.exists(labeled_path) and os.path.exists(unlabeled_path) and os.path.exists(test_path):
            print(f"Data pools already exist for {self.experiment_name or 'default'}. Skipping creation.")
            return

        image_dir = os.path.join(self.raw_data_dir, 'images')
        mask_dir = os.path.join(self.raw_data_dir, 'masks')
        if not os.path.exists(image_dir):
            image_dir = os.path.join(self.raw_data_dir, 'TrainData', 'img')
        if not os.path.exists(mask_dir):
            mask_dir = os.path.join(self.raw_data_dir, 'TrainData', 'mask')
        
        image_files = []
        valid_ext = ('.h5', '.png', '.tif', '.tiff', '.jpg', '.jpeg')
        if os.path.exists(image_dir):
            names = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)]
            names.sort()
            image_files.extend([os.path.join(image_dir, f) for f in names])
            
        if not image_files:
            if getattr(self.config, "RESEARCH_MODE", False):
                raise RuntimeError("No image files found in dataset directory")
            print("No image files found in dataset directory")
            return

        data = []
        for img_path in image_files:
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            mask_filename = filename
            if filename.startswith('image_'):
                mask_filename = filename.replace('image_', 'mask_')
            
            mask_path = os.path.join(mask_dir, mask_filename)
            if os.path.exists(mask_path):
                data.append({
                    'sample_id': name_without_ext,
                    'image_path': img_path,
                    'mask_path': mask_path
                })
        
        df = pd.DataFrame(data)
        if df.empty:
            if getattr(self.config, "RESEARCH_MODE", False):
                raise RuntimeError("No valid image/mask pairs found in dataset directory")
            print("No valid image/mask pairs found in dataset directory")
            return

        seed = int(getattr(self.config, "RANDOM_SEED", 42) or 42)
        eval_split = str(getattr(self.config, "EVAL_SPLIT", "internal") or "internal").strip().lower()
        external_eval = eval_split in ("val", "valid", "official_val", "l4s_val")

        if external_eval:
            train_val_df = df
            test_df = df.iloc[:0].copy()
        else:
            train_val_df, test_df = train_test_split(
                df, test_size=self.config.TEST_SIZE, random_state=seed
            )
        
        # 划分初始标注集和未标注池
        labeled_df, unlabeled_df = train_test_split(
            train_val_df, train_size=self.config.INITIAL_LABELED_SIZE, random_state=seed
        )

        # 保存文件
        os.makedirs(self.pools_dir, exist_ok=True)
        suffix = f".{os.getpid()}.{int(time.time() * 1_000_000)}"
        labeled_tmp = labeled_path + suffix + ".tmp"
        unlabeled_tmp = unlabeled_path + suffix + ".tmp"
        test_tmp = test_path + suffix + ".tmp"
        labeled_df.to_csv(labeled_tmp, index=False)
        unlabeled_df.to_csv(unlabeled_tmp, index=False)
        test_df.to_csv(test_tmp, index=False)
        os.replace(labeled_tmp, labeled_path)
        os.replace(unlabeled_tmp, unlabeled_path)
        os.replace(test_tmp, test_path)

        print(f"Created data pools:")
        print(f"- Labeled pool: {len(labeled_df)} samples")
        print(f"- Unlabeled pool: {len(unlabeled_df)} samples")
        print(f"- Test pool: {len(test_df)} samples")
