from config import Config
from core.data_preprocessing import DataPreprocessor


def test_official_splits_create_empty_val_and_test_pools(tmp_path):
    root = tmp_path / "l4s"
    (root / "TrainData" / "img").mkdir(parents=True)
    (root / "TrainData" / "mask").mkdir(parents=True)
    (root / "ValidData" / "img").mkdir(parents=True)
    (root / "ValidData" / "mask").mkdir(parents=True)

    import h5py
    import numpy as np

    for i in range(20):
        img = np.zeros((128, 128, 14), dtype=np.float32)
        mask = np.zeros((128, 128), dtype=np.uint8)
        if i % 2 == 0:
            mask[0, 0] = 1
        with h5py.File(root / "TrainData" / "img" / f"sample_{i}.h5", "w") as f:
            f.create_dataset("img", data=img)
        with h5py.File(root / "TrainData" / "mask" / f"sample_{i}.h5", "w") as f:
            f.create_dataset("mask", data=mask)

    cfg = Config()
    cfg.DATA_DIR = str(root)
    cfg.POOLS_DIR = str(tmp_path / "pools")
    cfg.INITIAL_LABELED_SIZE = 0.2
    cfg.RESEARCH_MODE = True

    pre = DataPreprocessor(cfg, experiment_name="exp")
    pre.create_data_pools(force=True)

    import pandas as pd

    pools_dir = tmp_path / "pools" / "exp"
    labeled = pd.read_csv(pools_dir / "labeled_pool.csv")
    unlabeled = pd.read_csv(pools_dir / "unlabeled_pool.csv")
    assert not (pools_dir / "val_pool.csv").exists()
    assert not (pools_dir / "test_pool.csv").exists()
    assert len(labeled) + len(unlabeled) == 20


def test_test_split_returns_mask_when_available(tmp_path):
    root = tmp_path / "l4s_h5"
    (root / "TestData" / "img").mkdir(parents=True)
    (root / "TestData" / "mask").mkdir(parents=True)

    import numpy as np

    img = np.zeros((128, 128, 14), dtype=np.float32)
    mask = np.zeros((128, 128), dtype=np.uint8)

    (root / "TestData" / "img" / "sample_0.h5").write_bytes(b"")
    (root / "TestData" / "mask" / "sample_0.h5").write_bytes(b"")

    from core import dataset as dataset_module
    from core.dataset import Landslide4SenseDataset

    class _FakeDataset:
        def __init__(self, arr):
            self._arr = arr

        def __getitem__(self, _):
            return self._arr

    class _FakeH5File:
        def __init__(self, data):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __contains__(self, key):
            return key in self._data

        def keys(self):
            return self._data.keys()

        def __getitem__(self, key):
            return self._data[key]

    def _fake_h5py_file(_path, _mode):
        return _FakeH5File({"img": _FakeDataset(img), "mask": _FakeDataset(mask)})

    original_file = dataset_module.h5py.File
    dataset_module.h5py.File = _fake_h5py_file

    try:
        ds = Landslide4SenseDataset(str(root), split="test")
        item = ds[0]
        assert isinstance(item, dict)
        assert int(item["mask"].numel()) > 0
    finally:
        dataset_module.h5py.File = original_file


def test_pipeline_works_with_empty_val_and_test_pools(tmp_path):
    root = tmp_path / "l4s_h5"
    (root / "TrainData" / "img").mkdir(parents=True)
    (root / "TrainData" / "mask").mkdir(parents=True)
    (root / "ValidData" / "img").mkdir(parents=True)
    (root / "ValidData" / "mask").mkdir(parents=True)

    import h5py
    import numpy as np

    for i in range(30):
        img = np.zeros((128, 128, 14), dtype=np.float32)
        mask = np.zeros((128, 128), dtype=np.uint8)
        if i % 2 == 0:
            mask[0, 0] = 1
        with h5py.File(root / "TrainData" / "img" / f"sample_{i}.h5", "w") as f:
            f.create_dataset("img", data=img)
        with h5py.File(root / "TrainData" / "mask" / f"sample_{i}.h5", "w") as f:
            f.create_dataset("mask", data=mask)

    for i in range(2):
        img = np.zeros((128, 128, 14), dtype=np.float32)
        mask = np.zeros((128, 128), dtype=np.uint8)
        with h5py.File(root / "ValidData" / "img" / f"val_{i}.h5", "w") as f:
            f.create_dataset("img", data=img)
        with h5py.File(root / "ValidData" / "mask" / f"val_{i}.h5", "w") as f:
            f.create_dataset("mask", data=mask)

    cfg = Config()
    cfg.DATA_DIR = str(root)
    cfg.POOLS_DIR = str(tmp_path / "pools")
    cfg.RESULTS_DIR = str(tmp_path / "results")
    cfg.CHECKPOINT_DIR = str(tmp_path / "checkpoints")
    cfg.INITIAL_LABELED_SIZE = 0.2
    cfg.RESEARCH_MODE = True
    cfg.RANDOM_SEED = 123
    cfg.DETERMINISTIC = True
    cfg.ALLOW_LEGACY_POOLS = False
    cfg.START_MODE = "fresh"

    from main import ActiveLearningPipeline
    import pandas as pd

    p1 = ActiveLearningPipeline(cfg, "baseline_random", run_id="r1")
    assert not hasattr(p1, "val_indices")
    assert not hasattr(p1, "test_indices")

    pools_dir = tmp_path / "pools" / "r1" / "baseline_random"
    assert not (pools_dir / "val_pool.csv").exists()
    assert not (pools_dir / "test_pool.csv").exists()
