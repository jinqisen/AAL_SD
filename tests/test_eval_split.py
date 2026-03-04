from config import Config
from core.data_preprocessing import DataPreprocessor


def test_eval_split_val_skips_internal_test_split(tmp_path, monkeypatch):
    root = tmp_path / "l4s"
    (root / "TrainData" / "img").mkdir(parents=True)
    (root / "TrainData" / "mask").mkdir(parents=True)

    for i in range(20):
        (root / "TrainData" / "img" / f"image_{i}.png").write_bytes(b"img")
        (root / "TrainData" / "mask" / f"mask_{i}.png").write_bytes(b"mask")

    cfg = Config()
    cfg.DATA_DIR = str(root)
    cfg.POOLS_DIR = str(tmp_path / "pools")
    cfg.INITIAL_LABELED_SIZE = 0.2
    cfg.TEST_SIZE = 0.5
    cfg.EVAL_SPLIT = "val"
    cfg.RESEARCH_MODE = True

    pre = DataPreprocessor(cfg, experiment_name="exp")
    pre.create_data_pools(force=True)

    import pandas as pd

    pools_dir = tmp_path / "pools" / "exp"
    labeled = pd.read_csv(pools_dir / "labeled_pool.csv")
    unlabeled = pd.read_csv(pools_dir / "unlabeled_pool.csv")
    test = pd.read_csv(pools_dir / "test_pool.csv")

    assert len(test) == 0
    assert len(labeled) + len(unlabeled) == 20
