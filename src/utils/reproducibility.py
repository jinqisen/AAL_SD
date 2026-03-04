import os
import random
import numpy as np
import torch

def set_global_seed(seed: int, deterministic: bool = True):
    """
    统一设置全局随机种子 (Python, NumPy, PyTorch, CUDA)
    
    Args:
        seed: 随机种子数值
        deterministic: 是否强制使用确定性算法 (影响 cudnn 和 torch 算子)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # 强制使用确定性算法，可能会牺牲少量性能，但保证可复现性
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # 设置环境变量以确保某些库的行为一致
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 某些 PyTorch 操作需要此设置才能在 deterministic 模式下运行
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def worker_init_fn(worker_id: int):
    """
    DataLoader worker 初始化函数，确保多进程加载时的数据确定性。
    每个 worker 的种子基于全局种子和 worker_id 生成。
    """
    try:
        strategy = os.getenv("AAL_SD_SHARING_STRATEGY", "file_descriptor")
        torch.multiprocessing.set_sharing_strategy(strategy)
    except Exception:
        pass
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # 使用 worker_info.seed，它是由 DataLoader 的 generator 生成的
        seed = worker_info.seed % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        # Fallback (通常不应发生)
        import time
        seed = int(time.time()) + worker_id
        random.seed(seed)
        np.random.seed(seed)
