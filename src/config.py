
import os
import json
import torch

def _coerce_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off", ""):
            return False
    return default

def _load_llm_config(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _resolve_llm_config_path():
    env_path = os.getenv('AAL_SD_LLM_CONFIG_PATH') or os.getenv('LLM_CONFIG_PATH')
    if env_path:
        return env_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_config.json')

def _default_worker_count():
    cpu_count = os.cpu_count() or 2
    return max(1, min(4, cpu_count // 2))

def _looks_like_dataset_root(path: str | None) -> bool:
    if not path:
        return False
    if not os.path.isdir(path):
        return False
    required_dirs = (
        "TrainData/img",
        "TrainData/mask",
        "ValidData/img",
        "ValidData/mask",
    )
    for rel in required_dirs:
        if not os.path.isdir(os.path.join(path, rel)):
            return False
    return True

class Config:
    # 路径配置
    # config.py is in src/, so project root is one level up
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    _L4S_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'Landslide4Sense')
    _DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'landslide4sense')
    _CAS_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'CAS_Landslide')
    _CAS_DATA_DIR_ALT = os.path.join(ROOT_DIR, 'data', 'Data', 'CAS_Landslide')
    _HOME_AAL_SD_DIR = os.path.join(os.path.expanduser("~"), "AAL_SD")
    _HOME_L4S_DATA_DIR = os.path.join(_HOME_AAL_SD_DIR, "data", "Landslide4Sense")
    _HOME_DEFAULT_DATA_DIR = os.path.join(_HOME_AAL_SD_DIR, "data", "landslide4sense")

    _DATA_DIR_CANDIDATES = (
        _L4S_DATA_DIR,
        _DEFAULT_DATA_DIR,
        _CAS_DATA_DIR,
        _CAS_DATA_DIR_ALT,
        _HOME_L4S_DATA_DIR,
        _HOME_DEFAULT_DATA_DIR,
    )

    DATA_DIR = None
    _ENV_DATA_DIR = os.getenv("AAL_SD_DATA_DIR") or os.getenv("DATA_DIR")
    if _ENV_DATA_DIR is not None and str(_ENV_DATA_DIR).strip() != "":
        if not _looks_like_dataset_root(_ENV_DATA_DIR):
            raise RuntimeError(
                f"AAL_SD_DATA_DIR/DATA_DIR is set but not a valid dataset root: {_ENV_DATA_DIR}"
            )
        DATA_DIR = str(_ENV_DATA_DIR)
    else:
        for _candidate in _DATA_DIR_CANDIDATES:
            if _looks_like_dataset_root(_candidate):
                DATA_DIR = _candidate
                break
    if DATA_DIR is None:
        DATA_DIR = _L4S_DATA_DIR
        
    # Results and Checkpoints
    RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

    POOLS_DIR = os.path.join(RESULTS_DIR, 'pools')

    # 确保目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(POOLS_DIR, exist_ok=True)

    # 数据集配置
    # 学术合理性分析：
    # 1. 冷启动问题：语义分割是数据密集型任务，128x128 的 Patch 包含的上下文有限。
    #    过少的初始样本（如 2% ≈ 75 张）会导致特征提取器（Encoder）在 14 通道输入上无法收敛，
    #    提取的特征充满噪声，导致基于特征聚类的 KUCS 策略失效（GIGO - Garbage In Garbage Out）。
    # 2. 14通道适配：ResNet50 预训练权重是 RGB 3通道。前 3 个通道可以复用权重，
    #    但后 11 个通道的权重通常是随机初始化或通过启发式方法（如平均复制）初始化的。
    #    这部分参数需要更多的数据来"对齐"特征空间。
    # 3. Agent 稳定性：LLM Agent 依赖稳定的指标（Loss, mIoU 变化）做决策。
    #    极不稳定的初始模型会导致指标剧烈波动，引发 Agent 误判（如频繁触发 Fallback）。
    # 因此，5% (约 190 张) 是一个兼顾 "挑战性" (仍然很少) 和 "稳定性" (足以启动) 的平衡点。
    INITIAL_LABELED_SIZE = 0.05  # 初始标注比例 5%
    ESTIMATED_TOTAL_SAMPLES = 3799 # 实际训练集总大小
    NUM_CLASSES = 2
    IN_CHANNELS = 14
    TRAIN_SPLIT = "train"
    VAL_SPLIT = "val"
    TEST_SPLIT = "test"
    MODEL_SELECTION = str(os.getenv("AAL_SD_MODEL_SELECTION", "best_val") or "best_val").strip().lower()
    FAIL_ON_NONFINITE = _coerce_bool(os.getenv("AAL_SD_FAIL_ON_NONFINITE"), default=True)

    RANDOM_SEED = int(os.getenv("AAL_SD_RANDOM_SEED", "42"))
    DETERMINISTIC = _coerce_bool(os.getenv("AAL_SD_DETERMINISTIC"), default=True)
    
    # 训练配置
    BATCH_SIZE = 4
    LR = 1e-4
    _NUM_WORKERS_ENV = os.getenv("AAL_SD_NUM_WORKERS")
    if _NUM_WORKERS_ENV is None:
        NUM_WORKERS = _default_worker_count()
    else:
        try:
            NUM_WORKERS = int(_NUM_WORKERS_ENV)
        except ValueError:
            NUM_WORKERS = 0
    _FEATURE_NUM_WORKERS_ENV = os.getenv("AAL_SD_FEATURE_NUM_WORKERS")
    if _FEATURE_NUM_WORKERS_ENV is None:
        FEATURE_NUM_WORKERS = _default_worker_count()
    else:
        try:
            FEATURE_NUM_WORKERS = int(_FEATURE_NUM_WORKERS_ENV)
        except ValueError:
            FEATURE_NUM_WORKERS = 0
    EPOCHS_PER_ROUND = 10
    EPOCHS_PER_ROUND_SCHEDULE = None
    FIX_EPOCHS_PER_ROUND = True
    DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    PIN_MEMORY = _coerce_bool(os.getenv("AAL_SD_PIN_MEMORY"), default=(DEVICE == "cuda"))
    FEATURE_PIN_MEMORY = _coerce_bool(os.getenv("AAL_SD_FEATURE_PIN_MEMORY"), default=PIN_MEMORY)
    PERSISTENT_WORKERS = _coerce_bool(os.getenv("AAL_SD_PERSISTENT_WORKERS"), default=NUM_WORKERS > 0)
    FEATURE_PERSISTENT_WORKERS = _coerce_bool(os.getenv("AAL_SD_FEATURE_PERSISTENT_WORKERS"), default=FEATURE_NUM_WORKERS > 0)
    SHARING_STRATEGY = os.getenv("AAL_SD_SHARING_STRATEGY", "file_descriptor")
    try:
        PREFETCH_FACTOR = int(os.getenv("AAL_SD_PREFETCH_FACTOR", "2"))
    except ValueError:
        PREFETCH_FACTOR = 2
    try:
        FEATURE_PREFETCH_FACTOR = int(os.getenv("AAL_SD_FEATURE_PREFETCH_FACTOR", "2"))
    except ValueError:
        FEATURE_PREFETCH_FACTOR = 2

    AMP_ENABLED = _coerce_bool(os.getenv("AAL_SD_AMP"), default=False)
    AMP_DTYPE = str(os.getenv("AAL_SD_AMP_DTYPE", "float16") or "float16").strip().lower()
    TORCH_COMPILE = _coerce_bool(os.getenv("AAL_SD_TORCH_COMPILE"), default=False)
    TORCH_COMPILE_MODE = str(os.getenv("AAL_SD_TORCH_COMPILE_MODE", "default") or "default").strip()
    TF32 = _coerce_bool(os.getenv("AAL_SD_TF32"), default=False)
    CUDNN_BENCHMARK = _coerce_bool(os.getenv("AAL_SD_CUDNN_BENCHMARK"), default=False)
    try:
        TORCH_NUM_THREADS = int(os.getenv("AAL_SD_TORCH_NUM_THREADS", "0"))
    except ValueError:
        TORCH_NUM_THREADS = 0
    try:
        TORCH_NUM_INTEROP_THREADS = int(os.getenv("AAL_SD_TORCH_NUM_INTEROP_THREADS", "0"))
    except ValueError:
        TORCH_NUM_INTEROP_THREADS = 0
    GRAD_LOGGING = True
    GRAD_LOG_MAX_BATCHES = 8
    GRAD_LOG_PARAM_MAX_ELEMENTS = 200000
    GRAD_LOG_VAL_ALIGNMENT = True
    
    # 主动学习配置
    N_ROUNDS = 16
    STRATEGY = 'ad_kucs'
    FIXED_LAMBDA = 0.5
    ALPHA = 5.0
    # 预算建议设为总数据的 40%-60% 以展示完整曲线。
    # 训练集约 3800 张，15轮*100张 + 初始 ≈ 1560。
    # 自动计算总预算：基于预估总样本数和预算比例
    # 为什么选择 40% 而非 60%？
    # 1. 边际效应递减：主动学习曲线通常呈对数增长。大部分性能提升发生在 0%-40% 区间。
    #    超过 40% 后，模型往往已经接近全监督性能（95%+），此时再增加预算，
    #    AL 策略与随机采样的差距会迅速缩小，难以区分算法优劣。
    # 2. 成本效益（Cost-Benefit）：在实际工程中，标注 40% 数据的成本通常是可以接受的上限。
    #    如果需要标注 60% 才能达到目标，用户往往会选择全量标注以省去 AL 的计算开销。
    #    证明在 40% 数据量下达到具有竞争力的性能，更具学术和应用价值。
    BUDGET_RATIO = 0.4  # 40%
    TOTAL_BUDGET = int(ESTIMATED_TOTAL_SAMPLES * BUDGET_RATIO)

    # 自动计算 QUERY_SIZE：如果没有显式指定 QUERY_SIZE，则根据预算和轮数自动计算
    # 默认值 100 仅作为备用
    _QUERY_SIZE_RAW = 100
    
    @property
    def QUERY_SIZE(self):
        # 优先使用显式设置的 _QUERY_SIZE_RAW，除非它是默认值且我们有足够的参数来自动计算
        if hasattr(self, '_QUERY_SIZE_RAW') and self._QUERY_SIZE_RAW != 100:
            return self._QUERY_SIZE_RAW
            
        # 尝试自动计算：(总预算 - 初始已标注) / 轮数
        # 注意：这里我们假设初始标注量很少（例如60），如果无法获取准确初始值，则忽略
        if self.TOTAL_BUDGET and self.N_ROUNDS:
             # 估算每轮查询数：(Total - Initial_Estimate) / Rounds
             # 假设初始只有极少量样本(如2%)，这里简单用 Total / Rounds 做个上限估算，
             # 或者更精确地：(Total - Initial) / Rounds
             initial_estimate = 0
             if hasattr(self, 'INITIAL_LABELED_SIZE') and hasattr(self, 'TOTAL_BUDGET'):
                 # 估算初始数量 (基于总数据量假设为 TOTAL_BUDGET / 0.5 左右? 不，最好直接用配置)
                 # 由于这里无法访问 dataset 长度，我们只能基于 TOTAL_BUDGET 估算
                 # 假设 TOTAL_BUDGET 约占 50%，则 Total Data = 2 * TOTAL_BUDGET
                 # Initial = 0.02 * 2 * TOTAL_BUDGET = 0.04 * TOTAL_BUDGET
                 # 或者更保守地，直接忽略初始，或者减去一个固定值
                 # 更稳健的方法：
                 # (TOTAL_BUDGET - (TOTAL_BUDGET * (INITIAL_LABELED_SIZE / 0.5))) / N_ROUNDS
                 # 简化：假设初始已占用一部分预算。
                 pass
             
             # 直接修改：减去预估的初始样本数
             # TOTAL_BUDGET=1500, Initial = 3000 * 0.02 = 60. Available for AL = 1440.
             # 1440 / 15 = 96.
             initial_estimate = int(getattr(self, 'ESTIMATED_TOTAL_SAMPLES', 3000) * getattr(self, 'INITIAL_LABELED_SIZE', 0.02))
             selection_rounds = int(max(1, int(self.N_ROUNDS) - 1))
             estimated = int((self.TOTAL_BUDGET - initial_estimate) / selection_rounds)
             # 保证至少为1
             return max(1, estimated)
        return 100

    @QUERY_SIZE.setter
    def QUERY_SIZE(self, value):
        self._QUERY_SIZE_RAW = value
    
    LLM_CONFIG_PATH = _resolve_llm_config_path()
    _LLM_CONFIG = _load_llm_config(LLM_CONFIG_PATH)
    LLM_PROVIDER = _LLM_CONFIG.get('provider')
    _RAW_LLM_API_KEY = (_LLM_CONFIG.get('api_key') or "").strip()
    LLM_API_KEY = _RAW_LLM_API_KEY or os.getenv(_LLM_CONFIG.get('api_key_env', 'SILICONFLOW_API_KEY'))
    LLM_BASE_URL = _LLM_CONFIG.get('base_url')
    LLM_MODEL = _LLM_CONFIG.get('model')
    LLM_TEMPERATURE = _LLM_CONFIG.get('temperature', 0.0)
    LLM_TIMEOUT = _LLM_CONFIG.get('timeout', 60)
    LLM_MAX_RETRIES = int(_LLM_CONFIG.get('max_retries', 3))
    LLM_RETRY_BASE_SECONDS = float(_LLM_CONFIG.get('retry_base_seconds', 5.0))
    LLM_RETRY_BACKOFF = float(_LLM_CONFIG.get('retry_backoff', 2.0))
    LLM_RETRY_MAX_SECONDS = float(_LLM_CONFIG.get('retry_max_seconds', 60.0))
    STOP_ON_LLM_FAILURE = _coerce_bool(_LLM_CONFIG.get('stop_on_llm_failure', True), default=True)
    STOP_ALL_EXPERIMENTS_ON_LLM_FAILURE = _coerce_bool(_LLM_CONFIG.get('stop_all_experiments_on_llm_failure', True), default=True)
    RESUME_FROM_LOGS = _coerce_bool(_LLM_CONFIG.get('resume_from_logs', True), default=True)

    RESEARCH_MODE = True
    STRICT_RESUME = True

    STRICT_INNOVATION = _coerce_bool(_LLM_CONFIG.get('strict_innovation', True), default=True)
    FAIL_ON_AGENT_FALLBACK = _coerce_bool(_LLM_CONFIG.get('fail_on_agent_fallback', True), default=True)
    FAIL_ON_RANKING_DEGRADED = _coerce_bool(_LLM_CONFIG.get('fail_on_ranking_degraded', True), default=True)
    REQUIRE_LLM_FOR_AGENT = _coerce_bool(_LLM_CONFIG.get('require_llm_for_agent', True), default=True)
    TRACE_AGENT_THOUGHT = _coerce_bool(os.getenv("AAL_SD_TRACE_AGENT_THOUGHT"), default=False)
    TRACE_AGENT_PROMPT = _coerce_bool(os.getenv("AAL_SD_TRACE_AGENT_PROMPT"), default=False)

    # 损失函数配置
    LOSS_TYPE = 'CrossEntropyLoss' # Options: CrossEntropyLoss, FocalLoss
