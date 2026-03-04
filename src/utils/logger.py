
import logging
import os
import sys

def setup_logger(name="AAL_SD", log_file=None, level=logging.INFO):
    """
    配置全局 Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplicates
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - pid=%(process)d - tid=%(threadName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

logger = setup_logger()
