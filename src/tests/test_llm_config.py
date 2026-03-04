import os
import json
import importlib
import tempfile
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_llm_api_key_from_private_config():
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, 'llm_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                "provider": "siliconflow",
                "api_key": "sk-test-key",
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "Pro/deepseek-ai/DeepSeek-V3.2",
                "temperature": 0.0,
                "timeout": 60
            }, f)
        original = os.environ.get('AAL_SD_LLM_CONFIG_PATH')
        os.environ['AAL_SD_LLM_CONFIG_PATH'] = config_path
        import config as config_module
        importlib.reload(config_module)
        assert config_module.Config.LLM_API_KEY == "sk-test-key"
        if original is None:
            del os.environ['AAL_SD_LLM_CONFIG_PATH']
        else:
            os.environ['AAL_SD_LLM_CONFIG_PATH'] = original
        importlib.reload(config_module)
