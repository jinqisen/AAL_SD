import pytest
import os
import json
from datetime import datetime
from unittest.mock import MagicMock, patch, mock_open
from monitor_and_recover import TrainingMonitor

@pytest.fixture
def monitor():
    with patch('monitor_and_recover.Config'):
        with patch('monitor_and_recover.SiliconFlowClient') as MockClient:
            mon = TrainingMonitor()
            mon.llm_client = MockClient.return_value
            return mon

@pytest.fixture
def mock_log_content():
    return """
[2024-01-01 10:00:00] INFO: Starting...
## Round 1
Epoch 1: Loss=0.5, mIoU=0.4
Epoch 2: Loss=0.4, mIoU=0.5
Round=1, Labeled=100, mIoU=0.5, F1=0.6
## Round 2
Epoch 1: Loss=0.3, mIoU=0.55
Epoch 2: Loss=0.2, mIoU=0.6
Round=2, Labeled=200, mIoU=0.6, F1=0.7
"""

def test_parse_experiment_history(monitor, mock_log_content):
    with patch("builtins.open", mock_open(read_data=mock_log_content)):
        with patch("os.path.getmtime", return_value=datetime.now().timestamp()):
            history = monitor.parse_experiment_history("dummy.md")
            
            assert len(history["rounds"]) == 2
            assert len(history["epochs"]) == 4
            
            assert history["rounds"][0]["round"] == 1
            assert history["rounds"][0]["mIoU"] == 0.5
            
            assert history["epochs"][-1]["loss"] == 0.2
            assert history["epochs"][-1]["mIoU"] == 0.6

def test_detect_anomalies_loss_drop(monitor):
    history = {
        "epochs": [
            {"round": 1, "epoch": 1, "loss": 0.8, "mIoU": 0.5},
            {"round": 1, "epoch": 2, "loss": 0.05, "mIoU": 0.5}  # Drop > 90%
        ],
        "rounds": []
    }
    anomalies = monitor.detect_anomalies("test_exp", history)
    assert any("Loss dropped suspiciously fast" in a for a in anomalies)

def test_detect_anomalies_stagnation(monitor):
    history = {
        "epochs": [{"round": 1, "epoch": 1, "loss": 0.5, "mIoU": 0.5}], # Dummy epoch to pass check
        "rounds": [
            {"round": 1, "mIoU": 0.500, "labeled": 100, "F1": 0.5},
            {"round": 2, "mIoU": 0.501, "labeled": 200, "F1": 0.5},
            {"round": 3, "mIoU": 0.501, "labeled": 300, "F1": 0.5},
            {"round": 4, "mIoU": 0.501, "labeled": 400, "F1": 0.5}
        ]
    }
    anomalies = monitor.detect_anomalies("test_exp", history)
    assert any("Performance stagnation" in a for a in anomalies)

def test_generate_stage_report(monitor):
    history = {
        "rounds": [
            {"round": 1, "mIoU": 0.5, "labeled": 100, "F1": 0.5},
            {"round": 2, "mIoU": 0.6, "labeled": 200, "F1": 0.6}
        ],
        "epochs": [],
        "status": "running",
        "last_update": datetime.now()
    }
    
    with patch("builtins.open", mock_open()) as mock_file:
        monitor.generate_stage_report("test_exp", history)
        
        # Check if file was opened for writing
        mock_file.assert_called()
        handle = mock_file()
        
        # Check content
        writes = "".join(call.args[0] for call in handle.write.call_args_list)
        # Relaxed checks
        assert "Stage Report: test_exp - Round 2" in writes
        assert "0.6000" in writes
        assert "IMPROVED" in writes
        assert "+0.1000" in writes

def test_generate_llm_report(monitor):
    history = {
        "status": "running",
        "last_update": str(datetime.now()),
        "epochs": [],
        "rounds": []
    }
    anomalies = ["Something wrong"]
    
    monitor.llm_client.chat.return_value = "Analysis Report Content"
    
    report = monitor.generate_llm_report("test_exp", history, anomalies)
    
    assert report == "Analysis Report Content"
    monitor.llm_client.chat.assert_called_once()
