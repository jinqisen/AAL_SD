
import unittest
import shutil
import os
import json
from src.core.checkpoint import CheckpointManager

class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_checkpoints"
        self.exp_name = "test_exp"
        os.makedirs(self.test_dir, exist_ok=True)
        self.manager = CheckpointManager(self.test_dir, self.exp_name)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_save_and_load(self):
        state = {
            'round': 1,
            'performance_history': [{'mIoU': 0.5, 'round': 1}],
            'budget_history': [100],
            'fallback_history': []
        }
        
        # Save
        success = self.manager.save(state)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.manager.checkpoint_path))
        
        # Load
        loaded = self.manager.load()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['round'], 1)
        self.assertEqual(loaded['experiment_name'], self.exp_name)
        self.assertEqual(loaded['performance_history'][0]['mIoU'], 0.5)
        
    def test_load_non_existent(self):
        loaded = self.manager.load()
        self.assertIsNone(loaded)

if __name__ == '__main__':
    unittest.main()
