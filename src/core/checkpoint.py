
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, checkpoint_dir, experiment_name):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_state.json")
        
    def save(self, state_dict):
        """
        Save the experiment state to a JSON file.
        
        Args:
            state_dict (dict): Dictionary containing state information.
                               Expected keys: round, labeled_indices, performance_history, budget_history, fallback_history, etc.
        """
        try:
            state_dict['timestamp'] = datetime.now().isoformat()
            state_dict['experiment_name'] = self.experiment_name
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            temp_path = self.checkpoint_path + ".tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, indent=4, ensure_ascii=False)
            os.rename(temp_path, self.checkpoint_path)
            logger.info(f"Checkpoint saved to {self.checkpoint_path}")
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint: {e}") from e

    def load(self):
        """
        Load the experiment state from JSON file.
        
        Returns:
            dict or None: The state dictionary if successful, None otherwise.
        """
        if not os.path.exists(self.checkpoint_path):
            logger.info(f"No checkpoint found at {self.checkpoint_path}")
            return None
            
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                state_dict = json.load(f)
            logger.info(f"Checkpoint loaded from {self.checkpoint_path}")
            return state_dict
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e
