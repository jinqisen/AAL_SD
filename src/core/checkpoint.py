
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from utils import atomic_write_json, read_json_dict

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
            atomic_write_json(self.checkpoint_path, state_dict, indent=4)
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
            state_dict = read_json_dict(self.checkpoint_path)
            logger.info(f"Checkpoint loaded from {self.checkpoint_path}")
            return state_dict
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e
