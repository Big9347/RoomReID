from typing import List,  Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json, yaml
from mtmc_reid.configs.app_config import AppConfig
import os
@dataclass
class DirectionConfig:
    """Configuration for a specific mode."""
    roi1: List[List[int]]
    roi2: List[List[int]]
    background_image: str
    output_path: str
@dataclass
class ROIConfig:
    """Configuration for ROI analysis with mode-specific settings."""
    base_path: str
    db_path: str
    collection_name: str
    directions: List[str]  # ["Enter", "Exit"]
    enter_config: DirectionConfig
    exit_config: DirectionConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], app_config: AppConfig) -> 'ROIConfig':
        """Create ROIConfig from dictionary."""

        # Extract mode-specific configurations
        enter_config = DirectionConfig(
            roi1=config_dict['enter_config']['roi1'],
            roi2=config_dict['enter_config']['roi2'],
            background_image= os.path.join(app_config.BASE_PATH,"backgroundEnter.png"),
            output_path=os.path.join(app_config.BASE_PATH,"roi_transitions_enter.txt")
        )
        
        exit_config = DirectionConfig(
            roi1=config_dict['exit_config']['roi1'],
            roi2=config_dict['exit_config']['roi2'],
             background_image= os.path.join(app_config.BASE_PATH,"backgroundExit.png"),
            output_path=os.path.join(app_config.BASE_PATH,"roi_transitions_exit.txt")
        )
        
        return cls(
            base_path=app_config.BASE_PATH,
            db_path=app_config.DEFAULT_SQL_DB,
            collection_name=app_config.COLLECTION_NAME,
            directions=app_config.ROI_MODES,
            enter_config=enter_config,
            exit_config=exit_config
        )
    
    @classmethod
    def load_from_file(cls, app_config: AppConfig) -> 'ROIConfig':
        """Load configuration from JSON or YAML file."""

        config_path = Path(app_config.ROI_CONFIG_PATH)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(config_dict, app_config)
    
    def get_direction_config(self, direction: str) -> DirectionConfig:
        """Get configuration for a specific direction."""
        if direction == "Enter":
            return self.enter_config
        elif direction == "Exit":
            return self.exit_config
        else:
            raise ValueError(f"Unknown direction: {direction}")

