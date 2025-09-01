# 3D Stroke Segmentation Pipeline - Refactored Production Version

from __future__ import annotations
import os, sys, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import numpy as np, nibabel as nib
from pathlib import Path
import json, time, logging, random, warnings, shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="Progress", **kwargs):
        return iterable
from scipy import ndimage
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.float32)

# CONSTANTS - Extracted magic numbers for better maintainability
class Constants:
    # Tversky Loss Parameters
    TVERSKY_ALPHA = 0.25  # False positive penalty
    TVERSKY_BETA = 0.85   # False negative penalty (higher for small lesions)
    TVERSKY_GAMMA = 2.0   # Focal parameter
    
    # Training Thresholds
    POSITIVE_RATIO = 0.7
    
    # Interpolation Orders
    IMAGE_INTERPOLATION_ORDER = 3  # 3rd order spline
    MASK_INTERPOLATION_ORDER = 1   # Linear for one-hot
    NEAREST_INTERPOLATION_ORDER = 0  # Nearest neighbor fallback

try:
    from google.colab import drive
    drive.mount('/content/drive')
    COLAB_ENV = True
except ImportError:
    COLAB_ENV = False

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    logger.warning("Using CPU - training will be slow")

class Config:
    """Configuration class for the stroke segmentation pipeline"""
    def __init__(self):
        self.is_colab = COLAB_ENV
        # Load folder config
        if self.is_colab:
            config_path = '/content/drive/MyDrive/stroke_segmentation_3d/folders_config.json'
        else:
            config_path = 'folders_config.json'
            
        try:
            with open(config_path, 'r') as f:
                self.folder_names = json.load(f)
        except FileNotFoundError:
            # Default folder names if config not found
            self.folder_names = {
                'preprocessed': 'preprocessed_universal',
                'aligned': 'aligned_multimodal',
                'slices': 'dl_focus_slices_2_multimodal',
                'labels': 'dl_focus_labels_2_multimodal',
                'models': 'saved_models_3d',
                'logs': 'training_logs_3d',
                'checkpoints': 'checkpoints_3d'
            }
            
        self._setup_paths()
        self._setup_training()
        self._setup_preprocessing_config()
        
    def _setup_paths(self):
        # Auto-detect base path
        if self.is_colab:
            possible_bases = [Path("/content/drive/MyDrive/stroke_segmentation_3d")]
        else:
            possible_bases = [
                Path.cwd(),
                Path("/mnt/e/FINIAL PROJECT"), 
                Path("/mnt/d/FINIAL PROJECT"),
                Path("/mnt/c/FINIAL PROJECT"),
                Path.home() / "FINIAL PROJECT",
                Path.home() / "stroke_segmentation"
            ]
        
        self.BASE_PATH = next((p for p in possible_bases if p.exists()), Path.cwd())
        
        # Auto-detect ISLES-2022 structure (handle nested folders)
        self.ISLES_PATH = self._find_isles_dataset()
        
        self.OUTPUT_DIRS = {
            'preprocessed_universal': self.BASE_PATH / self.folder_names['preprocessed'],
            'aligned_multimodal': self.BASE_PATH / self.folder_names['aligned'],
            'dl_focus_slices_2_multimodal': self.BASE_PATH / self.folder_names['slices'],
            'dl_focus_labels_2_multimodal': self.BASE_PATH / self.folder_names['labels'], 
            'saved_models_3d': self.BASE_PATH / self.folder_names['models'],
            'training_logs_3d': self.BASE_PATH / self.folder_names['logs'],
            'checkpoints_3d': self.BASE_PATH / self.folder_names['checkpoints']
        }
        
        # Create directories
        for dir_path in self.OUTPUT_DIRS.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _find_isles_dataset(self):
        """Find ISLES-2022 dataset path"""
        possible_isles_paths = [
            self.BASE_PATH / "ISLES-2022",
            self.BASE_PATH / "ISLES-2022" / "ISLES-2022",
            self.BASE_PATH / "dataset" / "ISLES-2022",
        ]
        
        for path in possible_isles_paths:
            if path.exists() and any(path.glob("sub-strokecase*")):
                logger.info(f"Found ISLES dataset at: {path}")
                return path
        
        logger.warning("ISLES dataset not found - using default path")
        return self.BASE_PATH / "ISLES-2022"
    
    def _setup_training(self):
        """Setup training configuration"""
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 2
        self.NUM_EPOCHS = 100
        self.PATIENCE = 15
        self.MIN_LR = 1e-7
        self.WEIGHT_DECAY = 1e-4
        
    def _setup_preprocessing_config(self):
        """Setup preprocessing configuration"""
        self.TARGET_SPACING = (1.5, 1.5, 1.5)
        self.PATCH_SIZE = (128, 128, 64)
        self.OVERLAP_RATIO = 0.25
        
    def get_path(self, key: str) -> Path:
        """Get path for a specific directory"""
        if key in self.OUTPUT_DIRS:
            return self.OUTPUT_DIRS[key]
        elif key == 'base':
            return self.BASE_PATH
        elif key == 'isles':
            return self.ISLES_PATH
        else:
            raise KeyError(f"Unknown path key: {key}")

# Initialize global configuration
config = Config()

# Rest of the implementation would continue here...
# This is a truncated version to fit the GitHub file upload

def main():
    """Main execution function"""
    logger.info("3D Stroke Segmentation Pipeline")
    logger.info(f"Base path: {config.BASE_PATH}")
    logger.info(f"ISLES path: {config.ISLES_PATH}")
    logger.info(f"Device: {device}")
    
    # Pipeline stages would be implemented here
    logger.info("Pipeline initialized successfully!")

if __name__ == "__main__":
    main()