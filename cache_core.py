import os
import json
import hashlib
import time
import shutil
from typing import Dict, Any, List, Tuple, Optional, Union

class CacheCore:
    """
    Core cache management functionality.
    Provides base methods for cache operations.
    """
    
    def __init__(self, cache_dir=None, max_cache_age_days=30):
        """
        Initialize the cache manager.
        
        Parameters:
        -----------
        cache_dir : str, optional
            Directory to store cache files. If None, defaults to 'TEMP' in the current directory.
        max_cache_age_days : int, optional
            Maximum age of cache files in days before they're considered stale.
        """
        if cache_dir is None:
            # Default to a 'TEMP' directory in the current working directory
            self.cache_dir = os.path.join(os.getcwd(), 'TEMP')
        else:
            self.cache_dir = cache_dir
            
        self.max_cache_age_days = max_cache_age_days
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Create subdirectories for different types of cache
        self.stabilized_cache_dir = os.path.join(self.cache_dir, 'stabilized')
        self.analysis_cache_dir = os.path.join(self.cache_dir, 'analysis')
        
        for dir_path in [self.stabilized_cache_dir, self.analysis_cache_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
        # Map to store cache information in memory
        self.cache_info = self._load_cache_info()
    
    def _load_cache_info(self) -> Dict[str, Any]:
        """Load cache information from disk, or create if it doesn't exist."""
        cache_info_path = os.path.join(self.cache_dir, 'cache_info.json')
        
        if os.path.exists(cache_info_path):
            try:
                with open(cache_info_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading cache info: {e}")
                # Return empty cache info if file is corrupted
                return {'stabilized': {}, 'analysis': {}, 'last_cleaned': time.time()}
        else:
            # Create new cache info
            cache_info = {
                'stabilized': {},  # Maps video hash to stabilized video info
                'analysis': {},    # Maps video hash to analysis info
                'last_cleaned': time.time()  # Timestamp of last cache cleanup
            }
            self._save_cache_info(cache_info)
            return cache_info
    
    def _save_cache_info(self, cache_info=None):
        """Save cache information to disk."""
        if cache_info is None:
            cache_info = self.cache_info
            
        cache_info_path = os.path.join(self.cache_dir, 'cache_info.json')
        
        try:
            with open(cache_info_path, 'w') as f:
                json.dump(cache_info, f, indent=2)
        except IOError as e:
            print(f"Error saving cache info: {e}")
    
    def get_file_hash(self, file_path: str, block_size: int = 65536) -> Optional[str]:
        """
        Calculate MD5 hash of a file for cache validation.
        
        Parameters:
        -----------
        file_path : str
            Path to the file to hash
        block_size : int, optional
            Size of blocks to read when hashing large files
            
        Returns:
        --------
        str or None
            MD5 hash of the file, or None if file doesn't exist
        """
        if not os.path.exists(file_path):
            return None
        
        hash_obj = hashlib.md5()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                hash_obj.update(block)
        return hash_obj.hexdigest()
    
    def get_param_hash(self, params: Dict[str, Any]) -> str:
        """
        Generate a hash from processing parameters.
        
        Parameters:
        -----------
        params : dict
            Dictionary of parameters
            
        Returns:
        --------
        str
            Hash representing the parameters
        """
        # Convert parameters to a sorted, string representation
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def get_cache_size(self) -> Dict[str, Union[int, str]]:
        """
        Get the current size of the cache.
        
        Returns:
        --------
        dict
            Dictionary with keys 'stabilized', 'analysis', and 'total', containing the size in bytes and human-readable format
        """
        stabilized_size = 0
        analysis_size = 0
        
        # Get size of stabilized videos
        for root, dirs, files in os.walk(self.stabilized_cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    stabilized_size += os.path.getsize(file_path)
        
        # Get size of analysis files
        for root, dirs, files in os.walk(self.analysis_cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    analysis_size += os.path.getsize(file_path)
        
        total_size = stabilized_size + analysis_size
        
        # Convert to human-readable format
        def get_human_size(size_bytes):
            """Convert bytes to human-readable format."""
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                return f"{size_bytes/(1024**2):.1f} MB"
            else:
                return f"{size_bytes/(1024**3):.1f} GB"
        
        return {
            'stabilized': {
                'bytes': stabilized_size,
                'human': get_human_size(stabilized_size)
            },
            'analysis': {
                'bytes': analysis_size,
                'human': get_human_size(analysis_size)
            },
            'total': {
                'bytes': total_size,
                'human': get_human_size(total_size)
            }
        }
