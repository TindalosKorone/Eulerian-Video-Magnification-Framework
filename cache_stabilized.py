import os
import shutil
import time
from typing import Dict, Any, Optional, Tuple

from cache_core import CacheCore

class StabilizedVideoCache:
    """
    Manages caching of stabilized videos.
    """
    
    def __init__(self, cache_core: CacheCore):
        """
        Initialize the stabilized video cache manager.
        
        Parameters:
        -----------
        cache_core : CacheCore
            Core cache manager instance
        """
        self.core = cache_core
    
    def get_stabilized_video_path(self, video_path: str, stabilize_params: Dict[str, Any] = None) -> Optional[str]:
        """
        Check if a stabilized version of the video exists in cache.
        
        Parameters:
        -----------
        video_path : str
            Path to the original video file
        stabilize_params : dict, optional
            Dictionary of stabilization parameters, used to create the cache key
            
        Returns:
        --------
        str or None
            Path to the cached stabilized video if it exists and is valid, otherwise None
        """
        video_hash = self.core.get_file_hash(video_path)
        if not video_hash:
            return None
            
        # Default parameters if none provided
        if stabilize_params is None:
            stabilize_params = {}
            
        param_hash = self.core.get_param_hash(stabilize_params)
        cache_key = f"{video_hash}_{param_hash}"
        
        # Check if we have this video in cache
        if cache_key in self.core.cache_info['stabilized']:
            cache_entry = self.core.cache_info['stabilized'][cache_key]
            cached_path = cache_entry.get('path')
            
            # Verify the cached file exists
            if cached_path and os.path.exists(cached_path):
                # Check if the cache is stale
                if time.time() - cache_entry.get('timestamp', 0) > self.core.max_cache_age_days * 86400:
                    print(f"Cached stabilized video is older than {self.core.max_cache_age_days} days. Considering it stale.")
                    return None
                    
                print(f"Using cached stabilized video: {cached_path}")
                return cached_path
                
        return None
    
    def add_stabilized_video(self, original_video_path: str, stabilized_video_path: str, 
                           stabilize_params: Dict[str, Any] = None) -> None:
        """
        Add a stabilized video to the cache.
        
        Parameters:
        -----------
        original_video_path : str
            Path to the original video file
        stabilized_video_path : str
            Path to the stabilized video file
        stabilize_params : dict, optional
            Dictionary of stabilization parameters used
        """
        video_hash = self.core.get_file_hash(original_video_path)
        if not video_hash:
            print(f"Warning: Could not hash video {original_video_path}. Cache entry not created.")
            return
        
        # Default parameters if none provided
        if stabilize_params is None:
            stabilize_params = {}
            
        param_hash = self.core.get_param_hash(stabilize_params)
        cache_key = f"{video_hash}_{param_hash}"
        
        # Copy the stabilized video to the cache directory if it's not already there
        cache_filename = f"{os.path.splitext(os.path.basename(original_video_path))[0]}_stabilized_{param_hash}.mp4"
        cache_path = os.path.join(self.core.stabilized_cache_dir, cache_filename)
        
        if stabilized_video_path != cache_path:
            try:
                shutil.copy2(stabilized_video_path, cache_path)
                print(f"Copied stabilized video to cache: {cache_path}")
            except (IOError, shutil.Error) as e:
                print(f"Error copying stabilized video to cache: {e}")
                # If copy failed, use the original path
                cache_path = stabilized_video_path
        
        # Update cache info
        self.core.cache_info['stabilized'][cache_key] = {
            'path': cache_path,
            'original_video': original_video_path,
            'parameters': stabilize_params,
            'timestamp': time.time()
        }
        
        self.core._save_cache_info()
    
    def clean_stabilized_cache(self) -> int:
        """
        Clean up old stabilized video cache files.
        
        Returns:
        --------
        int
            Number of files removed
        """
        files_removed = 0
        stale_stabilized_keys = []
        
        for cache_key, cache_entry in self.core.cache_info['stabilized'].items():
            if time.time() - cache_entry.get('timestamp', 0) > self.core.max_cache_age_days * 86400:
                cached_path = cache_entry.get('path')
                if cached_path and os.path.exists(cached_path) and cached_path.startswith(self.core.stabilized_cache_dir):
                    try:
                        os.remove(cached_path)
                        files_removed += 1
                    except OSError as e:
                        print(f"Error removing stale cache file {cached_path}: {e}")
                
                stale_stabilized_keys.append(cache_key)
        
        # Remove stale entries from cache info
        for key in stale_stabilized_keys:
            del self.core.cache_info['stabilized'][key]
        
        return files_removed
