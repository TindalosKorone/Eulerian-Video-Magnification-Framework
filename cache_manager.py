import os
import time
from typing import Dict, Any, Optional, Union

from cache_core import CacheCore
from cache_stabilized import StabilizedVideoCache
from cache_analysis import AnalysisResultsCache

class CacheManager:
    """
    Manages caching of temporary files and analysis results.
    This is the main entry point for cache operations, using the specialized
    cache components for different types of cached data.
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
        # Initialize the core cache manager
        self.core = CacheCore(cache_dir, max_cache_age_days)
        
        # Initialize specialized cache managers
        self.stabilized = StabilizedVideoCache(self.core)
        self.analysis = AnalysisResultsCache(self.core)
    
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
        return self.stabilized.get_stabilized_video_path(video_path, stabilize_params)
    
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
        self.stabilized.add_stabilized_video(original_video_path, stabilized_video_path, stabilize_params)
    
    def get_analysis_results_path(self, video_path: str, analysis_params: Dict[str, Any] = None) -> Optional[str]:
        """
        Check if frequency analysis results exist in cache.
        
        Parameters:
        -----------
        video_path : str
            Path to the video file
        analysis_params : dict, optional
            Dictionary of analysis parameters, used to create the cache key
            
        Returns:
        --------
        str or None
            Path to the cached analysis results JSON if it exists and is valid, otherwise None
        """
        return self.analysis.get_analysis_results_path(video_path, analysis_params)
    
    def add_analysis_results(self, video_path: str, metadata_path: str, visualization_paths: Dict[str, Any],
                           analysis_params: Dict[str, Any] = None) -> None:
        """
        Add frequency analysis results to the cache.
        
        Parameters:
        -----------
        video_path : str
            Path to the video file
        metadata_path : str
            Path to the analysis metadata JSON file
        visualization_paths : dict
            Dictionary mapping visualization types to file paths
        analysis_params : dict, optional
            Dictionary of analysis parameters used
        """
        self.analysis.add_analysis_results(video_path, metadata_path, visualization_paths, analysis_params)
    
    def clean_cache(self, force=False) -> int:
        """
        Clean up old cache files.
        
        Parameters:
        -----------
        force : bool, optional
            If True, clean the cache regardless of when it was last cleaned
            
        Returns:
        --------
        int
            Number of files removed
        """
        # Check if it's time to clean the cache
        if not force:
            last_cleaned = self.core.cache_info.get('last_cleaned', 0)
            if time.time() - last_cleaned < 7 * 86400:  # Clean weekly
                return 0
        
        print("Cleaning cache...")
        
        # Clean stabilized videos
        stabilized_removed = self.stabilized.clean_stabilized_cache()
        
        # Clean analysis files
        analysis_removed = self.analysis.clean_analysis_cache()
        
        # Update last cleaned timestamp
        self.core.cache_info['last_cleaned'] = time.time()
        self.core._save_cache_info()
        
        total_removed = stabilized_removed + analysis_removed
        print(f"Cache cleaned, {total_removed} files removed.")
        return total_removed
    
    def get_cache_size(self) -> Dict[str, Union[int, str]]:
        """
        Get the current size of the cache.
        
        Returns:
        --------
        dict
            Dictionary with keys 'stabilized', 'analysis', and 'total', containing the size in bytes and human-readable format
        """
        return self.core.get_cache_size()
