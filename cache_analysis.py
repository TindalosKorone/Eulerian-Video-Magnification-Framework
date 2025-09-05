import os
import shutil
import time
from typing import Dict, Any, Optional, List

from cache_core import CacheCore

class AnalysisResultsCache:
    """
    Manages caching of frequency analysis results.
    """
    
    def __init__(self, cache_core: CacheCore):
        """
        Initialize the analysis results cache manager.
        
        Parameters:
        -----------
        cache_core : CacheCore
            Core cache manager instance
        """
        self.core = cache_core
    
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
        video_hash = self.core.get_file_hash(video_path)
        if not video_hash:
            return None
            
        # Default parameters if none provided
        if analysis_params is None:
            analysis_params = {}
            
        param_hash = self.core.get_param_hash(analysis_params)
        cache_key = f"{video_hash}_{param_hash}"
        
        # Check if we have this analysis in cache
        if cache_key in self.core.cache_info['analysis']:
            cache_entry = self.core.cache_info['analysis'][cache_key]
            cached_path = cache_entry.get('metadata_path')
            
            # Verify the cached file exists
            if cached_path and os.path.exists(cached_path):
                # Check if the cache is stale
                if time.time() - cache_entry.get('timestamp', 0) > self.core.max_cache_age_days * 86400:
                    print(f"Cached analysis is older than {self.core.max_cache_age_days} days. Considering it stale.")
                    return None
                
                # Verify that visualization files exist
                viz_paths = cache_entry.get('visualization_paths', {})
                all_viz_exist = True
                
                for viz_type, viz_path in viz_paths.items():
                    if isinstance(viz_path, dict):
                        for freq, path in viz_path.items():
                            if not os.path.exists(path):
                                all_viz_exist = False
                                break
                    elif not os.path.exists(viz_path):
                        all_viz_exist = False
                        break
                
                if not all_viz_exist:
                    print("Some visualization files are missing. Cache is invalid.")
                    return None
                    
                print(f"Using cached analysis results: {cached_path}")
                return cached_path
                
        return None
    
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
        video_hash = self.core.get_file_hash(video_path)
        if not video_hash:
            print(f"Warning: Could not hash video {video_path}. Cache entry not created.")
            return
        
        # Default parameters if none provided
        if analysis_params is None:
            analysis_params = {}
            
        param_hash = self.core.get_param_hash(analysis_params)
        cache_key = f"{video_hash}_{param_hash}"
        
        # Copy the metadata and visualizations to the cache directory if not already there
        if not metadata_path.startswith(self.core.analysis_cache_dir):
            cache_basename = f"{os.path.splitext(os.path.basename(video_path))[0]}_analysis_{param_hash}"
            
            # Copy metadata file
            cache_metadata_path = os.path.join(self.core.analysis_cache_dir, f"{cache_basename}.json")
            try:
                shutil.copy2(metadata_path, cache_metadata_path)
                metadata_path = cache_metadata_path
            except (IOError, shutil.Error) as e:
                print(f"Error copying analysis metadata to cache: {e}")
            
            # Copy visualization files
            new_viz_paths = {}
            for viz_type, viz_path in visualization_paths.items():
                if isinstance(viz_path, dict):
                    new_viz_paths[viz_type] = {}
                    for freq, path in viz_path.items():
                        viz_filename = f"{cache_basename}_{viz_type}_{freq}.png"
                        cache_viz_path = os.path.join(self.core.analysis_cache_dir, viz_filename)
                        try:
                            shutil.copy2(path, cache_viz_path)
                            new_viz_paths[viz_type][freq] = cache_viz_path
                        except (IOError, shutil.Error) as e:
                            print(f"Error copying visualization to cache: {e}")
                            new_viz_paths[viz_type][freq] = path
                else:
                    viz_filename = f"{cache_basename}_{viz_type}.png"
                    cache_viz_path = os.path.join(self.core.analysis_cache_dir, viz_filename)
                    try:
                        shutil.copy2(viz_path, cache_viz_path)
                        new_viz_paths[viz_type] = cache_viz_path
                    except (IOError, shutil.Error) as e:
                        print(f"Error copying visualization to cache: {e}")
                        new_viz_paths[viz_type] = viz_path
            
            visualization_paths = new_viz_paths
        
        # Update cache info
        self.core.cache_info['analysis'][cache_key] = {
            'metadata_path': metadata_path,
            'visualization_paths': visualization_paths,
            'video_path': video_path,
            'parameters': analysis_params,
            'timestamp': time.time()
        }
        
        self.core._save_cache_info()
    
    def clean_analysis_cache(self) -> int:
        """
        Clean up old analysis cache files.
        
        Returns:
        --------
        int
            Number of files removed
        """
        files_removed = 0
        stale_analysis_keys = []
        
        for cache_key, cache_entry in self.core.cache_info['analysis'].items():
            if time.time() - cache_entry.get('timestamp', 0) > self.core.max_cache_age_days * 86400:
                # Remove metadata file
                metadata_path = cache_entry.get('metadata_path')
                if metadata_path and os.path.exists(metadata_path) and metadata_path.startswith(self.core.analysis_cache_dir):
                    try:
                        os.remove(metadata_path)
                        files_removed += 1
                    except OSError as e:
                        print(f"Error removing stale cache file {metadata_path}: {e}")
                
                # Remove visualization files
                for viz_type, viz_path in cache_entry.get('visualization_paths', {}).items():
                    if isinstance(viz_path, dict):
                        for freq, path in viz_path.items():
                            if path and os.path.exists(path) and path.startswith(self.core.analysis_cache_dir):
                                try:
                                    os.remove(path)
                                    files_removed += 1
                                except OSError as e:
                                    print(f"Error removing stale cache file {path}: {e}")
                    elif viz_path and os.path.exists(viz_path) and viz_path.startswith(self.core.analysis_cache_dir):
                        try:
                            os.remove(viz_path)
                            files_removed += 1
                        except OSError as e:
                            print(f"Error removing stale cache file {viz_path}: {e}")
                
                stale_analysis_keys.append(cache_key)
        
        # Remove stale entries from cache info
        for key in stale_analysis_keys:
            del self.core.cache_info['analysis'][key]
        
        return files_removed
