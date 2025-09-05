import os
import json
import hashlib
import time
import shutil
from typing import Dict, Any, List, Tuple, Optional, Union

class CacheManager:
    """
    Manages caching of temporary files and analysis results.
    Provides functionality to check if cached files are valid,
    create new cache entries, and clean up old cache files.
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
        video_hash = self.get_file_hash(video_path)
        if not video_hash:
            return None
            
        # Default parameters if none provided
        if stabilize_params is None:
            stabilize_params = {}
            
        param_hash = self.get_param_hash(stabilize_params)
        cache_key = f"{video_hash}_{param_hash}"
        
        # Check if we have this video in cache
        if cache_key in self.cache_info['stabilized']:
            cache_entry = self.cache_info['stabilized'][cache_key]
            cached_path = cache_entry.get('path')
            
            # Verify the cached file exists
            if cached_path and os.path.exists(cached_path):
                # Check if the cache is stale
                if time.time() - cache_entry.get('timestamp', 0) > self.max_cache_age_days * 86400:
                    print(f"Cached stabilized video is older than {self.max_cache_age_days} days. Considering it stale.")
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
        video_hash = self.get_file_hash(original_video_path)
        if not video_hash:
            print(f"Warning: Could not hash video {original_video_path}. Cache entry not created.")
            return
        
        # Default parameters if none provided
        if stabilize_params is None:
            stabilize_params = {}
            
        param_hash = self.get_param_hash(stabilize_params)
        cache_key = f"{video_hash}_{param_hash}"
        
        # Copy the stabilized video to the cache directory if it's not already there
        cache_filename = f"{os.path.splitext(os.path.basename(original_video_path))[0]}_stabilized_{param_hash}.mp4"
        cache_path = os.path.join(self.stabilized_cache_dir, cache_filename)
        
        if stabilized_video_path != cache_path:
            try:
                shutil.copy2(stabilized_video_path, cache_path)
                print(f"Copied stabilized video to cache: {cache_path}")
            except (IOError, shutil.Error) as e:
                print(f"Error copying stabilized video to cache: {e}")
                # If copy failed, use the original path
                cache_path = stabilized_video_path
        
        # Update cache info
        self.cache_info['stabilized'][cache_key] = {
            'path': cache_path,
            'original_video': original_video_path,
            'parameters': stabilize_params,
            'timestamp': time.time()
        }
        
        self._save_cache_info()
    
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
        video_hash = self.get_file_hash(video_path)
        if not video_hash:
            return None
            
        # Default parameters if none provided
        if analysis_params is None:
            analysis_params = {}
            
        param_hash = self.get_param_hash(analysis_params)
        cache_key = f"{video_hash}_{param_hash}"
        
        # Check if we have this analysis in cache
        if cache_key in self.cache_info['analysis']:
            cache_entry = self.cache_info['analysis'][cache_key]
            cached_path = cache_entry.get('metadata_path')
            
            # Verify the cached file exists
            if cached_path and os.path.exists(cached_path):
                # Check if the cache is stale
                if time.time() - cache_entry.get('timestamp', 0) > self.max_cache_age_days * 86400:
                    print(f"Cached analysis is older than {self.max_cache_age_days} days. Considering it stale.")
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
        video_hash = self.get_file_hash(video_path)
        if not video_hash:
            print(f"Warning: Could not hash video {video_path}. Cache entry not created.")
            return
        
        # Default parameters if none provided
        if analysis_params is None:
            analysis_params = {}
            
        param_hash = self.get_param_hash(analysis_params)
        cache_key = f"{video_hash}_{param_hash}"
        
        # Copy the metadata and visualizations to the cache directory if not already there
        if not metadata_path.startswith(self.analysis_cache_dir):
            cache_basename = f"{os.path.splitext(os.path.basename(video_path))[0]}_analysis_{param_hash}"
            
            # Copy metadata file
            cache_metadata_path = os.path.join(self.analysis_cache_dir, f"{cache_basename}.json")
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
                        cache_viz_path = os.path.join(self.analysis_cache_dir, viz_filename)
                        try:
                            shutil.copy2(path, cache_viz_path)
                            new_viz_paths[viz_type][freq] = cache_viz_path
                        except (IOError, shutil.Error) as e:
                            print(f"Error copying visualization to cache: {e}")
                            new_viz_paths[viz_type][freq] = path
                else:
                    viz_filename = f"{cache_basename}_{viz_type}.png"
                    cache_viz_path = os.path.join(self.analysis_cache_dir, viz_filename)
                    try:
                        shutil.copy2(viz_path, cache_viz_path)
                        new_viz_paths[viz_type] = cache_viz_path
                    except (IOError, shutil.Error) as e:
                        print(f"Error copying visualization to cache: {e}")
                        new_viz_paths[viz_type] = viz_path
            
            visualization_paths = new_viz_paths
        
        # Update cache info
        self.cache_info['analysis'][cache_key] = {
            'metadata_path': metadata_path,
            'visualization_paths': visualization_paths,
            'video_path': video_path,
            'parameters': analysis_params,
            'timestamp': time.time()
        }
        
        self._save_cache_info()
    
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
            last_cleaned = self.cache_info.get('last_cleaned', 0)
            if time.time() - last_cleaned < 7 * 86400:  # Clean weekly
                return 0
        
        print("Cleaning cache...")
        files_removed = 0
        
        # Clean stabilized videos
        stale_stabilized_keys = []
        for cache_key, cache_entry in self.cache_info['stabilized'].items():
            if time.time() - cache_entry.get('timestamp', 0) > self.max_cache_age_days * 86400:
                cached_path = cache_entry.get('path')
                if cached_path and os.path.exists(cached_path) and cached_path.startswith(self.stabilized_cache_dir):
                    try:
                        os.remove(cached_path)
                        files_removed += 1
                    except OSError as e:
                        print(f"Error removing stale cache file {cached_path}: {e}")
                
                stale_stabilized_keys.append(cache_key)
        
        # Remove stale entries from cache info
        for key in stale_stabilized_keys:
            del self.cache_info['stabilized'][key]
        
        # Clean analysis files
        stale_analysis_keys = []
        for cache_key, cache_entry in self.cache_info['analysis'].items():
            if time.time() - cache_entry.get('timestamp', 0) > self.max_cache_age_days * 86400:
                # Remove metadata file
                metadata_path = cache_entry.get('metadata_path')
                if metadata_path and os.path.exists(metadata_path) and metadata_path.startswith(self.analysis_cache_dir):
                    try:
                        os.remove(metadata_path)
                        files_removed += 1
                    except OSError as e:
                        print(f"Error removing stale cache file {metadata_path}: {e}")
                
                # Remove visualization files
                for viz_type, viz_path in cache_entry.get('visualization_paths', {}).items():
                    if isinstance(viz_path, dict):
                        for freq, path in viz_path.items():
                            if path and os.path.exists(path) and path.startswith(self.analysis_cache_dir):
                                try:
                                    os.remove(path)
                                    files_removed += 1
                                except OSError as e:
                                    print(f"Error removing stale cache file {path}: {e}")
                    elif viz_path and os.path.exists(viz_path) and viz_path.startswith(self.analysis_cache_dir):
                        try:
                            os.remove(viz_path)
                            files_removed += 1
                        except OSError as e:
                            print(f"Error removing stale cache file {viz_path}: {e}")
                
                stale_analysis_keys.append(cache_key)
        
        # Remove stale entries from cache info
        for key in stale_analysis_keys:
            del self.cache_info['analysis'][key]
        
        # Update last cleaned timestamp
        self.cache_info['last_cleaned'] = time.time()
        self._save_cache_info()
        
        print(f"Cache cleaned, {files_removed} files removed.")
        return files_removed
    
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
