import os
import shutil
import time
from typing import Dict, Any, Optional, Tuple

from cache_core import CacheCore

class StabilizedVideoCache:
    """
    管理稳定化视频的缓存。
    """
    
    def __init__(self, cache_core: CacheCore):
        """
        初始化稳定化视频缓存管理器。
        
        参数:
        -----------
        cache_core : CacheCore
            核心缓存管理器实例
        """
        self.core = cache_core
    
    def get_stabilized_video_path(self, video_path: str, stabilize_params: Dict[str, Any] = None) -> Optional[str]:
        """
        检查视频的稳定化版本是否存在于缓存中。
        
        参数:
        -----------
        video_path : str
            原始视频文件的路径
        stabilize_params : dict, optional
            稳定化参数字典，用于创建缓存键
            
        返回:
        --------
        str or None
            如果缓存的稳定化视频存在且有效，则返回其路径，否则返回None
        """
        video_hash = self.core.get_file_hash(video_path)
        if not video_hash:
            return None
            
        # 如果没有提供参数，则使用默认参数
        if stabilize_params is None:
            stabilize_params = {}
            
        param_hash = self.core.get_param_hash(stabilize_params)
        cache_key = f"{video_hash}_{param_hash}"
        
        # 检查缓存中是否有此视频
        if cache_key in self.core.cache_info['stabilized']:
            cache_entry = self.core.cache_info['stabilized'][cache_key]
            cached_path = cache_entry.get('path')
            
            # 验证缓存文件是否存在
            if cached_path and os.path.exists(cached_path):
                # 检查缓存是否过期
                if time.time() - cache_entry.get('timestamp', 0) > self.core.max_cache_age_days * 86400:
                    print(f"缓存的稳定化视频超过 {self.core.max_cache_age_days} 天。视为过期。")
                    return None
                    
                print(f"使用缓存的稳定化视频: {cached_path}")
                return cached_path
                
        return None
    
    def add_stabilized_video(self, original_video_path: str, stabilized_video_path: str, 
                           stabilize_params: Dict[str, Any] = None) -> None:
        """
        将稳定化视频添加到缓存中。
        
        参数:
        -----------
        original_video_path : str
            原始视频文件的路径
        stabilized_video_path : str
            稳定化视频文件的路径
        stabilize_params : dict, optional
            使用的稳定化参数字典
        """
        video_hash = self.core.get_file_hash(original_video_path)
        if not video_hash:
            print(f"警告: 无法对视频 {original_video_path} 进行哈希。未创建缓存条目。")
            return
        
        # 如果没有提供参数，则使用默认参数
        if stabilize_params is None:
            stabilize_params = {}
            
        param_hash = self.core.get_param_hash(stabilize_params)
        cache_key = f"{video_hash}_{param_hash}"
        
        # 如果稳定化视频不在缓存目录中，则复制到缓存目录
        cache_filename = f"{os.path.splitext(os.path.basename(original_video_path))[0]}_stabilized_{param_hash}.mp4"
        cache_path = os.path.join(self.core.stabilized_cache_dir, cache_filename)
        
        if stabilized_video_path != cache_path:
            try:
                shutil.copy2(stabilized_video_path, cache_path)
                print(f"已将稳定化视频复制到缓存: {cache_path}")
            except (IOError, shutil.Error) as e:
                print(f"复制稳定化视频到缓存时出错: {e}")
                # 如果复制失败，使用原始路径
                cache_path = stabilized_video_path
        
        # 更新缓存信息
        self.core.cache_info['stabilized'][cache_key] = {
            'path': cache_path,
            'original_video': original_video_path,
            'parameters': stabilize_params,
            'timestamp': time.time()
        }
        
        self.core._save_cache_info()
    
    def clean_stabilized_cache(self) -> int:
        """
        清理旧的稳定化视频缓存文件。
        
        返回:
        --------
        int
            移除的文件数量
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
                        print(f"移除过期缓存文件 {cached_path} 时出错: {e}")
                
                stale_stabilized_keys.append(cache_key)
        
        # 从缓存信息中移除过期条目
        for key in stale_stabilized_keys:
            del self.core.cache_info['stabilized'][key]
        
        return files_removed
