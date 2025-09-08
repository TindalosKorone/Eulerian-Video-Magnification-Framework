import os
import shutil
import time
from typing import Dict, Any, Optional, List

from cache_core import CacheCore

class AnalysisResultsCache:
    """
    管理频率分析结果的缓存。
    """
    
    def __init__(self, cache_core: CacheCore):
        """
        初始化分析结果缓存管理器。
        
        参数:
        -----------
        cache_core : CacheCore
            核心缓存管理器实例
        """
        self.core = cache_core
    
    def get_analysis_results_path(self, video_path: str, analysis_params: Dict[str, Any] = None) -> Optional[str]:
        """
        检查频率分析结果是否存在于缓存中。
        
        参数:
        -----------
        video_path : str
            视频文件路径
        analysis_params : dict, optional
            分析参数字典，用于创建缓存键
            
        返回:
        --------
        str or None
            如果缓存的分析结果JSON存在且有效，则返回其路径，否则返回None
        """
        video_hash = self.core.get_file_hash(video_path)
        if not video_hash:
            return None
            
        # 如果没有提供参数，则使用默认参数
        if analysis_params is None:
            analysis_params = {}
            
        param_hash = self.core.get_param_hash(analysis_params)
        cache_key = f"{video_hash}_{param_hash}"
        
        # 检查缓存中是否有此分析
        if cache_key in self.core.cache_info['analysis']:
            cache_entry = self.core.cache_info['analysis'][cache_key]
            cached_path = cache_entry.get('metadata_path')
            
            # 验证缓存文件是否存在
            if cached_path and os.path.exists(cached_path):
                # 检查缓存是否过期
                if time.time() - cache_entry.get('timestamp', 0) > self.core.max_cache_age_days * 86400:
                    print(f"缓存的分析结果超过 {self.core.max_cache_age_days} 天。视为过期。")
                    return None
                
                # 验证可视化文件是否存在
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
                    print("部分可视化文件丢失。缓存无效。")
                    return None
                    
                print(f"使用缓存的分析结果: {cached_path}")
                return cached_path
                
        return None
    
    def add_analysis_results(self, video_path: str, metadata_path: str, visualization_paths: Dict[str, Any],
                           analysis_params: Dict[str, Any] = None) -> None:
        """
        将频率分析结果添加到缓存中。
        
        参数:
        -----------
        video_path : str
            视频文件路径
        metadata_path : str
            分析元数据JSON文件路径
        visualization_paths : dict
            可视化类型到文件路径的映射字典
        analysis_params : dict, optional
            使用的分析参数字典
        """
        video_hash = self.core.get_file_hash(video_path)
        if not video_hash:
            print(f"警告: 无法对视频 {video_path} 进行哈希。未创建缓存条目。")
            return
        
        # 如果没有提供参数，则使用默认参数
        if analysis_params is None:
            analysis_params = {}
            
        param_hash = self.core.get_param_hash(analysis_params)
        cache_key = f"{video_hash}_{param_hash}"
        
        # 如果元数据和可视化文件不在缓存目录中，则复制到缓存目录
        if not metadata_path.startswith(self.core.analysis_cache_dir):
            cache_basename = f"{os.path.splitext(os.path.basename(video_path))[0]}_analysis_{param_hash}"
            
            # 复制元数据文件
            cache_metadata_path = os.path.join(self.core.analysis_cache_dir, f"{cache_basename}.json")
            try:
                shutil.copy2(metadata_path, cache_metadata_path)
                metadata_path = cache_metadata_path
            except (IOError, shutil.Error) as e:
                print(f"复制分析元数据到缓存时出错: {e}")
            
            # 复制可视化文件
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
                            print(f"复制可视化文件到缓存时出错: {e}")
                            new_viz_paths[viz_type][freq] = path
                else:
                    viz_filename = f"{cache_basename}_{viz_type}.png"
                    cache_viz_path = os.path.join(self.core.analysis_cache_dir, viz_filename)
                    try:
                        shutil.copy2(viz_path, cache_viz_path)
                        new_viz_paths[viz_type] = cache_viz_path
                    except (IOError, shutil.Error) as e:
                        print(f"复制可视化文件到缓存时出错: {e}")
                        new_viz_paths[viz_type] = viz_path
            
            visualization_paths = new_viz_paths
        
        # 更新缓存信息
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
        清理旧的分析缓存文件。
        
        返回:
        --------
        int
            移除的文件数量
        """
        files_removed = 0
        stale_analysis_keys = []
        
        for cache_key, cache_entry in self.core.cache_info['analysis'].items():
            if time.time() - cache_entry.get('timestamp', 0) > self.core.max_cache_age_days * 86400:
                # 移除元数据文件
                metadata_path = cache_entry.get('metadata_path')
                if metadata_path and os.path.exists(metadata_path) and metadata_path.startswith(self.core.analysis_cache_dir):
                    try:
                        os.remove(metadata_path)
                        files_removed += 1
                    except OSError as e:
                        print(f"移除过期缓存文件 {metadata_path} 时出错: {e}")
                
                # 移除可视化文件
                for viz_type, viz_path in cache_entry.get('visualization_paths', {}).items():
                    if isinstance(viz_path, dict):
                        for freq, path in viz_path.items():
                            if path and os.path.exists(path) and path.startswith(self.core.analysis_cache_dir):
                                try:
                                    os.remove(path)
                                    files_removed += 1
                                except OSError as e:
                                    print(f"移除过期缓存文件 {path} 时出错: {e}")
                    elif viz_path and os.path.exists(viz_path) and viz_path.startswith(self.core.analysis_cache_dir):
                        try:
                            os.remove(viz_path)
                            files_removed += 1
                        except OSError as e:
                            print(f"移除过期缓存文件 {viz_path} 时出错: {e}")
                
                stale_analysis_keys.append(cache_key)
        
        # 从缓存信息中移除过期条目
        for key in stale_analysis_keys:
            del self.core.cache_info['analysis'][key]
        
        return files_removed
