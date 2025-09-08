import os
import time
from typing import Dict, Any, Optional, Union

from cache_core import CacheCore
from cache_stabilized import StabilizedVideoCache
from cache_analysis import AnalysisResultsCache

class CacheManager:
    """
    管理临时文件和分析结果的缓存。
    这是缓存操作的主要入口点，使用专门的
    缓存组件来处理不同类型的缓存数据。
    """
    
    def __init__(self, cache_dir=None, max_cache_age_days=30):
        """
        初始化缓存管理器。
        
        参数:
        -----------
        cache_dir : str, optional
            存储缓存文件的目录。如果为None，默认为当前目录下的'TEMP'。
        max_cache_age_days : int, optional
            缓存文件被视为过期前的最大保存天数。
        """
        # 初始化核心缓存管理器
        self.core = CacheCore(cache_dir, max_cache_age_days)
        
        # 初始化专门的缓存管理器
        self.stabilized = StabilizedVideoCache(self.core)
        self.analysis = AnalysisResultsCache(self.core)
    
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
        return self.stabilized.get_stabilized_video_path(video_path, stabilize_params)
    
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
        self.stabilized.add_stabilized_video(original_video_path, stabilized_video_path, stabilize_params)
    
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
        return self.analysis.get_analysis_results_path(video_path, analysis_params)
    
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
        self.analysis.add_analysis_results(video_path, metadata_path, visualization_paths, analysis_params)
    
    def clean_cache(self, force=False) -> int:
        """
        清理旧的缓存文件。
        
        参数:
        -----------
        force : bool, optional
            如果为True，无论上次清理时间如何都清理缓存
            
        返回:
        --------
        int
            移除的文件数量
        """
        # 检查是否是时候清理缓存
        if not force:
            last_cleaned = self.core.cache_info.get('last_cleaned', 0)
            if time.time() - last_cleaned < 7 * 86400:  # 每周清理
                return 0
        
        print("正在清理缓存...")
        
        # 清理稳定化视频
        stabilized_removed = self.stabilized.clean_stabilized_cache()
        
        # 清理分析文件
        analysis_removed = self.analysis.clean_analysis_cache()
        
        # 更新上次清理时间戳
        self.core.cache_info['last_cleaned'] = time.time()
        self.core._save_cache_info()
        
        total_removed = stabilized_removed + analysis_removed
        print(f"缓存已清理，移除了 {total_removed} 个文件。")
        return total_removed
    
    def get_cache_size(self) -> Dict[str, Union[int, str]]:
        """
        获取当前缓存的大小。
        
        返回:
        --------
        dict
            包含键'stabilized'、'analysis'和'total'的字典，包含字节大小和人类可读格式
        """
        return self.core.get_cache_size()
