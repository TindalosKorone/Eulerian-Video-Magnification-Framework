import os
import json
import hashlib
import time
import shutil
from typing import Dict, Any, List, Tuple, Optional, Union

class CacheCore:
    """
    核心缓存管理功能。
    提供缓存操作的基础方法。
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
        if cache_dir is None:
            # 默认为当前工作目录下的'TEMP'目录
            self.cache_dir = os.path.join(os.getcwd(), 'TEMP')
        else:
            self.cache_dir = cache_dir
            
        self.max_cache_age_days = max_cache_age_days
        
        # 如果缓存目录不存在，则创建
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # 为不同类型的缓存创建子目录
        self.stabilized_cache_dir = os.path.join(self.cache_dir, 'stabilized')
        self.analysis_cache_dir = os.path.join(self.cache_dir, 'analysis')
        
        for dir_path in [self.stabilized_cache_dir, self.analysis_cache_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
        # 用于在内存中存储缓存信息的映射
        self.cache_info = self._load_cache_info()
    
    def _load_cache_info(self) -> Dict[str, Any]:
        """从磁盘加载缓存信息，如果不存在则创建。"""
        cache_info_path = os.path.join(self.cache_dir, 'cache_info.json')
        
        if os.path.exists(cache_info_path):
            try:
                with open(cache_info_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"加载缓存信息时出错: {e}")
                # 如果文件损坏，返回空的缓存信息
                return {'stabilized': {}, 'analysis': {}, 'last_cleaned': time.time()}
        else:
            # 创建新的缓存信息
            cache_info = {
                'stabilized': {},  # 将视频哈希映射到稳定化视频信息
                'analysis': {},    # 将视频哈希映射到分析信息
                'last_cleaned': time.time()  # 上次缓存清理的时间戳
            }
            self._save_cache_info(cache_info)
            return cache_info
    
    def _save_cache_info(self, cache_info=None):
        """将缓存信息保存到磁盘。"""
        if cache_info is None:
            cache_info = self.cache_info
            
        cache_info_path = os.path.join(self.cache_dir, 'cache_info.json')
        
        try:
            with open(cache_info_path, 'w') as f:
                json.dump(cache_info, f, indent=2)
        except IOError as e:
            print(f"保存缓存信息时出错: {e}")
    
    def get_file_hash(self, file_path: str, block_size: int = 65536) -> Optional[str]:
        """
        计算文件的MD5哈希值，用于缓存验证。
        
        参数:
        -----------
        file_path : str
            要哈希的文件路径
        block_size : int, optional
            哈希大文件时读取的块大小
            
        返回:
        --------
        str or None
            文件的MD5哈希值，如果文件不存在则返回None
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
        从处理参数生成哈希值。
        
        参数:
        -----------
        params : dict
            参数字典
            
        返回:
        --------
        str
            表示参数的哈希值
        """
        # 将参数转换为排序的字符串表示
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def get_cache_size(self) -> Dict[str, Union[int, str]]:
        """
        获取当前缓存的大小。
        
        返回:
        --------
        dict
            包含键'stabilized'、'analysis'和'total'的字典，包含字节大小和人类可读格式
        """
        stabilized_size = 0
        analysis_size = 0
        
        # 获取稳定化视频的大小
        for root, dirs, files in os.walk(self.stabilized_cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    stabilized_size += os.path.getsize(file_path)
        
        # 获取分析文件的大小
        for root, dirs, files in os.walk(self.analysis_cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    analysis_size += os.path.getsize(file_path)
        
        total_size = stabilized_size + analysis_size
        
        # 转换为人类可读格式
        def get_human_size(size_bytes):
            """将字节转换为人类可读格式。"""
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
