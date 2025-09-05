import sys
import time
from datetime import datetime

class AnalysisProgressTracker:
    """跟踪和显示分析进度的类"""
    
    def __init__(self, total_frames, width=70):
        """
        初始化进度跟踪器
        
        Parameters:
        -----------
        total_frames : int
            总帧数
        width : int, optional
            显示宽度
        """
        self.total_frames = total_frames
        self.width = width
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 0.25  # 更新间隔（秒）
        
        # 打印标题
        self._print_header()
    
    def _print_header(self):
        """打印进度跟踪器标题"""
        print("\n" + "="*self.width)
        print(f"📊 视频频率分析 - 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        print("="*self.width)
    
    def update(self, current_frame, stats=None):
        """
        更新进度显示
        
        Parameters:
        -----------
        current_frame : int
            当前处理的帧
        stats : dict, optional
            附加统计信息
        """
        current_time = time.time()
        if current_time - self.last_update < self.update_interval and current_frame < self.total_frames - 1:
            return
        
        self.last_update = current_time
        progress = current_frame / self.total_frames
        
        # 创建进度条
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # 计算剩余时间
        elapsed = current_time - self.start_time
        if progress > 0:
            eta = elapsed / progress - elapsed
        else:
            eta = 0
        
        # 格式化消息
        message = f"[{bar}] {current_frame}/{self.total_frames} ({int(100*progress)}%)"
        
        # 添加统计信息
        if stats:
            stats_str = " | ".join([f"{k}: {v}" for k, v in stats.items()])
            message += f" | {stats_str}"
        
        # 添加时间信息
        minutes, seconds = divmod(int(eta), 60)
        message += f" | ETA: {minutes:02d}:{seconds:02d}"
        
        # 打印消息
        sys.stdout.write("\r" + " " * self.width + "\r")
        sys.stdout.write(message)
        sys.stdout.flush()
    
    def finish(self):
        """完成分析，显示总耗时"""
        elapsed = time.time() - self.start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(f"\n✅ 分析完成 - 耗时: {minutes:02d}:{seconds:02d}")
        print("="*self.width)
