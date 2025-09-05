import numpy as np
import cv2
import os
import json
from scipy import signal
from datetime import datetime

from frequency_presets import FrequencyPresets
from analysis_progress import AnalysisProgressTracker
import frequency_visualizer as viz

def load_analysis_results(metadata_path):
    """加载之前生成的频率分析结果"""
    try:
        with open(metadata_path, 'r') as f:
            analysis_data = json.load(f)
        return analysis_data
    except Exception as e:
        print(f"加载分析结果出错: {e}")
        return None

def visualize_frequency_preset(preset_name, fps=30, output_dir=None):
    """
    可视化预设的频率响应。
    
    Parameters:
    -----------
    preset_name : str
        预设名称
    fps : float, optional
        帧率，用于计算频率响应
    output_dir : str, optional
        输出目录
    """
    preset_data = FrequencyPresets.get_preset(preset_name)
    viz.visualize_frequency_preset(preset_name, preset_data, fps, output_dir)

def analyze_video_frequencies(video_path, output_dir=None, sampling_rate=0.5, 
                             region_of_interest=None, preset_name=None, 
                             generate_visualizations=True):
    """
    分析视频中的运动频率并生成可视化结果。
    
    Parameters:
    -----------
    video_path : str
        输入视频路径
    output_dir : str, optional
        输出目录，如不指定则使用视频所在目录
    sampling_rate : float, optional
        采样率（0.0-1.0），较低的值可加快分析速度
    region_of_interest : tuple, optional
        感兴趣区域 (x, y, width, height)，如不指定则分析整个画面
    preset_name : str, optional
        预设名称，用于设置显示范围
    generate_visualizations : bool, optional
        是否生成可视化图像
        
    Returns:
    --------
    dict
        包含分析结果和可视化路径的字典
    """
    print(f"正在分析视频: {os.path.basename(video_path)}")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(video_path))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 提取基本名称
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 如果指定了预设，获取频率范围
    if preset_name:
        preset = FrequencyPresets.get_preset(preset_name)
        max_freq = preset['freq_max']
    else:
        max_freq = min(fps / 2 * 0.8, 30.0)  # 默认最高频率（不超过奈奎斯特频率）
    
    # 计算采样
    sample_step = max(1, int(1/sampling_rate))
    sampled_frames = range(0, n_frames, sample_step)
    analysis_fps = fps / sample_step
    
    # 设置感兴趣区域
    if region_of_interest:
        roi_x, roi_y, roi_width, roi_height = region_of_interest
        roi = (slice(roi_y, roi_y + roi_height), slice(roi_x, roi_x + roi_width))
    else:
        # 确保在没有指定ROI时也设置roi_width和roi_height
        roi_width, roi_height = width, height
        roi = (slice(0, height), slice(0, width))
    
    # 打印基本信息
    print(f"视频信息: {width}x{height}, {fps:.2f}fps, {n_frames}帧")
    print(f"分析设置: 采样率1/{sample_step}帧, 有效帧率{analysis_fps:.2f}fps")
    if region_of_interest:
        print(f"感兴趣区域: {roi_x},{roi_y} - {roi_width}x{roi_height}")
    
    # 初始化进度跟踪器
    progress = AnalysisProgressTracker(len(sampled_frames))
    
    # 初始化运动数据存储
    pixel_values = []
    
    # 读取第一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    if not ret:
        raise IOError("无法读取第一帧")
    
    # 转换为灰度
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[roi]
    
    # 选择要跟踪的点（均匀网格）
    grid_step = max(5, min(roi_width, roi_height) // 50)
    y_coords, x_coords = np.mgrid[0:prev_gray.shape[0]:grid_step, 0:prev_gray.shape[1]:grid_step]
    points = np.vstack((x_coords.flatten(), y_coords.flatten())).T.astype(np.float32)
    
    # 跟踪点的运动
    for i, frame_num in enumerate(sampled_frames[1:]):
        # 更新进度
        progress.update(i, {"区域": f"{len(points)}个点"})
        
        # 设置帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换为灰度
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi]
        
        # 计算光流
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, points.reshape(-1, 1, 2), None
        )
        
        # 过滤掉无法跟踪的点
        good_points = points[status.flatten() == 1]
        good_next_points = next_points[status.flatten() == 1].reshape(-1, 2)
        
        # 计算运动向量
        if len(good_points) > 0:
            # 计算运动幅度
            motion_vectors = good_next_points - good_points
            magnitudes = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)
            
            # 存储平均运动幅度
            pixel_values.append(np.mean(magnitudes))
        else:
            pixel_values.append(0)
        
        # 更新
        prev_gray = curr_gray
        points = good_next_points.reshape(-1, 2)
    
    cap.release()
    progress.finish()
    
    # 计算频率分析
    if len(pixel_values) > 10:  # 确保有足够的数据点
        print("正在计算频率分析...")
        
        # 应用窗口函数减少频谱泄漏
        windowed_values = pixel_values * signal.windows.hann(len(pixel_values))
        
        # 计算FFT
        fft_result = np.abs(np.fft.rfft(windowed_values))
        freqs = np.fft.rfftfreq(len(pixel_values), d=1/analysis_fps)
        
        # 找到主要频率
        # 只考虑正频率部分（跳过直流分量）
        peak_idx = np.argmax(fft_result[1:]) + 1
        peak_freq = freqs[peak_idx]
        peak_amp = fft_result[peak_idx]
        
        # 找到所有显著峰值
        peaks, _ = signal.find_peaks(fft_result, height=0.2*peak_amp)
        peak_freqs = [(freqs[idx], fft_result[idx]) for idx in peaks if freqs[idx] > 0.1]
        peak_freqs.sort(key=lambda x: x[1], reverse=True)
        
        # 限制到最高5个峰值
        peak_freqs = peak_freqs[:5]
        
        # 准备建议频率范围
        suggested_ranges = []
        for freq, amp in peak_freqs:
            # 创建一个频率范围
            lower = max(0.1, freq - 0.2)
            upper = freq + 0.2
            
            suggested_ranges.append({
                'freq_min': float(lower),
                'freq_max': float(upper),
                'peak_freq': float(freq),
                'amplitude': float(amp / np.max(fft_result)),
                'description': f"峰值频率 {freq:.2f} Hz"
            })
        
        # 如果没有找到显著峰值，提供默认范围
        if len(suggested_ranges) == 0:
            suggested_ranges.append(FrequencyPresets.PRESETS['low'])
            
        # 寻找最佳预设匹配
        best_preset = None
        best_match_score = 0
        
        for name, preset in FrequencyPresets.PRESETS.items():
            for peak_freq, _ in peak_freqs:
                if preset['freq_min'] <= peak_freq <= preset['freq_max']:
                    # 越接近预设中间值，匹配度越高
                    preset_mid = (preset['freq_min'] + preset['freq_max']) / 2
                    match_score = 1 - abs(peak_freq - preset_mid) / (preset['freq_max'] - preset['freq_min'])
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_preset = name
        
        # 生成可视化
        visualization_paths = {}
        if generate_visualizations:
            print("正在生成可视化...")
            
            # 1. 时域信号
            time_path = viz.create_time_domain_visualization(
                np.arange(len(pixel_values)), 
                pixel_values, 
                analysis_fps, 
                output_dir, 
                base_name
            )
            visualization_paths['time_domain'] = time_path
            
            # 2. 频谱
            preset_data = FrequencyPresets.get_preset(best_preset) if best_preset else None
            spectrum_path = viz.create_frequency_spectrum_visualization(
                freqs, 
                fft_result, 
                peak_freqs, 
                best_preset, 
                preset_data, 
                max_freq, 
                output_dir, 
                base_name
            )
            visualization_paths['frequency_spectrum'] = spectrum_path
        
        # 准备分析结果
        analysis_data = {
            'video': {
                'path': video_path,
                'fps': float(fps),
                'dimensions': [width, height],
                'frame_count': n_frames
            },
            'analysis': {
                'sampling_rate': sampling_rate,
                'analysis_fps': float(analysis_fps),
                'analyzed_frames': len(sampled_frames),
                'peak_frequency': float(peak_freq),
                'best_matching_preset': best_preset
            },
            'suggested_ranges': suggested_ranges,
            'visualization_paths': visualization_paths,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存分析数据
        metadata_path = os.path.join(output_dir, f"{base_name}_frequency_analysis.json")
        with open(metadata_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\n分析结果已保存至: {metadata_path}")
        print("\n建议频率范围:")
        for i, range_info in enumerate(suggested_ranges):
            print(f"  {i+1}. {range_info['freq_min']:.2f}-{range_info['freq_max']:.2f} Hz "
                  f"(峰值: {range_info['peak_freq']:.2f} Hz)")
        
        if best_preset:
            print(f"\n最佳匹配预设: {best_preset} "
                  f"({FrequencyPresets.PRESETS[best_preset]['freq_min']:.1f}-"
                  f"{FrequencyPresets.PRESETS[best_preset]['freq_max']:.1f} Hz)")
        
        return {
            'metadata_path': metadata_path,
            'suggested_ranges': suggested_ranges,
            'visualization_paths': visualization_paths,
            'best_preset': best_preset
        }
    else:
        print("警告: 没有足够的数据进行频率分析")
        return None
