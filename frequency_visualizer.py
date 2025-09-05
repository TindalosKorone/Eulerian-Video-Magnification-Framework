import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_frequency_preset(preset_name, preset_data, fps=30, output_dir=None):
    """
    可视化预设的频率响应。
    
    Parameters:
    -----------
    preset_name : str
        预设名称
    preset_data : dict
        预设数据，包含freq_min和freq_max
    fps : float, optional
        帧率，用于计算频率响应
    output_dir : str, optional
        输出目录
    """
    # 创建频率范围
    freqs = np.linspace(0, fps/2, 1000)
    
    # 计算巴特沃斯滤波器响应 (5阶)
    order = 5
    low_mask = 1 / (1 + (freqs / preset_data['freq_max'])**(2 * order))
    high_mask = 1 - 1 / (1 + (freqs / preset_data['freq_min'])**(2 * order))
    response = low_mask * high_mask
    
    # 绘制频率响应
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, response)
    plt.title(f"频率预设: {preset_name} ({preset_data['freq_min']:.1f}-{preset_data['freq_max']:.1f} Hz)")
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅度响应')
    plt.grid(True)
    plt.axvline(x=preset_data['freq_min'], color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=preset_data['freq_max'], color='r', linestyle='--', alpha=0.5)
    plt.fill_between(freqs, 0, response, alpha=0.2)
    
    # 添加注释
    plt.annotate(f'低频截止: {preset_data["freq_min"]:.1f} Hz', 
                 xy=(preset_data['freq_min'], 0.5), xytext=(preset_data['freq_min']+0.5, 0.6),
                 arrowprops=dict(arrowstyle='->'))
    plt.annotate(f'高频截止: {preset_data["freq_max"]:.1f} Hz', 
                 xy=(preset_data['freq_max'], 0.5), xytext=(preset_data['freq_max']-0.5, 0.6),
                 arrowprops=dict(arrowstyle='->'))
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"preset_{preset_name}.png"), dpi=150)
        print(f"预设可视化已保存至: {output_dir}/preset_{preset_name}.png")
    
    plt.show()

def create_time_domain_visualization(time_values, pixel_values, analysis_fps, output_dir, base_name):
    """
    创建时域信号可视化
    
    Parameters:
    -----------
    time_values : array
        时间值数组
    pixel_values : array
        像素值数组
    analysis_fps : float
        分析帧率
    output_dir : str
        输出目录
    base_name : str
        基本文件名
        
    Returns:
    --------
    str
        可视化文件路径
    """
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(pixel_values)) / analysis_fps, pixel_values)
    plt.xlabel('时间 (秒)')
    plt.ylabel('平均运动幅度')
    plt.title('运动幅度时域信号')
    plt.grid(True)
    
    time_path = os.path.join(output_dir, f"{base_name}_time_domain.png")
    plt.savefig(time_path, dpi=150)
    plt.close()
    
    return time_path

def create_frequency_spectrum_visualization(freqs, fft_result, peak_freqs, best_preset, 
                                          preset_data, max_freq, output_dir, base_name):
    """
    创建频谱可视化
    
    Parameters:
    -----------
    freqs : array
        频率数组
    fft_result : array
        FFT结果数组
    peak_freqs : list
        峰值频率列表，每个元素是(freq, amp)元组
    best_preset : str
        最佳匹配预设名称
    preset_data : dict
        预设数据，包含freq_min和freq_max
    max_freq : float
        最大显示频率
    output_dir : str
        输出目录
    base_name : str
        基本文件名
        
    Returns:
    --------
    str
        可视化文件路径
    """
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, fft_result)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅度')
    plt.title('运动频谱')
    plt.grid(True)
    plt.xlim(0, min(max_freq, freqs[-1]))
    
    # 标记峰值
    for freq, amp in peak_freqs:
        plt.plot(freq, amp, 'ro')
        plt.annotate(f"{freq:.2f} Hz", 
                     xy=(freq, amp), 
                     xytext=(freq+0.1, amp*1.1),
                     arrowprops=dict(arrowstyle='->'))
    
    # 如果有最佳预设，显示其频率范围
    if best_preset and preset_data:
        plt.axvspan(preset_data['freq_min'], preset_data['freq_max'], alpha=0.2, color='green')
        plt.annotate(f"预设: {best_preset}", 
                    xy=((preset_data['freq_min'] + preset_data['freq_max'])/2, max(fft_result)/2),
                    ha='center')
    
    spectrum_path = os.path.join(output_dir, f"{base_name}_frequency_spectrum.png")
    plt.savefig(spectrum_path, dpi=150)
    plt.close()
    
    return spectrum_path
