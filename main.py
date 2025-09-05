import argparse
import torch
import cv2
import video_handler as vh
import motion_amplifier as ma
from tqdm import tqdm
import gc
import os
import tempfile
import stabilizer
import frequency_analyzer
import cache_manager
import sys
import analysis_commands

def setup_common_args(parser):
    """添加所有子命令共享的参数"""
    # 缓存相关参数
    parser.add_argument('--no-cache', action='store_true',
                        help='禁用缓存（默认启用）')
    # 分析相关参数
    parser.add_argument('--analysis-dir', type=str, default=None,
                        help='分析结果保存目录（默认：与输出在同一目录）')
    parser.add_argument('--sampling-rate', type=float, default=0.5,
                        help='分析采样率 (0.0-1.0)，较低的值可加快分析速度')
    parser.add_argument('--roi', type=str, metavar='X,Y,WIDTH,HEIGHT',
                        help='感兴趣区域 (格式: "100,100,200,200")')
    parser.add_argument('--skip-visualizations', action='store_true',
                        help='跳过生成可视化图像以加快分析速度')

def setup_magnify_args(parser):
    """添加视频放大相关参数"""
    # 基本参数
    parser.add_argument('--preset', type=str,
                        help='使用预设频率范围（运行 list-presets 命令查看选项）')
    parser.add_argument('--amplify', type=float, default=10.0,
                        help='运动放大系数')
    
    # 滤波参数
    filter_group = parser.add_argument_group('滤波参数')
    filter_group.add_argument('--freq-min', type=float, default=0.4,
                            help='最小放大频率 (Hz)')
    filter_group.add_argument('--freq-max', type=float, default=3.0,
                            help='最大放大频率 (Hz)')
    filter_group.add_argument('--freq-bands', type=str, 
                            help='多频段放大 (格式: "0.3-3.0,45.0-55.0")')
    filter_group.add_argument('--blur', type=float, default=0,
                            help='空间模糊强度 (0表示禁用)')
    filter_group.add_argument('--motion-threshold', type=float, default=0,
                            help='最小运动幅度阈值 (0表示禁用)')
    
    # 金字塔参数
    parser.add_argument('--levels', type=int, default=3,
                        help='金字塔层数')
    
    # 处理参数
    processing_group = parser.add_argument_group('处理参数')
    processing_group.add_argument('--chunk-size', type=int, default=20,
                                help='一次处理的帧数')
    processing_group.add_argument('--overlap', type=int, default=8,
                                help='数据块之间的重叠帧数')
    
    # 增强选项
    enhancement_group = parser.add_argument_group('增强选项')
    enhancement_group.add_argument('--stabilize', action='store_true',
                                help='放大前执行视频稳定')
    enhancement_group.add_argument('--stabilize-radius', type=int, default=None,
                                help='稳定化平滑半径（帧数）（默认：等于帧率）')
    enhancement_group.add_argument('--stabilize-strength', type=float, default=0.95,
                                help='稳定化强度 (0.0-1.0)')
    enhancement_group.add_argument('--adaptive', action='store_true',
                                help='使用自适应放大（对大结构放大更强）')
    enhancement_group.add_argument('--bilateral', action='store_true',
                                help='使用双边滤波（更好地保留边缘）')
    enhancement_group.add_argument('--color-stabilize', action='store_true',
                                help='稳定颜色减少闪烁')
    enhancement_group.add_argument('--multiband', action='store_true',
                                help='多频段处理频率范围')
    enhancement_group.add_argument('--keep-temp', action='store_true',
                                help='保留临时文件（如稳定化视频）')

def stabilize_video_with_caching(args, input_path, output_dir, cm):
    """稳定视频，支持缓存"""
    # 检查缓存的稳定化视频
    cached_stabilized_path = None
    if not args.no_cache:
        stabilize_params = {
            'radius': args.stabilize_radius,
            'strength': args.stabilize_strength
        }
        cached_stabilized_path = cm.get_stabilized_video_path(input_path, stabilize_params)
    
    if cached_stabilized_path and os.path.exists(cached_stabilized_path):
        # 使用缓存的稳定化视频
        print(f"使用缓存的稳定化视频: {cached_stabilized_path}")
        return cached_stabilized_path, cached_stabilized_path
    
    # 执行稳定化
    print("正在稳定视频，这可能需要一些时间...")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成稳定化视频的文件名
    input_filename = os.path.basename(input_path)
    input_name, input_ext = os.path.splitext(input_filename)
    temp_stabilized_path = os.path.join(output_dir, f"{input_name}_stabilized{input_ext}")
    
    # 执行稳定化
    stabilizer.stabilize_video(
        input_path, 
        temp_stabilized_path, 
        smoothing_radius=args.stabilize_radius,
        smoothing_strength=args.stabilize_strength
    )
    print(f"稳定化视频已保存至: {temp_stabilized_path}")
    
    # 缓存稳定化视频
    if not args.no_cache:
        stabilize_params = {
            'radius': args.stabilize_radius,
            'strength': args.stabilize_strength
        }
        cm.add_stabilized_video(input_path, temp_stabilized_path, stabilize_params)
    
    return temp_stabilized_path, None

def cmd_magnify(args):
    """放大视频中的运动"""
    # 初始化缓存管理器
    cm = cache_manager.CacheManager()
    
    # 应用预设（如果指定）
    if args.preset:
        preset_data = frequency_analyzer.FrequencyPresets.get_preset(args.preset)
        args.freq_min = preset_data['freq_min']
        args.freq_max = preset_data['freq_max']
        args.amplify = preset_data['alpha']
        print(f"使用 '{args.preset}' 预设: {args.freq_min:.1f}-{args.freq_max:.1f} Hz, "
              f"放大系数={args.amplify}, {preset_data['description']}")
    
    # 参数验证
    if args.overlap >= args.chunk_size:
        print("错误: 重叠帧数必须小于数据块大小")
        sys.exit(1)
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置输入输出路径
    input_video_path = args.input
    temp_stabilized_path = None
    cached_stabilized_path = None
    
    # 确定输出目录
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 如果使用已有分析结果
    if args.use_analysis:
        if os.path.exists(args.use_analysis):
            analysis_results = frequency_analyzer.load_analysis_results(args.use_analysis)
            if analysis_results:
                print(f"已加载分析结果: {args.use_analysis}")
                # 可以从分析结果中提取建议的参数
                # 这里只是打印建议，但不自动应用，因为用户可能已经明确指定了参数
                suggested_ranges = analysis_results.get('suggested_ranges', [])
                
                if suggested_ranges:
                    print("基于分析的建议频率范围:")
                    for i, range_info in enumerate(suggested_ranges):
                        desc = range_info.get('description', '')
                        if 'peak_freq' in range_info:
                            print(f"  {i+1}. {range_info['freq_min']:.2f}-{range_info['freq_max']:.2f} Hz "
                                  f"(峰值: {range_info['peak_freq']:.2f} Hz) - {desc}")
                        else:
                            print(f"  {i+1}. {range_info['freq_min']:.2f}-{range_info['freq_max']:.2f} Hz - {desc}")
                
                best_preset = analysis_results.get('analysis', {}).get('best_matching_preset')
                if best_preset:
                    preset_data = frequency_analyzer.FrequencyPresets.get_preset(best_preset)
                    print(f"\n最佳匹配预设: {best_preset}")
                    print(f"  频率范围: {preset_data['freq_min']:.1f}-{preset_data['freq_max']:.1f} Hz")
                    print(f"  放大系数: {preset_data['alpha']}")
            else:
                print(f"错误: 无法加载分析结果 {args.use_analysis}")
                sys.exit(1)
        else:
            print(f"错误: 分析文件 {args.use_analysis} 不存在")
            sys.exit(1)
    
    # 稳定化（如果启用）
    if args.stabilize:
        temp_stabilized_path, cached_stabilized_path = stabilize_video_with_caching(
            args, input_video_path, output_dir, cm
        )
        input_video_path = temp_stabilized_path
    
    # 处理频率段
    freq_bands = None
    if args.freq_bands:
        freq_bands = frequency_analyzer.FrequencyPresets.parse_frequency_bands(args.freq_bands)
        if freq_bands:
            print(f"使用多个频率段:")
            for i, band in enumerate(freq_bands):
                print(f"  段 {i+1}: {band['freq_min']:.1f}-{band['freq_max']:.1f} Hz")
        else:
            print("警告: 无法解析频率段，使用默认范围")
    
    # 打开视频并获取属性
    vidcap = cv2.VideoCapture(input_video_path)
    if not vidcap.isOpened():
        print(f"错误: 无法打开视频 {input_video_path}")
        sys.exit(1)
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 设置视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # 处理视频
    frames_buffer = []
    processed_frames_count = 0
    
    with tqdm(total=frame_count, desc="处理视频") as pbar:
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            frames_buffer.append(frame)
            
            if len(frames_buffer) == args.chunk_size:
                # 将帧转换为张量
                tensor = vh.frames_to_tensor(frames_buffer, device)
                
                # 处理多个频率段或单个频率范围
                if freq_bands:
                    # 从原始张量开始
                    result_tensor = tensor.clone()
                    
                    # 处理每个频率段并添加结果
                    for band in freq_bands:
                        band_result = ma.magnify_motion(
                            tensor, fps, band['freq_min'], band['freq_max'], 
                            args.amplify, args.levels, args.blur,
                            args.motion_threshold, args.adaptive,
                            args.bilateral, args.color_stabilize, args.multiband
                        )
                        # 添加运动分量
                        result_tensor = result_tensor + (band_result - tensor)
                    
                    # 确保值在有效范围内
                    result_tensor = torch.clamp(result_tensor, 0, 1)
                else:
                    # 处理单个频率范围
                    result_tensor = ma.magnify_motion(
                        tensor, fps, args.freq_min, args.freq_max, 
                        args.amplify, args.levels, args.blur,
                        args.motion_threshold, args.adaptive,
                        args.bilateral, args.color_stabilize, args.multiband
                    )
                
                # 将张量转换回帧
                result_frames = vh.tensor_to_frames(result_tensor)
                
                # 确定要写入的帧，考虑重叠
                start_index = args.overlap // 2 if processed_frames_count > 0 else 0
                end_index = args.chunk_size - (args.overlap // 2)
                
                for i in range(start_index, end_index):
                    out.write(result_frames[i])
                
                pbar.update(end_index - start_index)
                processed_frames_count += (end_index - start_index)
                
                # 滑动缓冲区
                frames_buffer = frames_buffer[args.chunk_size - args.overlap:]
                
                # 清理内存
                del tensor, result_tensor, result_frames
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                else:
                    gc.collect()
        
        # 处理剩余的帧
        if len(frames_buffer) > 0:
            # 如果缓冲区小于数据块大小，则填充
            while len(frames_buffer) < args.chunk_size and len(frames_buffer) > 0:
                frames_buffer.append(frames_buffer[-1])  # 通过重复最后一帧进行简单填充
            
            tensor = vh.frames_to_tensor(frames_buffer, device)
            
            # 处理单个或多个频率段
            if freq_bands:
                # 从原始张量开始
                result_tensor = tensor.clone()
                
                # 处理每个频率段并添加结果
                for band in freq_bands:
                    band_result = ma.magnify_motion(
                        tensor, fps, band['freq_min'], band['freq_max'], 
                        args.amplify, args.levels, args.blur,
                        args.motion_threshold, args.adaptive,
                        args.bilateral, args.color_stabilize, args.multiband
                    )
                    # 添加运动分量
                    result_tensor = result_tensor + (band_result - tensor)
                
                # 确保值在有效范围内
                result_tensor = torch.clamp(result_tensor, 0, 1)
            else:
                # 处理单个频率范围
                result_tensor = ma.magnify_motion(
                    tensor, fps, args.freq_min, args.freq_max, 
                    args.amplify, args.levels, args.blur,
                    args.motion_threshold, args.adaptive,
                    args.bilateral, args.color_stabilize, args.multiband
                )
            
            result_frames = vh.tensor_to_frames(result_tensor)
            
            remaining_to_write = frame_count - processed_frames_count
            start_index = args.overlap // 2
            
            for i in range(start_index, min(start_index + remaining_to_write, len(result_frames))):
                out.write(result_frames[i])
            
            pbar.update(min(start_index + remaining_to_write, len(result_frames)) - start_index)
            
            # 清理内存
            del tensor, result_tensor, result_frames
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            else:
                gc.collect()
    
    vidcap.release()
    out.release()
    
    # 处理临时文件
    if temp_stabilized_path and not args.keep_temp and temp_stabilized_path != cached_stabilized_path:
        print(f"清理临时文件: {temp_stabilized_path}")
        try:
            os.remove(temp_stabilized_path)
        except:
            print(f"警告: 无法删除临时文件: {temp_stabilized_path}")
    elif temp_stabilized_path and args.keep_temp:
        print(f"临时稳定化视频保存在: {temp_stabilized_path}")
    
    # 定期清理旧缓存文件
    cm.clean_cache()
    
    print(f"视频处理完成! 输出已保存至: {args.output}")

def main():
    # 创建主解析器
    parser = argparse.ArgumentParser(
        description='欧拉视频运动放大框架',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 1. magnify - 放大视频中的运动
    magnify_parser = subparsers.add_parser('magnify', 
                                          help='放大视频中的运动',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    magnify_parser.add_argument('input', type=str, help='输入视频路径')
    magnify_parser.add_argument('output', type=str, help='输出视频路径')
    magnify_parser.add_argument('--use-analysis', type=str, metavar='PATH',
                               help='使用指定JSON文件中的分析结果')
    setup_common_args(magnify_parser)
    setup_magnify_args(magnify_parser)
    magnify_parser.set_defaults(func=cmd_magnify)
    
    # 2. analyze - 分析视频频率
    analyze_parser = subparsers.add_parser('analyze', 
                                          help='分析视频中的运动频率',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    analyze_parser.add_argument('input', type=str, help='输入视频路径')
    analyze_parser.add_argument('--preset', type=str, help='可选：使用预设频率范围进行显示')
    setup_common_args(analyze_parser)
    analyze_parser.set_defaults(func=analysis_commands.cmd_analyze)
    
    # 3. suggest - 分析视频并建议参数
    suggest_parser = subparsers.add_parser('suggest', 
                                         help='分析视频并建议放大参数',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    suggest_parser.add_argument('input', type=str, help='输入视频路径')
    setup_common_args(suggest_parser)
    suggest_parser.set_defaults(func=analysis_commands.cmd_suggest)
    
    # 4. list-presets - 列出所有频率预设
    list_presets_parser = subparsers.add_parser('list-presets', 
                                              help='列出所有可用的频率预设',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    list_presets_parser.set_defaults(func=analysis_commands.cmd_list_presets)
    
    # 5. visualize-preset - 可视化频率预设
    visualize_parser = subparsers.add_parser('visualize-preset', 
                                           help='可视化频率预设响应',
                                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    visualize_parser.add_argument('preset', type=str, help='要可视化的预设名称')
    visualize_parser.add_argument('--video', type=str, help='可选：用于获取帧率的视频文件')
    visualize_parser.add_argument('--analysis-dir', type=str, help='可视化图像保存目录')
    visualize_parser.set_defaults(func=analysis_commands.cmd_visualize_preset)
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查是否提供了命令
    if not hasattr(args, 'func'):
        # 如果没有提供子命令，显示帮助信息
        parser.print_help()
        print("\n使用示例:")
        print("  # 列出所有可用预设:")
        print("  python main.py list-presets")
        print("\n  # 放大视频中的运动:")
        print("  python main.py magnify input.mp4 output.mp4 --preset pulse")
        print("\n  # 分析视频并建议参数:")
        print("  python main.py suggest input.mp4")
        return
    
    # 执行对应的命令函数
    args.func(args)

if __name__ == "__main__":
    main()
