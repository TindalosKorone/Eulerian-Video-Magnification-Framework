import os
import sys
import cv2
import frequency_analyzer
from frequency_presets import FrequencyPresets
import cache_manager

def parse_roi(roi_str):
    """解析感兴趣区域字符串"""
    if not roi_str:
        return None
    
    try:
        roi = tuple(map(int, roi_str.split(',')))
        if len(roi) != 4:
            raise ValueError("ROI必须包含4个值：x,y,width,height")
        return roi
    except ValueError as e:
        print(f"错误: ROI格式无效 - {e}")
        sys.exit(1)

def get_output_dir(args, input_path):
    """确定输出目录"""
    if hasattr(args, 'output') and args.output:
        return os.path.dirname(os.path.abspath(args.output))
    elif input_path:
        return os.path.dirname(os.path.abspath(input_path))
    else:
        return os.getcwd()

def cmd_list_presets(args):
    """列出所有可用的频率预设"""
    FrequencyPresets.list_presets()

def cmd_visualize_preset(args):
    """可视化频率预设的响应"""
    # 获取视频帧率（如果提供了视频）
    fps = 30  # 默认值
    if args.video:
        try:
            vidcap = cv2.VideoCapture(args.video)
            if vidcap.isOpened():
                fps = vidcap.get(cv2.CAP_PROP_FPS)
                vidcap.release()
        except Exception as e:
            print(f"警告: 无法从视频获取帧率: {e}")
    
    # 确定输出目录
    output_dir = args.analysis_dir
    if not output_dir and args.video:
        output_dir = os.path.dirname(os.path.abspath(args.video))
    
    # 可视化预设
    frequency_analyzer.visualize_frequency_preset(
        args.preset,
        fps=fps,
        output_dir=output_dir
    )

def cmd_analyze(args):
    """分析视频频率"""
    # 初始化缓存管理器
    cm = cache_manager.CacheManager()
    
    # 解析ROI
    roi = parse_roi(args.roi)
    
    # 确定输出目录
    output_dir = get_output_dir(args, args.input)
    if args.analysis_dir:
        analysis_dir = args.analysis_dir
    else:
        analysis_dir = output_dir
    
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # 检查缓存的分析结果
    cached_analysis_path = None
    if not args.no_cache:
        analysis_params = {
            'sampling_rate': args.sampling_rate,
            'roi': roi
        }
        cached_analysis_path = cm.get_analysis_results_path(args.input, analysis_params)
    
    if cached_analysis_path and os.path.exists(cached_analysis_path):
        # 使用缓存的分析结果
        analysis_results = frequency_analyzer.load_analysis_results(cached_analysis_path)
        print(f"使用缓存的频率分析: {cached_analysis_path}")
        suggested_ranges = analysis_results.get('suggested_ranges', [])
        
        # 显示建议的频率范围
        if suggested_ranges:
            print("基于分析的建议频率范围:")
            for i, range_info in enumerate(suggested_ranges):
                desc = range_info.get('description', '')
                if 'peak_freq' in range_info:
                    print(f"  {i+1}. {range_info['freq_min']:.2f}-{range_info['freq_max']:.2f} Hz "
                          f"(峰值: {range_info['peak_freq']:.2f} Hz) - {desc}")
                else:
                    print(f"  {i+1}. {range_info['freq_min']:.2f}-{range_info['freq_max']:.2f} Hz - {desc}")
        
        # 显示最佳匹配预设
        best_preset = analysis_results.get('analysis', {}).get('best_matching_preset')
        if best_preset:
            preset_data = FrequencyPresets.get_preset(best_preset)
            print(f"\n最佳匹配预设: {best_preset}")
            print(f"  频率范围: {preset_data['freq_min']:.1f}-{preset_data['freq_max']:.1f} Hz")
            print(f"  放大系数: {preset_data['alpha']}")
            print(f"  描述: {preset_data['description']}")
    else:
        # 执行新的分析
        print("执行频率分析...")
        analysis_output = frequency_analyzer.analyze_video_frequencies(
            args.input, 
            output_dir=analysis_dir,
            sampling_rate=args.sampling_rate,
            region_of_interest=roi,
            preset_name=args.preset if hasattr(args, 'preset') else None,
            generate_visualizations=not args.skip_visualizations
        )
        
        # 缓存分析结果
        if analysis_output and 'metadata_path' in analysis_output:
            analysis_params = {
                'sampling_rate': args.sampling_rate,
                'roi': roi
            }
            cm.add_analysis_results(
                args.input, 
                analysis_output['metadata_path'],
                analysis_output.get('visualization_paths', {}),
                analysis_params
            )
    
    # 清理旧缓存文件
    cm.clean_cache()

def cmd_suggest(args):
    """分析视频并建议参数"""
    # 执行分析
    cmd_analyze(args)
    
    # 加载分析结果
    analysis_dir = args.analysis_dir or os.path.dirname(os.path.abspath(args.input))
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    metadata_path = os.path.join(analysis_dir, f"{base_name}_frequency_analysis.json")
    
    if os.path.exists(metadata_path):
        analysis_results = frequency_analyzer.load_analysis_results(metadata_path)
        if analysis_results:
            # 提取建议参数
            suggested_ranges = analysis_results.get('suggested_ranges', [])
            best_preset = analysis_results.get('analysis', {}).get('best_matching_preset')
            
            # 打印建议命令
            print("\n========== 建议命令 ==========")
            
            if best_preset:
                preset_data = FrequencyPresets.get_preset(best_preset)
                print(f"使用预设:")
                print(f"  python main.py magnify {args.input} output.mp4 --preset {best_preset}")
                print(f"  # 这会使用 {preset_data['freq_min']:.1f}-{preset_data['freq_max']:.1f} Hz, 放大系数={preset_data['alpha']}")
            
            if suggested_ranges:
                best_range = suggested_ranges[0]
                print("\n自定义频率:")
                print(f"  python main.py magnify {args.input} output.mp4 --freq-min {best_range['freq_min']:.2f} "
                      f"--freq-max {best_range['freq_max']:.2f} --amplify 10")
                
                if len(suggested_ranges) > 1:
                    print("\n多频段放大:")
                    bands_str = ",".join([f"{r['freq_min']:.2f}-{r['freq_max']:.2f}" for r in suggested_ranges[:3]])
                    print(f"  python main.py magnify {args.input} output.mp4 --freq-bands \"{bands_str}\" --amplify 10")
            
            print("\n优化建议:")
            print("  • 添加 --stabilize 以减少相机抖动")
            print("  • 添加 --adaptive --bilateral 以获得更好的边缘保持")
            print("  • 添加 --blur 0.5 --motion-threshold 0.02 以减少噪点")
            print("==========================")
