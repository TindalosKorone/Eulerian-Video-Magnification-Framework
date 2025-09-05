class FrequencyPresets:
    """预定义频率预设的类"""
    
    # 基本预设定义
    PRESETS = {
        # 基本频率范围预设
        'ultra-low': {'freq_min': 0.05, 'freq_max': 0.3, 'alpha': 15.0, 
                    'description': '极慢运动 (植物生长、结构变形)'},
        'low': {'freq_min': 0.3, 'freq_max': 3.0, 'alpha': 10.0, 
               'description': '呼吸和心率 (呼吸运动、心跳)'},
        'medium': {'freq_min': 3.0, 'freq_max': 10.0, 'alpha': 8.0, 
                  'description': '人体运动 (步行、日常动作)'},
        'med-high': {'freq_min': 10.0, 'freq_max': 30.0, 'alpha': 5.0, 
                    'description': '快速振动 (机械运动、电机低速运转)'},
        'high': {'freq_min': 30.0, 'freq_max': 60.0, 'alpha': 3.0, 
                'description': '电机工频 (家电、风扇、电动机)'},
        'ultra-high': {'freq_min': 60.0, 'freq_max': 120.0, 'alpha': 2.0, 
                      'description': '高速机械 (精密设备、高速运转部件)'},
        
        # 特定应用预设
        'breathing': {'freq_min': 0.2, 'freq_max': 0.7, 'alpha': 12.0, 
                     'description': '人体呼吸专用频率'},
        'pulse': {'freq_min': 0.8, 'freq_max': 2.5, 'alpha': 15.0, 
                 'description': '人体心率专用频率'},
        'motor': {'freq_min': 45.0, 'freq_max': 55.0, 'alpha': 3.0, 
                 'description': '电机工频专用频率'},
    }
    
    @classmethod
    def get_preset(cls, name):
        """获取预设参数"""
        return cls.PRESETS.get(name.lower(), cls.PRESETS['medium'])
    
    @classmethod
    def list_presets(cls):
        """列出所有可用预设"""
        print("\n=== 可用频率预设 ===")
        for name, data in sorted(cls.PRESETS.items()):
            print(f"{name:10}: {data['freq_min']:.1f}-{data['freq_max']:.1f} Hz - {data['description']}")
        print("")
    
    @classmethod
    def parse_frequency_bands(cls, bands_str):
        """解析用户指定的多频段字符串 (如 "0.3-3.0,45.0-55.0")"""
        bands = []
        for band_str in bands_str.split(','):
            try:
                low, high = map(float, band_str.split('-'))
                bands.append({'freq_min': low, 'freq_max': high})
            except ValueError:
                continue
        return bands if bands else [cls.PRESETS['medium']]
