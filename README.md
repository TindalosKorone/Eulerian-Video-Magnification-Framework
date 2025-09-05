# 欧拉视频运动放大框架

> 放大人眼难以察觉的微小运动，实现视觉增强。

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/yourusername/Eulerian-Video-Magnification.git
cd Eulerian-Video-Magnification
pip install -r requirements.txt
```
> **注意**：pytorch需单独安装。CUDA支持请参考 [官方教程](https://pytorch.org/get-started/locally/)

### 核心命令

```bash
# 分析视频中的运动频率
python main.py analyze input.mp4

# 使用预设放大视频中的运动
python main.py magnify input.mp4 output.mp4 --preset pulse

# 获取最佳放大参数建议
python main.py suggest input.mp4
```

## 📋 命令说明

命令 | 功能 | 重要说明
---|---|---
`magnify` | 放大视频中的运动 | 使用预设或自定义参数进行带通滤波
`analyze` | 分析视频中的频率 | 基于视频内容，不受预设限制
`suggest` | 建议最佳放大参数 | 结合分析和预设建议
`list-presets` | 列出所有频率预设 | 显示可用预设和参数
`visualize-preset` | 可视化预设响应 | 生成频率响应曲线

### ⚙️ 常用示例

#### 分析命令

```bash
# 分析视频频率（结果基于视频内容，不受预设限制）
python main.py analyze input.mp4

# 分析特定区域
python main.py analyze input.mp4 --roi "100,100,200,200"

# 获取放大建议
python main.py suggest input.mp4
```

#### 放大命令

```bash
# 放大心跳 (0.8-2.5 Hz)
python main.py magnify input.mp4 output.mp4 --preset pulse

# 放大呼吸 (0.2-0.7 Hz)
python main.py magnify input.mp4 output.mp4 --preset breathing

# 放大电机工频 (45.0-55.0 Hz)
python main.py magnify input.mp4 output.mp4 --preset motor

# 自定义频率和放大系数
python main.py magnify input.mp4 output.mp4 --freq-min 0.8 --freq-max 2.0 --amplify 15

# 多频段同时放大
python main.py magnify input.mp4 output.mp4 --freq-bands "0.3-3.0,45.0-55.0"
```

## 🛠️ 核心参数

参数 | 说明 | 默认值
---|---|---
`--preset` | 预设参数 (pulse/breathing/motor 等) | 无
`--amplify` | 运动放大系数 | 10.0
`--freq-min` | 最小频率 (Hz) | 0.4
`--freq-max` | 最大频率 (Hz) | 3.0
`--freq-bands` | 多频段 ("0.3-3.0,45.0-55.0") | 无
`--stabilize` | 视频稳定 | False
`--blur` | 空间模糊 (0=禁用) | 0
`--roi` | 感兴趣区域 ("x,y,width,height") | 整个画面

<details>
<summary>查看更多参数</summary>

### 增强参数

参数 | 说明 | 默认值
---|---|---
`--levels` | 金字塔层数 | 3
`--chunk-size` | 一次处理帧数 | 20
`--overlap` | 数据块重叠帧数 | 8
`--adaptive` | 自适应放大 | False
`--bilateral` | 双边滤波 | False
`--color-stabilize` | 颜色稳定 | False
`--multiband` | 多频段处理 | False
`--sampling-rate` | 分析采样率 (0.0-1.0) | 0.5
`--no-cache` | 禁用缓存 | False

</details>

## 💡 使用技巧

### 关键概念

- **两种主要功能**：
  - **分析**：检测视频中实际存在的频率，不受预设限制
  - **放大**：使用指定频率范围的带通滤波器放大运动

### 推荐工作流程

1. **分析** → 了解视频中存在的频率：`analyze`
2. **选择** → 根据分析结果选择合适的预设或参数
3. **放大** → 使用所选参数放大视频：`magnify`

### 常见场景参数

目标 | 推荐预设 | 频率范围
---|---|---
心跳/脉搏 | `pulse` | 0.8-2.5 Hz
呼吸 | `breathing` | 0.2-0.7 Hz
电机工频 | `motor` | 45.0-55.0 Hz
极慢运动 | `ultra-low` | 0.05-0.3 Hz
高频振动 | `med-high` | 10.0-30.0 Hz

### 优化建议

- **减少噪点**：`--blur 0.5 --motion-threshold 0.02`
- **处理抖动**：`--stabilize`
- **保留边缘**：`--bilateral`
- **大型视频**：降低 `--sampling-rate` 和 `--overlap`
- **高频率视频**：确保帧率至少是目标频率的两倍

## 📂 项目结构

- `main.py` - 主程序入口
- `analysis_commands.py` - 分析命令实现
- `motion_amplifier.py` - 欧拉放大实现
- `frequency_analyzer.py` - 频率分析
- `frequency_presets.py` - 预设定义
- `cache_manager.py` - 缓存管理
- `TEMP/` - 缓存和临时文件
- `Image/` - 分析结果和可视化图像
