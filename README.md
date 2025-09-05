# 欧拉视频运动放大框架 (Eulerian Video Magnification Framework)

一个强大的视频运动放大工具，能够放大人眼难以察觉的微小运动。

## 特性

- **强大的线性放大**：基于拉普拉斯金字塔的欧拉运动放大
- **预设模式**：提供针对不同场景优化的参数预设
- **视频稳定**：内置视频稳定功能，减少不相关运动干扰
- **高级降噪**：多种噪点抑制机制，包括空间模糊和运动阈值控制
- **自适应放大**：根据不同尺度的结构智能调整放大强度
- **高效处理**：支持分块处理和GPU加速，高效处理大型视频
- **频率分析**：内置频率分析工具，生成热点图帮助选择最佳频率范围
- **智能缓存**：自动缓存中间结果，加速重复处理和参数调优
- **性能优化**：提供多种性能优化选项，减少内存占用和处理时间

## 安装

1. 克隆此仓库：

```bash
git clone https://github.com/yourusername/Eulerian-Video-Magnification.git
cd Eulerian-Video-Magnification
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

最简单的命令只需指定输入和输出视频路径：

```bash
python main.py input.mp4 output.mp4
```

### 频率分析

在放大前分析视频中的运动频率，以便选择最佳参数：

```bash
# 仅执行分析
python main.py input.mp4 --analyze-only

# 提示最佳频率参数
python main.py input.mp4 --suggest-params

# 使用已有分析结果
python main.py input.mp4 output.mp4 --use-analysis path/to/analysis.json
```

### 使用预设模式

为常见场景提供了优化的参数预设：

```bash
# 放大脉搏/心跳
python main.py input.mp4 output.mp4 --preset pulse

# 放大呼吸运动
python main.py input.mp4 output.mp4 --preset breathing

# 中等强度放大
python main.py input.mp4 output.mp4 --preset medium

# 极端强度放大
python main.py input.mp4 output.mp4 --preset extreme

# 微妙/轻柔放大
python main.py input.mp4 output.mp4 --preset subtle
```

### 自定义参数

可以精确控制放大过程的各个方面：

```bash
python main.py input.mp4 output.mp4 --amplify 15 --freq-min 0.8 --freq-max 2.0
```

### 高级降噪选项

使用高级降噪选项减少伪影：

```bash
python main.py input.mp4 output.mp4 --blur 0.5 --motion-threshold 0.02 --adaptive
```

### 视频稳定

对于包含相机抖动的视频，可以启用稳定功能：

```bash
python main.py input.mp4 output.mp4 --stabilize

# 自定义稳定参数
python main.py input.mp4 output.mp4 --stabilize --stabilize-radius 15 --stabilize-strength 0.8
```

### 性能优化

处理大型视频或在低配置硬件上运行时，使用性能优化选项：

```bash
# 降低频率分析时的采样率和跟踪点数量
python main.py input.mp4 --analyze-only --sampling-rate 0.2 --max-points 100

# 减少频率热图内存占用
python main.py input.mp4 --analyze-only --downsample-factor 2 --max-freq-bands 10 --skip-visualizations

# 禁用缓存功能（默认启用）
python main.py input.mp4 output.mp4 --no-cache
```

## 参数说明

### 基本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入视频路径 | (必需) |
| `output` | 输出视频路径 | (必需) |
| `--amplify` | 运动放大系数 | 10.0 |
| `--preset` | 使用预定义参数集 | (无) |
| `--no-cache` | 禁用缓存功能 | False |

### 滤波参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--freq-min` | 最小放大频率 (Hz) | 0.4 |
| `--freq-max` | 最大放大频率 (Hz) | 3.0 |
| `--blur` | 空间模糊强度（0表示禁用） | 0 |
| `--motion-threshold` | 最小运动幅度阈值（0表示禁用） | 0 |

### 金字塔参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--levels` | 拉普拉斯金字塔层数 | 3 |

### 处理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--chunk-size` | 一次处理的帧数 | 20 |
| `--overlap` | 数据块之间的重叠帧数 | 8 |

### 增强选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--stabilize` | 放大前执行视频稳定 | False |
| `--stabilize-radius` | 稳定化平滑半径（帧数） | 视频帧率 |
| `--stabilize-strength` | 稳定化强度 (0.0-1.0) | 0.95 |
| `--adaptive` | 使用自适应放大（对大结构放大更强） | False |
| `--bilateral` | 使用双边滤波（更好地保留边缘） | False |
| `--color-stabilize` | 稳定颜色减少闪烁 | False |
| `--multiband` | 多频段处理频率范围 | False |
| `--keep-temp` | 保留临时文件 | False |

### 分析参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--analyze-only` | 仅执行频率分析而不放大 | False |
| `--suggest-params` | 根据分析建议最佳参数并退出 | False |
| `--use-analysis` | 使用指定的分析结果文件 | (无) |
| `--analysis-dir` | 保存分析结果的目录 | 输出目录 |
| `--sampling-rate` | 分析时采样的帧比例 (0.0-1.0) | 0.5 |
| `--max-points` | 每帧最大跟踪点数量 | 200 |
| `--downsample-factor` | 频率热图的空间下采样因子 | 1 |
| `--max-freq-bands` | 最大频率带数量 | 20 |
| `--skip-visualizations` | 跳过生成可视化图像 | False |

## 提示与技巧

1. **找到合适的频率范围**：针对不同类型的运动使用不同的频率范围：
   - 脉搏/心跳：0.8-2.0 Hz
   - 呼吸：0.1-0.5 Hz
   - **使用频率分析**：使用 `--analyze-only` 或 `--suggest-params` 自动找出最佳频率

2. **减少噪点**：
   - 增加 `--blur` 值（例如0.5-1.0）
   - 设置 `--motion-threshold` 滤除微弱运动（例如0.01-0.05）
   - 使用 `--adaptive` 更好地保留细节
   - 尝试 `--bilateral` 在减少噪点的同时保留边缘

3. **处理大视频**：
   - 增加 `--chunk-size` 提高处理速度（如果内存允许）
   - 减小 `--overlap` 减少内存使用
   - 分析时使用 `--downsample-factor` 和 `--max-freq-bands` 减少内存占用

4. **处理抖动视频**：
   - 使用 `--stabilize` 参数
   - 调整 `--stabilize-radius` 和 `--stabilize-strength` 获得最佳效果
   - 使用 `--skip-existing` 避免重复处理已稳定的视频

5. **加速重复处理**：
   - 默认启用缓存系统自动保存中间结果
   - 如果需要禁用缓存使用 `--no-cache` 参数
   - 所有缓存文件存储在 `TEMP` 目录，可通过 `.gitignore` 忽略

## 注意事项

- 处理高分辨率视频需要较大内存，如遇内存问题，尝试减小 `--chunk-size` 或使用性能优化选项
- 某些参数组合可能导致放大结果中出现伪影，需要针对具体视频内容进行调整
- 自适应放大模式在保留细节的同时放大大尺度运动效果更佳
- 频率分析可能需要较长时间，尤其是对于长视频，可使用 `--sampling-rate` 加速
- 缓存文件会随时间增长，系统会自动清理30天以上的旧缓存

## 项目结构

- `main.py` - 主程序入口和参数解析
- `motion_amplifier.py` - 线性欧拉放大法实现
- `pyramid.py` - 拉普拉斯金字塔实现
- `temporal_filter.py` - 时域滤波实现
- `video_handler.py` - 视频I/O处理
- `stabilizer.py` - 视频稳定功能
- `frequency_analyzer.py` - 视频频率分析模块
- `cache_manager.py` - 缓存管理系统
- `TEMP/` - 存放缓存和临时文件

## 许可证

[待添加]
