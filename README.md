# 欧拉视频运动放大框架 (Eulerian Video Magnification Framework)

一个强大的视频运动放大工具，能够放大人眼难以察觉的微小运动。

## 特性

- **强大的线性放大**：基于拉普拉斯金字塔的欧拉运动放大
- **预设模式**：提供针对不同场景优化的参数预设
- **视频稳定**：内置视频稳定功能，减少不相关运动干扰
- **高级降噪**：多种噪点抑制机制，包括空间模糊和运动阈值控制
- **自适应放大**：根据不同尺度的结构智能调整放大强度
- **高效处理**：支持分块处理和GPU加速，高效处理大型视频

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
```


## 参数说明

### 基本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入视频路径 | (必需) |
| `output` | 输出视频路径 | (必需) |
| `--amplify` | 运动放大系数 | 10.0 |
| `--preset` | 使用预定义参数集 | (无) |

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
| `--adaptive` | 使用自适应放大（对大结构放大更强） | False |

## 提示与技巧

1. **找到合适的频率范围**：针对不同类型的运动使用不同的频率范围：
   - 脉搏/心跳：0.8-2.0 Hz
   - 呼吸：0.1-0.5 Hz

2. **减少噪点**：
   - 增加 `--blur` 值（例如0.5-1.0）
   - 设置 `--motion-threshold` 滤除微弱运动（例如0.01-0.05）
   - 使用 `--adaptive` 更好地保留细节

3. **处理大视频**：
   - 增加 `--chunk-size` 提高处理速度（如果内存允许）
   - 减小 `--overlap` 减少内存使用

4. **处理抖动视频**：
   - 使用 `--stabilize` 参数
   - 可能需要牺牲一些边缘区域

## 注意事项

- 处理高分辨率视频需要较大内存，如遇内存问题，尝试减小 `--chunk-size`
- 某些参数组合可能导致放大结果中出现伪影，需要针对具体视频内容进行调整
- 自适应放大模式在保留细节的同时放大大尺度运动效果更佳

## 项目结构

- `main.py` - 主程序入口和参数解析
- `motion_amplifier.py` - 线性欧拉放大法实现
- `pyramid.py` - 拉普拉斯金字塔实现
- `temporal_filter.py` - 时域滤波实现
- `video_handler.py` - 视频I/O处理
- `stabilizer.py` - 视频稳定功能

## 许可证

[待添加]
