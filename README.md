# 欧拉视频运动放大框架 (Eulerian Video Magnification Framework)

一个强大的视频运动放大工具，能够放大人眼难以察觉的微小运动。
**本工具使用AI生成**

## 快速开始

### 安装

```bash
git clone https://github.com/yourusername/Eulerian-Video-Magnification.git
cd Eulerian-Video-Magnification
pip install -r requirements.txt
```
**注意，pytorch需要根据本身的环境进行安装，默认情况下只安装CPU运行部分，如需使用cuda，请参考官方的安装教程 https://pytorch.org/get-started/locally/**

### 基本用法

```bash
# 基本放大
python main.py magnify input.mp4 output.mp4

# 使用预设模式（心跳/脉搏）
python main.py magnify input.mp4 output.mp4 --preset pulse

# 分析视频并建议参数
python main.py suggest input.mp4
```

## 核心功能

- **运动放大**：基于拉普拉斯金字塔的欧拉运动放大，可放大肉眼难以察觉的微小运动
- **智能频率分析**：自动检测视频频率特性，生成最优参数建议
- **预设模式**：针对常见场景（心跳、呼吸等）的优化参数
- **视频稳定与降噪**：减少相机抖动和噪点干扰
- **高效处理**：支持GPU加速和智能缓存，高效处理大型视频

## 常用命令

### 频率分析

```bash
# 分析视频并建议最佳参数
python main.py suggest input.mp4

# 列出所有可用预设
python main.py list-presets

# 可视化特定预设的频率响应
python main.py visualize-preset pulse
```

### 使用预设模式

```bash
# 放大脉搏/心跳
python main.py magnify input.mp4 output.mp4 --preset pulse

# 放大呼吸运动
python main.py magnify input.mp4 output.mp4 --preset breathing
```

### 自定义参数

```bash
# 自定义频率范围和放大系数
python main.py magnify input.mp4 output.mp4 --amplify 15 --freq-min 0.8 --freq-max 2.0

# 启用视频稳定和降噪
python main.py magnify input.mp4 output.mp4 --stabilize --blur 0.5 --motion-threshold 0.02
```

## 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--preset` | 使用预定义参数集 (pulse, breathing, etc.) | (无) |
| `--amplify` | 运动放大系数 | 10.0 |
| `--freq-min` | 最小放大频率 (Hz) | 0.4 |
| `--freq-max` | 最大放大频率 (Hz) | 3.0 |
| `--stabilize` | 放大前执行视频稳定 | False |
| `--blur` | 空间模糊强度（0表示禁用） | 0 |
| `--motion-threshold` | 最小运动幅度阈值（0表示禁用） | 0 |

<details>
<summary>查看更多参数</summary>

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
| `--analysis-dir` | 保存分析结果的目录 | 输出目录 |
| `--sampling-rate` | 分析时采样的帧比例 (0.0-1.0) | 0.5 |
| `--roi` | 感兴趣区域 (格式: "100,100,200,200") | 整个画面 |
| `--skip-visualizations` | 跳过生成可视化图像以加快分析速度 | False |
| `--no-cache` | 禁用缓存功能 | False |
| `--use-analysis` | 使用指定的分析结果文件 | (无) |

</details>

## 使用技巧

- **找到合适的频率范围**：使用 `suggest` 命令自动检测最佳频率
- **减少噪点**：增加 `--blur` 值（0.5-1.0）并设置 `--motion-threshold`（0.01-0.05）
- **处理抖动视频**：添加 `--stabilize` 参数
- **提高处理效率**：增加 `--chunk-size` 提高处理速度（如果内存允许）
- **加速重复处理**：系统会自动缓存分析结果和稳定化视频

<details>
<summary>更多高级技巧</summary>

- **针对不同类型的运动使用不同的频率范围**：
  - 脉搏/心跳：0.8-2.0 Hz
  - 呼吸：0.1-0.5 Hz
  - 快速振动：可高达10-30 Hz

- **高级降噪选项**：
  - 使用 `--adaptive` 更好地保留细节
  - 尝试 `--bilateral` 在减少噪点的同时保留边缘

- **处理大视频**：
  - 减小 `--overlap` 减少内存使用
  - 分析时使用 `--sampling-rate` 加速

- **处理抖动视频**：
  - 调整 `--stabilize-radius` 和 `--stabilize-strength` 获得最佳效果
  - 使用 `--keep-temp` 保留稳定化视频供后续使用

- **缓存管理**：
  - 所有缓存文件存储在 `TEMP` 目录，可通过 `.gitignore` 忽略
  - 系统会自动清理30天以上的旧缓存

</details>

## 项目结构

项目采用模块化架构，将相关功能组织到专门的模块中，提高了代码的可维护性和可扩展性。用户无需关心内部结构，只需通过`main.py`与系统交互即可。

- **命令处理模块**：
  - `main.py` - 主程序入口和参数解析
  - `analysis_commands.py` - 分析相关命令实现

- **核心算法模块**：
  - `motion_amplifier.py` - 线性欧拉放大法实现
  - `pyramid.py` - 拉普拉斯金字塔实现
  - `temporal_filter.py` - 时域滤波实现

- **视频处理模块**：
  - `video_handler.py` - 视频I/O处理
  - `stabilizer.py` - 视频稳定功能

- **频率分析模块**：
  - `frequency_analyzer.py` 及相关支持文件

- **缓存管理模块**：
  - `cache_manager.py` 及相关支持文件

- `TEMP/` - 存放缓存和临时文件

## 许可证

[待添加]
