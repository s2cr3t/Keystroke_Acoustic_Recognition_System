# 键盘声纹识别系统（Keystroke Acoustic Recognition System）

本项目旨在构建一套高鲁棒性、可扩展的键盘按键声音识别系统，基于深度神经网络、传统机器学习模型及序列建模算法，自动解析用户键盘操作产生的音频数据，进而恢复其按键输入序列。该系统适用于密码重构、声纹侧信道攻击研究、用户行为建模等信息安全相关场景。

---

## 系统特性

- 🌀 基于自适应能量包络与零交叉率的高精度音频分帧与事件检测算法；
- 🔍 集成多尺度特征提取模块，包括：MFCC、频谱特征、短时时域特征、小波域变换与Chroma和声结构；
- ⚙️ 多种传统分类模型支持（如 Random Forest、Gradient Boosting）；
- 🧠 基于卷积神经网络（CNN）的深度学习结构实现端到端声纹识别；
- ✏️ 结合 N-gram 与概率图方法进行序列预测与上下文校正；
- 🚀 提供命令行接口，支持全流程自动化训练、推理、样本录制；
- 📊 模型训练动态、分帧可视化、识别结果可视化全流程支持。

---

## 环境配置与依赖

```bash
pip install -r requirements.txt
```

依赖主要包括：`numpy`, `librosa`, `tensorflow`, `scikit-learn`, `matplotlib`, `pyaudio`, `soundfile`, `pywt` 等。

---

## 使用说明

### 📂 模型训练

将录制好的训练数据以 `key_<label>_xxx.wav` 命名格式存入指定目录，并可选提供对应文本标签序列 `seq_*.txt`。

```bash
python main.py train --samples data/samples
```

### 🎧 单文件预测

```bash
python main.py predict --file data/test.wav
```

### ☕ 静默模式运行：

```bash
python main.py predict --file data/test.wav --quiet
```

### ⏺ 指定按键录制样本

```bash
python main.py record --key a --output data/samples --count 5
```

### 📊 批量目录预测

```bash
python main.py batch --dir data/test_samples --output batch_results.txt
```

---

## 项目目录结构

```
.
├── models/                # 🔗 已训练模型存储路径
├── data/                  # 📂 原始音频与标注数据集
├── results/               # 📊 可视化图像与评估输出
├── main.py                # ▶️ 命令行入口文件
├── config.json            # 🔧 系统参数配置
└── requirements.txt       # 📦 环境依赖文件
```

---

## 核心算法结构

- ♫ 音频分段：基于 RMS 能量谱与零交叉率联合门控的击键检测器；
- 🔢 特征工程：融合 MFCC、频谱质心、带宽、Roll-off、小波能量与和声模板；
- 🧩 多模型融合：传统与深度分类器结果集成，动态加权投票；
- ✍️ 序列建模：应用 N-gram 概率语言模型进行序列重构与上下文纠错。

---

## 应用前景

- 🤖 安全研究：侧信道攻击演示、键盘行为重建、行为识别建模；
- 🔍 教学科研：音频事件检测、语音信号处理课程项目支撑；
- 🕵️‍♂️ 威胁分析：APT 工具链行为审计、设备监听取证支持。

---

## 程序接口与自动化

- 🔗 提供 `KeystrokeRecognitionSystem` 类供开发者直接集成；
- 🔢 所有功能支持命令行参数调用，便于批处理与脚本化集成。

---

## 获取帮助

📄 建议阅读源码中的注释、流程控制模块、特征工程与模型封装代码。

📢 欢迎提交 issue 或 pull request 以优化系统模块及算法策略，感谢您的贡献与同行交流。
