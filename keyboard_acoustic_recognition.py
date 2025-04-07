#-*-coding:gb2312-*-
import os
import numpy as np
import librosa
import soundfile as sf
import pickle
import json
import time
import pyaudio
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from collections import Counter, defaultdict
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import multiprocessing
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# 常量和配置
CONFIG = {
    'sample_rate': 44100,
    'frame_length': 1024,
    'hop_length': 256,
    'mfcc_coef': 40,          # 增加MFCC系数数量
    'min_segment_length': 0.05,
    'max_segment_length': 0.3,
    'energy_threshold_percentile': 85,
    'silence_threshold': 0.03,
    'min_silence_duration': 0.08,
    'bandpass_filter': {
        'lowcut': 100,        # 过滤低频噪声
        'highcut': 8000,      # 保留键盘声音的主要频段
        'order': 4
    },
    'feature_extraction': {
        'use_mfcc': True,
        'use_spectral': True, 
        'use_temporal': True,
        'use_wavelet': True,
        'use_chroma': True
    },
    'models': {
        'traditional': {
            'use_rf': True,
            'use_gb': True,
            'use_svm': False,
            'ensemble_weight': 0.3
        },
        'deep_learning': {
            'use_cnn': True,
            'use_lstm': True,
            'ensemble_weight': 0.7,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10
        }
    },
    'sequence_model': {
        'use_ngram': True,
        'use_hmm': True, 
        'ngram_weight': 0.3,
        'ngram_order': 3
    },
    'paths': {
        'model_dir': 'models',
        'data_dir': 'data',
        'results_dir': 'results',
        'feature_cache': 'features.pkl',
        'config_file': 'config.json'
    }
}

# 确保目录存在
for dir_path in CONFIG['paths'].values():
    if isinstance(dir_path, str) and dir_path.endswith('/'):
        os.makedirs(dir_path, exist_ok=True)

# 保存配置到JSON
with open(CONFIG['paths'].get('config_file', 'config.json'), 'w') as f:
    json.dump(CONFIG, f, indent=4)

# 音频处理工具类
class AudioProcessor:
    """高级音频处理类，提供预处理、滤波和分段功能"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.sr = self.config['sample_rate']
    
    def load_audio(self, file_path, normalize=True):
        """加载并预处理音频文件"""
        y, sr = librosa.load(file_path, sr=self.sr)
        
        if normalize:
            y = librosa.util.normalize(y)
        
        # 应用带通滤波器，保留键盘声音的主要频段
        if self.config['bandpass_filter']:
            y = self.bandpass_filter(
                y, 
                self.config['bandpass_filter']['lowcut'],
                self.config['bandpass_filter']['highcut'],
                sr,
                self.config['bandpass_filter']['order']
            )
        
        return y, sr
        
    def bandpass_filter(self, data, lowcut, highcut, sr, order=5):
        """应用带通滤波器，保留指定频段的信号"""
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def detect_keystrokes(self, y, sr, visualize=False):
        """先进的按键检测和分段算法"""
        # 计算能量包络
        frame_length = self.config['frame_length']
        hop_length = self.config['hop_length']
        
        # 使用RMS能量和ZCR共同检测击键
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 计算能量自适应阈值（基于百分位数）
        energy_threshold = np.percentile(energy, self.config['energy_threshold_percentile'])
        
        # 检测能量峰值作为潜在按键
        min_distance = int(self.config['min_silence_duration'] * sr / hop_length)
        peaks, properties = find_peaks(
            energy,
            height=energy_threshold,
            distance=min_distance,
            prominence=energy_threshold * 0.5
        )
        
        # 使用ZCR帮助过滤噪声峰值
        filtered_peaks = []
        for peak in peaks:
            # 按键通常具有较高的ZCR
            if peak < len(zcr) and zcr[peak] > np.mean(zcr) * 0.8:
                filtered_peaks.append(peak)
        
        # 如果过滤后没有足够的峰值，回退到原始峰值
        if len(filtered_peaks) < 0.5 * len(peaks):
            filtered_peaks = peaks
        
        # 转换到时间域
        peak_times = librosa.frames_to_time(filtered_peaks, sr=sr, hop_length=hop_length)
        
        # 提取段落
        segments = []
        segment_times = []
        
        for i, peak_time in enumerate(peak_times):
            # 确定段落长度 - 更长的距离下一个峰值则使用更大的窗口
            if i < len(peak_times) - 1:
                next_peak = peak_times[i + 1]
                segment_duration = min(
                    self.config['max_segment_length'],
                    (next_peak - peak_time) * 0.7
                )
            else:
                segment_duration = self.config['max_segment_length']
            
            # 确保最小段落长度
            segment_duration = max(self.config['min_segment_length'], segment_duration)
            
            # 提取段落
            start_time = max(0, peak_time - 0.05)  # 稍微提前抓取
            end_time = min(len(y) / sr, start_time + segment_duration)
            
            start_idx = int(start_time * sr)
            end_idx = int(end_time * sr)
            
            if end_idx > start_idx:  # 确保段落有效
                segments.append(y[start_idx:end_idx])
                segment_times.append((start_time, end_time))
        
        # 可视化调试
        if visualize:
            self.visualize_segmentation(y, sr, energy, zcr, filtered_peaks, hop_length, segment_times)
            
        return segments, segment_times, energy
    
    def visualize_segmentation(self, y, sr, energy, zcr, peaks, hop_length, segment_times):
        """可视化分段结果以便调试"""
        plt.figure(figsize=(15, 10))
        
        # 显示波形
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title('音频波形与检测到的按键段落')
        
        # 标记检测到的段落
        for start, end in segment_times:
            plt.axvspan(start, end, alpha=0.3, color='red')
            plt.axvline(x=start, color='green', linestyle='--')
        
        # 显示能量包络
        plt.subplot(3, 1, 2)
        times = librosa.times_like(energy, sr=sr, hop_length=hop_length)
        plt.plot(times, energy)
        plt.axhline(y=np.percentile(energy, self.config['energy_threshold_percentile']), 
                   color='r', linestyle='--', label='阈值')
        
        # 标记峰值
        peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        plt.scatter(peak_times, energy[peaks], color='red', label='检测到的峰值')
        plt.title('能量包络')
        plt.legend()
        
        # 显示ZCR
        plt.subplot(3, 1, 3)
        plt.plot(times, zcr)
        plt.title('过零率 (ZCR)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['paths']['results_dir'], 'segmentation.png'))
        plt.close()

# 特征提取类
class FeatureExtractor:
    """先进的特征提取器，支持时域、频域和时频联合分析"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG
    
    def extract_features(self, y, sr):
        """从音频片段中提取丰富的特征集"""
        features = []
        
        # 1. MFCC特征
        if self.config['feature_extraction']['use_mfcc']:
            mfcc_features = self.extract_mfcc_features(y, sr)
            features.extend(mfcc_features)
        
        # 2. 频谱特征
        if self.config['feature_extraction']['use_spectral']:
            spectral_features = self.extract_spectral_features(y, sr)
            features.extend(spectral_features)
        
        # 3. 时域特征
        if self.config['feature_extraction']['use_temporal']:
            temporal_features = self.extract_temporal_features(y, sr)
            features.extend(temporal_features)
        
        # 4. 小波变换特征
        if self.config['feature_extraction']['use_wavelet']:
            wavelet_features = self.extract_wavelet_features(y)
            features.extend(wavelet_features)
        
        # 5. 和声特征
        if self.config['feature_extraction']['use_chroma']:
            chroma_features = self.extract_chroma_features(y, sr)
            features.extend(chroma_features)
        
        return np.array(features)
    
    def extract_mfcc_features(self, y, sr):
        """提取增强的MFCC特征"""
        # 提取MFCCs
        n_mfcc = self.config['mfcc_coef']
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
        # 计算统计量
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
    
        # 添加MFCC的delta和delta-delta特征
        # 检查是否有足够的帧来计算delta
        if mfccs.shape[1] >= 9:  # 默认width=9
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
            delta_mean = np.mean(mfcc_delta, axis=1)
            delta_std = np.std(mfcc_delta, axis=1)
            delta2_mean = np.mean(mfcc_delta2, axis=1)
            delta2_std = np.std(mfcc_delta2, axis=1)
        else:
            # 如果帧数不足，则使用较小的width参数或填充零
            width = min(3, mfccs.shape[1] - 1)
            if width > 1:
                mfcc_delta = librosa.feature.delta(mfccs, width=width)
                mfcc_delta2 = librosa.feature.delta(mfccs, order=2, width=width)
            
                delta_mean = np.mean(mfcc_delta, axis=1)
                delta_std = np.std(mfcc_delta, axis=1)
                delta2_mean = np.mean(mfcc_delta2, axis=1)
                delta2_std = np.std(mfcc_delta2, axis=1)
            else:
                # 如果帧数太少，无法计算差分，则用零填充
                delta_mean = np.zeros_like(mfcc_mean)
                delta_std = np.zeros_like(mfcc_std)
                delta2_mean = np.zeros_like(mfcc_mean)
                delta2_std = np.zeros_like(mfcc_std)
    
        # 计算偏度和峰度（如果可能）
        try:
            mfcc_skew = stats.skew(mfccs, axis=1)
            mfcc_kurtosis = stats.kurtosis(mfccs, axis=1)
        except:
            # 如果计算失败，用零填充
            mfcc_skew = np.zeros_like(mfcc_mean)
            mfcc_kurtosis = np.zeros_like(mfcc_mean)
    
        # 合并所有特征
        features = np.concatenate([
            mfcc_mean, mfcc_std, mfcc_skew, mfcc_kurtosis,
            delta_mean, delta_std, delta2_mean, delta2_std
        ])
    
        return features
    
    def extract_spectral_features(self, y, sr):
        """提取频谱特征"""
        # 基本频谱特征
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        
        # 计算统计量
        features = [
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(np.mean(spectral_contrast, axis=0)),
            np.std(np.std(spectral_contrast, axis=0)),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(spectral_flatness),
            np.std(spectral_flatness)
        ]
        
        # 添加频谱峰值特征
        S = np.abs(librosa.stft(y))
        peak_freq_idx = np.argmax(S, axis=0)
        peak_freqs = librosa.fft_frequencies(sr=sr)[peak_freq_idx]
        
        features.extend([
            np.mean(peak_freqs),
            np.std(peak_freqs),
            np.median(peak_freqs)
        ])
        
        return features
    
    def extract_temporal_features(self, y, sr):
        """提取时域特征"""
        # 基本统计特征
        features = [
            np.mean(y),
            np.std(y),
            np.max(y),
            np.min(y),
            np.median(y),
            stats.skew(y),
            stats.kurtosis(y)
        ]
        
        # 过零率
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([
            np.mean(zcr),
            np.std(zcr)
        ])
        
        # RMS能量
        rms = librosa.feature.rms(y=y)[0]
        features.extend([
            np.mean(rms),
            np.std(rms),
            np.max(rms)
        ])
        
        # 包络特征
        envelope = np.abs(librosa.onset.onset_strength(y=y, sr=sr))
        features.extend([
            np.mean(envelope),
            np.std(envelope),
            np.max(envelope)
        ])
        
        # 攻击时间估计
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        try:
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            if len(onset_frames) > 0:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                # 使用第一个onset作为攻击时间
                attack_time = onset_times[0]
            else:
                attack_time = 0
        except:
            attack_time = 0
        
        features.append(attack_time)
        
        return features
    
    def extract_wavelet_features(self, y):
        """提取小波变换特征"""
        try:
            import pywt
            
            # 应用小波变换
            wavelet = 'db4'
            coeffs = pywt.wavedec(y, wavelet, level=5)
            
            # 提取每个系数集的统计特征
            features = []
            for i, coef in enumerate(coeffs):
                features.extend([
                    np.mean(np.abs(coef)),
                    np.std(coef),
                    np.max(np.abs(coef)),
                    np.sum(coef**2) / len(coef)  # 能量
                ])
            
            return features
        except ImportError:
            # 如果pywt不可用，返回空列表
            return []
    
    def extract_chroma_features(self, y, sr):
        """提取和声特征"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        chroma_max = np.max(chroma, axis=1)
        
        return np.concatenate([chroma_mean, chroma_std, chroma_max])
    
    def extract_all_features(self, audio_segments, sr):
        """从所有音频段中提取特征"""
        features_list = []
        
        for segment in audio_segments:
            # 确保段落有足够的样本
            if len(segment) >= self.config['frame_length']:
                features = self.extract_features(segment, sr)
                features_list.append(features)
        
        return features_list

# 模型训练和评估
class KeystrokeModelTrainer:
    """高级模型训练类，支持深度学习和传统机器学习模型"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.models = {}
        self.scaler = StandardScaler()
        self.initialize_models()
    
    def initialize_models(self):
        """初始化所有支持的模型"""
        # 传统机器学习模型
        if self.config['models']['traditional']['use_rf']:
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )
        
        if self.config['models']['traditional']['use_gb']:
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
        # 深度学习模型在训练时创建
    
    def train(self, X, y, deep_learning=True):
        """训练所有选择的模型"""
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        # 确保模型目录存在 - 在这里添加这行代码
        os.makedirs(self.config['paths']['model_dir'], exist_ok=True)
    
        # 保存标准化器
        with open(os.path.join(self.config['paths']['model_dir'], 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 训练传统机器学习模型
        for name, model in self.models.items():
            print(f"训练 {name} 模型...")
            model.fit(X_train_scaled, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred)
            print(f"{name} 性能:\n{report}")
            
            # 保存模型
            with open(os.path.join(self.config['paths']['model_dir'], f'{name}.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        # 训练深度学习模型
        if deep_learning and self.config['models']['deep_learning']['use_cnn']:
            # 准备深度学习数据
            classes = np.unique(y)
            class_to_idx = {cls: i for i, cls in enumerate(classes)}
            y_train_idx = np.array([class_to_idx[cls] for cls in y_train])
            y_test_idx = np.array([class_to_idx[cls] for cls in y_test])
            
            # 转换为分类格式
            y_train_cat = to_categorical(y_train_idx)
            y_test_cat = to_categorical(y_test_idx)
            
            # 为CNN准备输入数据
            X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
            X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
            
            # 创建并训练CNN模型
            cnn_model = self.create_cnn_model(input_shape=X_train_cnn.shape[1:], num_classes=len(classes))
            
            # 定义回调函数
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['models']['deep_learning']['patience'],
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    os.path.join(self.config['paths']['model_dir'], 'cnn_model.h5'),
                    save_best_only=True
                )
            ]
            
            # 训练模型
            history = cnn_model.fit(
                X_train_cnn, y_train_cat,
                validation_data=(X_test_cnn, y_test_cat),
                epochs=self.config['models']['deep_learning']['epochs'],
                batch_size=self.config['models']['deep_learning']['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # 评估模型
            y_pred_proba = cnn_model.predict(X_test_cnn)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = y_test_idx
            
            report = classification_report(y_true, y_pred)
            print(f"CNN 性能:\n{report}")
            
            # 保存类别索引映射
            with open(os.path.join(self.config['paths']['model_dir'], 'class_indices.json'), 'w') as f:
                json.dump(class_to_idx, f)

            os.makedirs(self.config['paths']['results_dir'], exist_ok=True)

            # 保存训练历史
            history_dict = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy']
            }
            
            with open(os.path.join(self.config['paths']['results_dir'], 'training_history.json'), 'w') as f:
                json.dump(history_dict, f)
            
            # 可视化训练历史
            self.plot_training_history(history)
            
            # 保存CNN模型
            cnn_model.save(os.path.join(self.config['paths']['model_dir'], 'cnn_model.h5'))
            self.models['cnn'] = cnn_model
        
        return self.models
    
    def create_cnn_model(self, input_shape, num_classes):
        """创建一个1D CNN模型用于音频特征分类"""
        model = Sequential([
            # 第一个卷积层
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            # 第二个卷积层
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            # 第三个卷积层
            Conv1D(filters=256, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            # 全连接层
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        return model
    
    def plot_training_history(self, history):
        """可视化模型训练历史"""
        os.makedirs(self.config['paths']['results_dir'], exist_ok=True)
        plt.figure(figsize=(12, 4))
        
        # 绘制损失
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制准确率
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['paths']['results_dir'], 'training_history.png'))
        plt.close()

# 语言模型和序列建模类
class SequenceModeling:
    """序列建模组件，使用N-gram模型和HMM增强预测结果"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.ngram_model = None
        self.hmm_model = None
    
    def train_ngram_model(self, sequences):
        """训练N-gram语言模型"""
        from collections import defaultdict
        
        order = self.config['sequence_model']['ngram_order']
        model = defaultdict(Counter)
        
        # 构建N-gram频率
        for sequence in sequences:
            # 添加开始和结束标记
            padded_seq = ['<s>'] * (order - 1) + list(sequence) + ['</s>']
            
            for i in range(len(padded_seq) - order + 1):
                context = tuple(padded_seq[i:i+order-1])
                next_char = padded_seq[i+order-1]
                model[context][next_char] += 1
        
        # 将频率转换为概率
        ngram_model = {}
        for context, counter in model.items():
            total = sum(counter.values())
            ngram_model[context] = {char: count/total for char, count in counter.items()}
        
        self.ngram_model = ngram_model
        
        # 保存模型
        with open(os.path.join(self.config['paths']['model_dir'], 'ngram_model.pkl'), 'wb') as f:
            pickle.dump(ngram_model, f)
        
        return ngram_model
    
    def score_sequence(self, sequence, smoothing=0.1):
        """使用N-gram模型计算序列概率"""
        if not self.ngram_model:
            return 0
        
        order = self.config['sequence_model']['ngram_order']
        log_prob = 0
        
        # 添加开始和结束标记
        padded_seq = ['<s>'] * (order - 1) + list(sequence) + ['</s>']
        
        for i in range(len(padded_seq) - order + 1):
            context = tuple(padded_seq[i:i+order-1])
            next_char = padded_seq[i+order-1]
            
            # 获取条件概率，使用平滑处理
            if context in self.ngram_model and next_char in self.ngram_model[context]:
                prob = self.ngram_model[context][next_char]
            else:
                # 拉普拉斯平滑
                prob = smoothing / (smoothing * len(self.get_vocab()))
            
            log_prob += np.log(prob)
        
        return log_prob
    
    def get_vocab(self):
        """获取N-gram模型的词汇表"""
        if not self.ngram_model:
            return set()
        
        vocab = set()
        for context_dict in self.ngram_model.values():
            vocab.update(context_dict.keys())
        
        return vocab
    
    def correct_sequence(self, predicted_sequence, top_k=3):
        """使用N-gram模型修正预测序列"""
        if not self.ngram_model or not predicted_sequence:
            return predicted_sequence
        
        # 生成可能的序列变种（允许一定数量的替换）
        variants = self.generate_variants(predicted_sequence, top_k)
        
        # 计算每个变种的得分
        scored_variants = [(variant, self.score_sequence(variant)) for variant in variants]
        
        # 选择得分最高的
        best_variant = max(scored_variants, key=lambda x: x[1])
        
        return best_variant[0]
    
    def generate_variants(self, sequence, top_k=3):
        """生成序列的可能变种"""
        if not self.ngram_model:
            return [sequence]
        
        # 获取词汇表
        vocab = self.get_vocab()
        vocab = [v for v in vocab if v not in ['<s>', '</s>']]
        
        variants = [sequence]
        
        # 为每个位置生成替换字符
        for i in range(len(sequence)):
            prefix = sequence[:i]
            suffix = sequence[i+1:]
            
            for char in vocab:
                if char != sequence[i]:
                    variant = prefix + char + suffix
                    variants.append(variant)
        
        # 计算每个变种的得分
        scored_variants = [(variant, self.score_sequence(variant)) for variant in variants]
        scored_variants.sort(key=lambda x: x[1], reverse=True)
        
        # 返回得分最高的top_k个变种
        return [v for v, _ in scored_variants[:top_k]]

# 主接口类
class KeystrokeRecognitionSystem:
    """键盘声音识别系统的主接口类"""
    
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.audio_processor = AudioProcessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.model_trainer = KeystrokeModelTrainer(self.config)
        self.sequence_model = SequenceModeling(self.config)
        self.models = {}
        self.class_indices = None
        self.scaler = None
    
    def load_models(self):
        """加载所有已训练的模型"""
        model_dir = self.config['paths']['model_dir']
        models = {}
        
        # 加载传统机器学习模型
        for model_name in ['random_forest', 'gradient_boosting']:
            model_path = os.path.join(model_dir, f'{model_name}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
        
        # 加载CNN模型
        cnn_path = os.path.join(model_dir, 'cnn_model.h5')
        if os.path.exists(cnn_path):
            try:
                models['cnn'] = load_model(cnn_path)
            except:
                print("警告: 无法加载CNN模型")
        
        # 加载类别索引映射
        class_indices_path = os.path.join(model_dir, 'class_indices.json')
        if os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                self.class_indices = json.load(f)
                # 创建逆映射
                self.idx_to_class = {int(v): k for k, v in self.class_indices.items()}
        
        # 加载缩放器
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # 加载N-gram模型
        ngram_path = os.path.join(model_dir, 'ngram_model.pkl')
        if os.path.exists(ngram_path):
            with open(ngram_path, 'rb') as f:
                self.sequence_model.ngram_model = pickle.load(f)
        
        self.models = models
        return models
    
    def train_from_samples(self, sample_dir):
        """从样本目录训练模型"""
        # 加载并预处理所有样本
        X = []
        y = []
        sample_sequences = []
        
        # 收集样本文件
        samples_by_key = defaultdict(list)
        for filename in os.listdir(sample_dir):
            if filename.endswith('.wav'):
                parts = filename.split('_')
                if len(parts) >= 2:
                    key = parts[0]
                    samples_by_key[key].append(os.path.join(sample_dir, filename))
        
        print(f"找到 {len(samples_by_key)} 种不同按键的样本")
        
        # 处理每个样本
        for key, files in samples_by_key.items():
            print(f"处理 '{key}' 的 {len(files)} 个样本...")
            
            for file_path in files:
                try:
                    # 加载音频
                    y_audio, sr = self.audio_processor.load_audio(file_path)
                    
                    # 分段
                    segments, _, _ = self.audio_processor.detect_keystrokes(y_audio, sr)
                    
                    # 如果找到了段落，使用它们
                    if segments:
                        for segment in segments:
                            features = self.feature_extractor.extract_features(segment, sr)
                            X.append(features)
                            y.append(key)
                    else:
                        # 如果没有找到段落，使用整个音频
                        features = self.feature_extractor.extract_features(y_audio, sr)
                        X.append(features)
                        y.append(key)
                        
                except Exception as e:
                    print(f"处理 {file_path} 时出错: {str(e)}")
        
        if not X:
            print("错误: 未能从样本中提取有效特征")
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"提取的特征形状: {X.shape}")
        print(f"标签数量: {len(y)}")
        print(f"唯一标签: {np.unique(y)}")
        
        # 训练模型
        models = self.model_trainer.train(X, y)
        self.models = models
        
        # 获取预测的按键序列用于训练N-gram模型
        # 我们假设特定文件前缀（如'seq_'）表示完整的输入序列
        sequence_files = [f for f in os.listdir(sample_dir) if f.startswith('seq_') and f.endswith('.txt')]
        
        if sequence_files:
            print("训练N-gram序列模型...")
            sequences = []
            
            for seq_file in sequence_files:
                with open(os.path.join(sample_dir, seq_file), 'r') as f:
                    content = f.read().strip()
                    if content:
                        sequences.append(content)
            
            if sequences:
                self.sequence_model.train_ngram_model(sequences)
        
        return models
    
    def predict_from_file(self, audio_file, verbose=True):
        """从音频文件预测按键序列"""
        if not self.models:
            self.load_models()
            if not self.models:
                print("错误: 未找到训练好的模型")
                return ""
        
        try:
            # 加载并预处理音频
            y, sr = self.audio_processor.load_audio(audio_file)
            
            # 分割按键
            segments, segment_times, energy = self.audio_processor.detect_keystrokes(y, sr, visualize=verbose)
            
            if verbose:
                print(f"检测到 {len(segments)} 个潜在按键")
            
            if not segments:
                return ""
            
            # 提取特征
            features_list = []
            for segment in segments:
                features = self.feature_extractor.extract_features(segment, sr)
                features_list.append(features)
            
            X = np.array(features_list)
            
            # 标准化特征
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # 使用所有模型进行预测
            predictions = {}
            confidences = {}
            
            for name, model in self.models.items():
                if name == 'cnn':
                    # 为CNN准备输入
                    X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                    
                    # 预测概率
                    y_proba = model.predict(X_cnn)
                    
                    # 获取最可能的类别
                    y_pred_indices = np.argmax(y_proba, axis=1)
                    
                    # 将索引转换回原始标签
                    y_pred = [self.idx_to_class[idx] for idx in y_pred_indices]
                    
                    # 获取置信度
                    conf = np.max(y_proba, axis=1)
                else:
                    # 传统模型预测
                    y_pred = model.predict(X_scaled)
                    
                    # 如果模型支持predict_proba，获取置信度
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_scaled)
                        conf = np.max(y_proba, axis=1)
                    else:
                        conf = np.ones(len(y_pred))
                
                predictions[name] = y_pred
                confidences[name] = conf
            
            # 整合各模型的结果
            final_predictions = []
            for i in range(len(segments)):
                # 收集各模型的预测
                votes = {}
                for name, preds in predictions.items():
                    if i < len(preds):
                        pred = preds[i]
                        conf = confidences[name][i]
                        
                        # 按权重累计投票
                        weight = self.config['models']['deep_learning']['ensemble_weight'] if name == 'cnn' else \
                                 self.config['models']['traditional']['ensemble_weight'] / (len(self.models) - 1) \
                                 if len(self.models) > 1 else 1.0
                        
                        if pred not in votes:
                            votes[pred] = 0
                        votes[pred] += conf * weight
                
                # 选择得票最高的预测
                if votes:
                    top_pred = max(votes.items(), key=lambda x: x[1])[0]
                    final_predictions.append(top_pred)
                    
                    if verbose:
                        print(f"片段 {i+1}: 预测为 '{top_pred}' (票数: {votes[top_pred]:.2f})")
            
            # 获取最终结果
            result = ''.join(final_predictions)
            
            # 使用N-gram模型纠正序列（如果已训练）
            if self.sequence_model.ngram_model and len(result) > 1:
                corrected_result = self.sequence_model.correct_sequence(result)
                
                if verbose and corrected_result != result:
                    print(f"\n原始预测: {result}")
                    print(f"N-gram校正后: {corrected_result}")
                
                result = corrected_result
            
            if verbose:
                print(f"\n最终预测: {result}")
            
            return result
            
        except Exception as e:
            print(f"预测过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    
    def visualize_predictions(self, audio_file, predictions):
        """可视化音频文件和预测结果"""
        y, sr = self.audio_processor.load_audio(audio_file)
        segments, segment_times, energy = self.audio_processor.detect_keystrokes(y, sr, visualize=False)
        
        # 创建可视化
        plt.figure(figsize=(15, 6))
        
        # 显示波形
        librosa.display.waveshow(y, sr=sr)
        
        # 标记检测到的按键并添加预测标签
        for i, (start, end) in enumerate(segment_times):
            if i < len(predictions):
                plt.axvspan(start, end, alpha=0.2, color='red')
                plt.text(start, 0, predictions[i], fontsize=12, color='red')
        
        plt.title('音频波形与按键预测')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        
        # 保存和显示
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['paths']['results_dir'], 'prediction_visualization.png'))
        plt.close()

# 命令行界面
def main():
    """主函数，提供命令行界面"""
    import argparse
    
    parser = argparse.ArgumentParser(description="高级键盘声音识别系统")
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='从样本训练模型')
    train_parser.add_argument('--samples', type=str, required=True, help='样本目录路径')
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='预测音频文件中的按键序列')
    predict_parser.add_argument('--file', type=str, required=True, help='音频文件路径')
    predict_parser.add_argument('--quiet', action='store_true', help='静默模式，只输出预测结果')
    
    # 录制命令
    record_parser = subparsers.add_parser('record', help='录制样本')
    record_parser.add_argument('--key', type=str, help='要录制的按键')
    record_parser.add_argument('--output', type=str, help='输出目录')
    record_parser.add_argument('--count', type=int, default=10, help='录制次数')
    
    # 批量预测命令
    batch_parser = subparsers.add_parser('batch', help='批量预测目录中的所有音频文件')
    batch_parser.add_argument('--dir', type=str, required=True, help='音频文件目录')
    batch_parser.add_argument('--output', type=str, help='结果输出文件')
    
    args = parser.parse_args()
    
    # 初始化系统
    system = KeystrokeRecognitionSystem()
    
    if args.command == 'train':
        print(f"从 {args.samples} 训练模型...")
        system.train_from_samples(args.samples)
        print("训练完成！")
    
    elif args.command == 'predict':
        if not os.path.exists(args.file):
            print(f"错误: 文件 {args.file} 不存在")
            return
        
        print(f"分析 {args.file}...")
        result = system.predict_from_file(args.file, verbose=not args.quiet)
        
        if args.quiet:
            print(result)
        else:
            print(f"\n预测结果: {result}")
    
    elif args.command == 'record':
        if not args.key:
            print("错误: 请指定要录制的按键 (--key)")
            return
        
        output_dir = args.output or "samples"
        os.makedirs(output_dir, exist_ok=True)
        
        # 录制样本的代码
        record_samples(args.key, output_dir, args.count)
    
    elif args.command == 'batch':
        if not os.path.exists(args.dir):
            print(f"错误: 目录 {args.dir} 不存在")
            return
        
        output_file = args.output or "batch_results.txt"
        batch_predict(system, args.dir, output_file)
    
    else:
        parser.print_help()

def record_samples(key, output_dir="samples", count=10, seconds=1.5):
    """
    为指定按键录制多个音频样本，避免录制到开始录制时的回车键声音
    
    参数:
    key (str): 要录制的按键名称
    output_dir (str): 保存样本的目录
    count (int): 要录制的样本数量
    seconds (float): 每个样本的录制时长（秒）
    """
    import wave  # 添加这一行导入
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 录音参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    p = pyaudio.PyAudio()
    
    print(f"\n准备录制 {count} 个 '{key}' 按键的样本...")
    
    for i in range(count):
        print(f"\n[{i+1}/{count}] 准备录制 '{key}'...")
        print(f"按下回车键，然后等待倒计时结束后按下 '{key}' 键...")
        
        # 等待用户按下回车
        input()
        
        # 显示倒计时，避免录制到回车键声音
        for countdown in range(3, 0, -1):
            print(f"\r准备录制: {countdown}秒...", end="")
            time.sleep(1)
        
        print(f"\r现在开始录制! 请按下 '{key}' 键            ")
        
        # 录制音频
        stream = p.open(format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK)
        
        frames = []
        for j in range(0, int(RATE / CHUNK * seconds)):
            data = stream.read(CHUNK)
            frames.append(data)
            
            # 显示进度
            if j % 5 == 0:
                remaining = seconds - (j * CHUNK / RATE)
                print(f"\r录制中: 剩余 {remaining:.1f}秒...", end="")
        
        print("\r录制完成!                         ")
        
        # 停止录制
        stream.stop_stream()
        stream.close()
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = os.path.join(output_dir, f"{key}_{timestamp}.wav")
        
        # 保存音频
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"样本已保存至: {filename}")
        
        if i < count - 1:
            print("准备下一次录制...")
            time.sleep(0.5)
    
    p.terminate()
    print(f"\n成功录制 {count} 个 '{key}' 的样本！")

def batch_predict(system, directory, output_file):
    """批量预测目录中的所有音频文件"""
    results = {}
    
    # 获取所有音频文件
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                audio_files.append(os.path.join(root, file))
    
    print(f"找到 {len(audio_files)} 个音频文件需要处理")
    
    # 预测每个文件
    for i, file_path in enumerate(audio_files):
        print(f"[{i+1}/{len(audio_files)}] 预测 {file_path}...")
        
        try:
            prediction = system.predict_from_file(file_path, verbose=False)
            results[file_path] = prediction
            print(f"预测结果: '{prediction}'")
        except Exception as e:
            print(f"处理 {file_path} 时出错: {str(e)}")
            results[file_path] = f"ERROR: {str(e)}"
    
    # 保存结果
    with open(output_file, 'w') as f:
        for file_path, prediction in results.items():
            f.write(f"{file_path}\t{prediction}\n")
    
    print(f"批量预测完成，结果已保存至 {output_file}")

if __name__ == "__main__":
    main()