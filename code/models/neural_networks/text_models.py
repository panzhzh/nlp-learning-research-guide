#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/neural_networks/text_models.py

"""
文本神经网络模型实现
包含TextCNN、BiLSTM、TextRCNN等经典文本分类模型
支持多语言文本和MR2数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_file = Path(__file__).resolve()
code_root = current_file.parent.parent.parent
sys.path.append(str(code_root))

# 导入项目模块
try:
    from datasets.data_loaders import create_all_dataloaders
    from utils.config_manager import get_config_manager, get_output_path
    from preprocessing.text_processing import TextProcessor
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块")
except ImportError as e:
    print(f"⚠️  导入项目模块失败: {e}")
    USE_PROJECT_MODULES = False

import logging
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """文本数据集类"""
    
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], 
                 max_length: int = 256):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            vocab: 词汇表
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 将文本转换为索引序列
        tokens = text.split()
        indices = [self.vocab.get(token, self.vocab.get('<UNK>', 0)) for token in tokens]
        
        # 截断或填充
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([self.vocab.get('<PAD>', 0)] * (self.max_length - len(indices)))
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'attention_mask': torch.tensor([1 if i != self.vocab.get('<PAD>', 0) else 0 for i in indices], dtype=torch.long)
        }


class TextCNN(nn.Module):
    """TextCNN模型"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300, 
                 filter_sizes: List[int] = [3, 4, 5], num_filters: int = 100,
                 num_classes: int = 3, dropout: float = 0.5):
        """
        初始化TextCNN
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            filter_sizes: 卷积核尺寸列表
            num_filters: 每种尺寸的卷积核数量
            num_classes: 分类数量
            dropout: dropout概率
        """
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 多个卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        # 词嵌入: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        
        # 转置为卷积输入格式: (batch_size, embedding_dim, seq_len)
        embedded = embedded.transpose(1, 2)
        
        # 多尺度卷积和池化
        conv_outputs = []
        for conv in self.convs:
            # 卷积: (batch_size, num_filters, new_seq_len)
            conv_out = F.relu(conv(embedded))
            # 最大池化: (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # 拼接所有卷积结果: (batch_size, len(filter_sizes) * num_filters)
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Dropout和全连接
        output = self.dropout(concatenated)
        logits = self.fc(output)
        
        return logits


class BiLSTM(nn.Module):
    """双向LSTM模型"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300,
                 hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 3, dropout: float = 0.5):
        """
        初始化BiLSTM
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类数量
            dropout: dropout概率
        """
        super(BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 因为是双向
        
    def forward(self, input_ids, attention_mask=None):
        # 词嵌入: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        
        # LSTM: output (batch_size, seq_len, hidden_dim * 2)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最后一个时间步的输出（考虑padding）
        if attention_mask is not None:
            # 获取每个序列的实际长度
            lengths = attention_mask.sum(dim=1) - 1  # -1因为索引从0开始
            lengths = lengths.clamp(min=0)
            
            # 提取每个序列的最后一个有效输出
            batch_size = lstm_out.size(0)
            last_outputs = lstm_out[range(batch_size), lengths]
        else:
            # 如果没有attention_mask，使用最后一个时间步
            last_outputs = lstm_out[:, -1, :]
        
        # Dropout和全连接
        output = self.dropout(last_outputs)
        logits = self.fc(output)
        
        return logits


class TextRCNN(nn.Module):
    """Text-RCNN模型（结合RNN和CNN）"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300,
                 hidden_dim: int = 128, num_classes: int = 3, dropout: float = 0.5):
        """
        初始化TextRCNN
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: RNN隐藏层维度
            num_classes: 分类数量
            dropout: dropout概率
        """
        super(TextRCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # 上下文表示的线性变换
        self.context_weight = nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        # 词嵌入: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        
        # LSTM: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(embedded)
        
        # 拼接词嵌入和LSTM输出: (batch_size, seq_len, hidden_dim * 2 + embedding_dim)
        combined = torch.cat([embedded, lstm_out], dim=2)
        
        # 上下文表示: (batch_size, seq_len, hidden_dim)
        context = torch.tanh(self.context_weight(combined))
        
        # 最大池化: (batch_size, hidden_dim)
        pooled = F.max_pool1d(context.transpose(1, 2), context.size(1)).squeeze(2)
        
        # Dropout和全连接
        output = self.dropout(pooled)
        logits = self.fc(output)
        
        return logits


class NeuralTextClassifier:
    """神经网络文本分类器训练器"""
    
    def __init__(self, data_dir: str = "data", device: str = "auto"):
        """
        初始化训练器
        
        Args:
            data_dir: 数据目录路径
            device: 计算设备
        """
        self.data_dir = data_dir
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  使用设备: {self.device}")
        
        # 初始化组件
        self.vocab = {}
        self.models = {}
        self.results = {}
        
        # 初始化文本处理器
        if USE_PROJECT_MODULES:
            self.text_processor = TextProcessor(language='mixed')
        else:
            self.text_processor = None
        
        # 设置输出目录
        if USE_PROJECT_MODULES:
            config_manager = get_config_manager()
            self.output_dir = get_output_path('models', 'neural_networks')
        else:
            self.output_dir = Path('outputs/models/neural_networks')
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 标签映射
        self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
        
        print(f"🧠 神经网络分类器训练器初始化完成")
        print(f"   数据目录: {self.data_dir}")
        print(f"   输出目录: {self.output_dir}")
    
    def load_data(self) -> Dict[str, Tuple[List[str], List[int]]]:
        """加载MR2数据集"""
        print("📚 加载MR2数据集...")
        
        if USE_PROJECT_MODULES:
            try:
                # 使用项目的数据加载器
                dataloaders = create_all_dataloaders(
                    data_dir=self.data_dir,
                    batch_sizes={'train': 32, 'val': 32, 'test': 32}
                )
                
                data = {}
                for split, dataloader in dataloaders.items():
                    texts = []
                    labels = []
                    
                    for batch in dataloader:
                        # 提取文本和标签
                        if 'text' in batch:
                            texts.extend(batch['text'])
                        elif 'caption' in batch:
                            texts.extend(batch['caption'])
                        
                        if 'labels' in batch:
                            labels.extend(batch['labels'].tolist())
                        elif 'label' in batch:
                            labels.extend(batch['label'])
                    
                    data[split] = (texts, labels)
                    print(f"✅ 加载 {split}: {len(texts)} 样本")
                
                return data
                
            except Exception as e:
                print(f"❌ 使用项目数据加载器失败: {e}")
                return self._create_demo_data()
        else:
            return self._create_demo_data()
    
    def _create_demo_data(self) -> Dict[str, Tuple[List[str], List[int]]]:
        """创建演示数据"""
        print("🔧 创建演示数据...")
        
        demo_texts = [
            "这是一个关于科技进步的真实新闻报道，包含了详细的技术细节",
            "This is fake news about celebrity scandal without any credible sources",
            "未经证实的传言需要进一步调查验证，目前无法确定真伪",
            "Breaking news: Major breakthrough in artificial intelligence technology announced by researchers",
            "网传某地发生重大事故，官方尚未确认消息真实性",
            "Scientists discover new species in deep ocean with advanced submarine technology",
            "谣传某知名公司即将倒闭，但公司官方已辟谣此消息",
            "Weather alert: Severe storm approaching coastal areas according to meteorological department",
            "社交媒体广泛流传的未证实消息引发公众关注和讨论",
            "Economic indicators show positive growth trends in multiple sectors this quarter",
            "新研究表明气候变化对生态系统产生深远影响",
            "Unverified claims about health benefits of new supplement spread online",
            "政府发布官方声明澄清网络传言并提供准确信息",
            "False information about vaccine side effects causes public concern",
            "专家呼吁公众理性对待网络信息，避免传播谣言"
        ]
        
        demo_labels = [0, 1, 2, 0, 2, 0, 1, 0, 2, 0, 0, 2, 0, 1, 0]
        
        # 扩展数据以便训练
        extended_texts = demo_texts * 8
        extended_labels = demo_labels * 8
        
        # 按比例分割数据
        total_size = len(extended_texts)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        
        return {
            'train': (extended_texts[:train_size], extended_labels[:train_size]),
            'val': (extended_texts[train_size:train_size+val_size], 
                   extended_labels[train_size:train_size+val_size]),
            'test': (extended_texts[train_size+val_size:], 
                    extended_labels[train_size+val_size:])
        }
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """预处理文本"""
        if self.text_processor:
            processed_texts = []
            for text in texts:
                cleaned_text = self.text_processor.clean_text(text)
                tokens = self.text_processor.tokenize(cleaned_text)
                processed_text = ' '.join(tokens) if tokens else cleaned_text
                processed_texts.append(processed_text)
            return processed_texts
        else:
            # 简单的文本清理
            import re
            processed_texts = []
            for text in texts:
                text = re.sub(r'http\S+', '', text)
                text = re.sub(r'@\w+', '', text)
                text = re.sub(r'#\w+', '', text)
                text = re.sub(r'\s+', ' ', text)
                text = text.strip().lower()
                processed_texts.append(text)
            return processed_texts
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 2, max_vocab_size: int = 10000) -> Dict[str, int]:
        """构建词汇表"""
        print("📖 构建词汇表...")
        
        # 统计词频
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序并构建词汇表
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, freq in sorted_words:
            if freq >= min_freq and len(vocab) < max_vocab_size:
                vocab[word] = len(vocab)
        
        print(f"✅ 词汇表构建完成，大小: {len(vocab)}")
        return vocab
    
    def create_models(self):
        """创建神经网络模型"""
        print("🧠 创建神经网络模型...")
        
        vocab_size = len(self.vocab)
        
        # TextCNN
        self.models['textcnn'] = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=128,  # 减小维度以适应小数据集
            filter_sizes=[3, 4, 5],
            num_filters=64,  # 减少滤波器数量
            num_classes=3,
            dropout=0.5
        ).to(self.device)
        
        # BiLSTM
        self.models['bilstm'] = BiLSTM(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=64,  # 减小隐藏层维度
            num_layers=2,
            num_classes=3,
            dropout=0.5
        ).to(self.device)
        
        # TextRCNN
        self.models['textrcnn'] = TextRCNN(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=64,
            num_classes=3,
            dropout=0.5
        ).to(self.device)
        
        print(f"✅ 创建了 {len(self.models)} 个神经网络模型")
        
        # 打印模型参数量
        for name, model in self.models.items():
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   {name}: {param_count:,} 参数")
    
    def train_single_model(self, model_name: str, train_loader: DataLoader, 
                          val_loader: DataLoader, epochs: int = 10, 
                          learning_rate: float = 0.001) -> Dict[str, Any]:
        """训练单个模型"""
        print(f"🏋️ 训练 {model_name} 模型...")
        
        model = self.models[model_name]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_accuracies = []
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                optimizer.zero_grad()
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100*train_correct/train_total:.2f}%'
                })
            
            # 验证阶段
            model.eval()
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    
                    logits = model(input_ids, attention_mask)
                    _, predicted = torch.max(logits.data, 1)
                    
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
        
        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 计算最终验证集F1分数
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        result = {
            'model_name': model_name,
            'best_val_accuracy': best_val_acc,
            'val_f1_score': val_f1,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'epochs_trained': epochs
        }
        
        print(f"✅ {model_name} 训练完成:")
        print(f"   最佳验证准确率: {best_val_acc:.2f}%")
        print(f"   验证F1分数: {val_f1:.4f}")
        
        return result
    
    def evaluate_model(self, model_name: str, test_loader: DataLoader) -> Dict[str, Any]:
        """评估模型在测试集上的性能"""
        print(f"📊 评估 {model_name} 模型...")
        
        model = self.models[model_name]
        model.eval()
        
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                logits = model(input_ids, attention_mask)
                _, predicted = torch.max(logits.data, 1)
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = 100 * test_correct / test_total
        test_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # 分类报告
        report = classification_report(
            all_labels, all_preds,
            target_names=list(self.label_mapping.values()),
            output_dict=True
        )
        
        result = {
            'test_accuracy': test_acc,
            'test_f1_score': test_f1,
            'classification_report': report
        }
        
        print(f"✅ {model_name} 测试结果:")
        print(f"   测试准确率: {test_acc:.2f}%")
        print(f"   测试F1分数: {test_f1:.4f}")
        
        return result
    
    def train_all_models(self, epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001):
        """训练所有神经网络模型"""
        print("🚀 开始训练所有神经网络模型...")
        
        # 加载数据
        data = self.load_data()
        
        # 预处理文本
        all_texts = data['train'][0] + data['val'][0] + data['test'][0]
        all_preprocessed = self.preprocess_texts(all_texts)
        
        # 构建词汇表
        train_preprocessed = self.preprocess_texts(data['train'][0])
        self.vocab = self.build_vocabulary(train_preprocessed)
        
        # 创建数据集
        train_dataset = TextDataset(train_preprocessed, data['train'][1], self.vocab)
        val_dataset = TextDataset(self.preprocess_texts(data['val'][0]), data['val'][1], self.vocab)
        test_dataset = TextDataset(self.preprocess_texts(data['test'][0]), data['test'][1], self.vocab)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        self.create_models()
        
        # 训练每个模型
        for model_name in self.models.keys():
            print(f"\n{'='*60}")
            print(f"训练模型: {model_name.upper()}")
            print(f"{'='*60}")
            
            # 训练模型
            train_result = self.train_single_model(
                model_name, train_loader, val_loader, epochs, learning_rate
            )
            
            # 测试评估
            test_result = self.evaluate_model(model_name, test_loader)
            
            # 合并结果
            self.results[model_name] = {**train_result, **test_result}
        
        # 保存模型和结果
        self.save_models_and_results()
        
        # 显示最终对比
        self.print_model_comparison()
    
    def save_models_and_results(self):
        """保存训练好的模型和结果"""
        print("\n💾 保存模型和结果...")
        
        # 保存词汇表
        vocab_file = self.output_dir / 'vocabulary.pkl'
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.vocab, f)
        print(f"✅ 保存词汇表: {vocab_file}")
        
        # 保存每个模型
        for model_name, model in self.models.items():
            model_file = self.output_dir / f'{model_name}_model.pth'
            torch.save(model.state_dict(), model_file)
            print(f"✅ 保存模型: {model_file}")
        
        # 保存结果
        results_file = self.output_dir / 'training_results.json'
        # 转换numpy类型为可序列化的类型
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (list, np.ndarray)):
                    serializable_result[key] = list(value) if hasattr(value, 'tolist') else value
                else:
                    serializable_result[key] = value
            serializable_results[model_name] = serializable_result
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"✅ 保存训练结果: {results_file}")
        
        # 保存模型比较
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Best_Val_Accuracy': result['best_val_accuracy'],
                'Test_Accuracy': result['test_accuracy'],
                'Val_F1': result['val_f1_score'],
                'Test_F1': result['test_f1_score'],
                'Parameters': sum(p.numel() for p in self.models[model_name].parameters())
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = self.output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        print(f"✅ 保存模型比较: {comparison_file}")
    
    def print_model_comparison(self):
        """打印模型比较结果"""
        print(f"\n📊 {'='*70}")
        print("神经网络模型性能对比")
        print(f"{'='*70}")
        
        print(f"{'模型':<12} {'验证准确率':<10} {'测试准确率':<10} {'验证F1':<8} {'测试F1':<8} {'参数量':<10}")
        print("-" * 70)
        
        # 按测试F1分数排序
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['test_f1_score'], 
                              reverse=True)
        
        for model_name, result in sorted_results:
            param_count = sum(p.numel() for p in self.models[model_name].parameters())
            print(f"{model_name:<12} "
                  f"{result['best_val_accuracy']:<10.2f} "
                  f"{result['test_accuracy']:<10.2f} "
                  f"{result['val_f1_score']:<8.4f} "
                  f"{result['test_f1_score']:<8.4f} "
                  f"{param_count:<10,}")
        
        # 找出最佳模型
        best_model = sorted_results[0]
        print(f"\n🏆 最佳模型: {best_model[0]}")
        print(f"   测试F1分数: {best_model[1]['test_f1_score']:.4f}")
        print(f"   测试准确率: {best_model[1]['test_accuracy']:.2f}%")


def main():
    """主函数，演示训练流程"""
    print("🚀 神经网络文本分类器训练演示")
    
    # 创建训练器
    trainer = NeuralTextClassifier(data_dir="data")
    
    # 训练所有模型
    trainer.train_all_models(
        epochs=15,          # 训练轮数
        batch_size=16,      # 批次大小
        learning_rate=0.001 # 学习率
    )
    
    print("\n✅ 神经网络模型训练演示完成!")
    print(f"📁 模型和结果已保存到: {trainer.output_dir}")


if __name__ == "__main__":
    main()