#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# examples/neural_network_demo.py

"""
神经网络模型训练演示 - 使用真实MR2数据
与traditional_ml_demo.py保持一致的命名和结构
修复了模块导入和数据加载问题，确保使用真实数据
"""

import sys
import os
import json
import time
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')

# 添加项目路径
current_file = Path(__file__).resolve()
code_root = current_file.parent.parent
sys.path.append(str(code_root))

print("🧠 MR2神经网络模型训练演示")
print("="*50)

class TextDataset(Dataset):
    """文本数据集类"""
    
    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 简单分词
        tokens = text.lower().split()
        
        # 转换为索引
        indices = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens]
        
        # 截断或填充
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([self.vocab.get('<PAD>', 0)] * (self.max_length - len(indices)))
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'attention_mask': torch.tensor([1 if i != 0 else 0 for i in indices], dtype=torch.long)
        }

class TextCNN(nn.Module):
    """TextCNN模型"""
    
    def __init__(self, vocab_size, embedding_dim=128, filter_sizes=[3, 4, 5], 
                 num_filters=64, num_classes=3, dropout=0.5):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = embedded.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        concatenated = torch.cat(conv_outputs, dim=1)
        output = self.dropout(concatenated)
        logits = self.fc(output)
        
        return logits

class BiLSTM(nn.Module):
    """双向LSTM模型"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, 
                 num_layers=2, num_classes=3, dropout=0.5):
        super(BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            batch_size = lstm_out.size(0)
            last_outputs = lstm_out[range(batch_size), lengths]
        else:
            last_outputs = lstm_out[:, -1, :]
        
        output = self.dropout(last_outputs)
        logits = self.fc(output)
        
        return logits

class TextRCNN(nn.Module):
    """Text-RCNN模型"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, 
                 num_classes=3, dropout=0.5):
        super(TextRCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.context_weight = nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        combined = torch.cat([embedded, lstm_out], dim=2)
        context = torch.tanh(self.context_weight(combined))
        pooled = F.max_pool1d(context.transpose(1, 2), context.size(1)).squeeze(2)
        output = self.dropout(pooled)
        logits = self.fc(output)
        
        return logits

def check_real_data():
    """检查真实数据是否存在"""
    print("\n📁 检查真实数据文件...")
    
    data_dir = code_root / "data"
    required_files = [
        "dataset_items_train.json",
        "dataset_items_val.json", 
        "dataset_items_test.json"
    ]
    
    existing_files = []
    
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            file_size = file_path.stat().st_size
            existing_files.append((file_name, file_size))
            print(f"   ✅ {file_name} ({file_size/1024:.1f} KB)")
        else:
            print(f"   ❌ {file_name} - 文件不存在")
            return False
    
    print(f"\n✅ 所有数据文件就绪! 共 {len(existing_files)} 个文件")
    return True

def load_real_mr2_data():
    """直接加载真实MR2数据"""
    print("\n📚 加载真实MR2数据...")
    
    data_dir = code_root / "data"
    datasets = {}
    total_samples = 0
    
    for split in ['train', 'val', 'test']:
        file_path = data_dir / f'dataset_items_{split}.json'
        
        print(f"📂 读取 {split} 数据: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        texts = []
        labels = []
        
        for item_id, item_data in raw_data.items():
            if 'caption' in item_data and 'label' in item_data:
                text = item_data['caption']
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                    labels.append(item_data['label'])
        
        datasets[split] = (texts, labels)
        total_samples += len(texts)
        
        # 显示标签分布
        label_dist = Counter(labels)
        print(f"   ✅ {split}: {len(texts)} 样本")
        print(f"   📊 标签分布: {dict(label_dist)}")
    
    print(f"\n📊 总样本数: {total_samples}")
    return datasets

def preprocess_text(text):
    """简单文本预处理"""
    import re
    
    if not isinstance(text, str):
        return ""
    
    # 基本清理
    text = re.sub(r'http\S+', '', text)  # 移除URL
    text = re.sub(r'@\w+', '', text)     # 移除@提及
    text = re.sub(r'#\w+', '', text)     # 移除#标签
    text = re.sub(r'\s+', ' ', text)     # 标准化空白
    text = text.strip().lower()
    
    return text

def build_vocabulary(texts, min_freq=2, max_vocab_size=5000):
    """构建词汇表"""
    print("📖 构建词汇表...")
    
    word_freq = Counter()
    for text in texts:
        processed_text = preprocess_text(text)
        words = processed_text.split()
        word_freq.update(words)
    
    # 按频率排序
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in sorted_words:
        if freq >= min_freq and len(vocab) < max_vocab_size:
            vocab[word] = len(vocab)
    
    print(f"✅ 词汇表大小: {len(vocab)}")
    return vocab

def train_single_model(model, train_loader, val_loader, device, epochs=10, learning_rate=0.001):
    """训练单个模型"""
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
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
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
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
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
    
    return {
        'best_val_accuracy': best_val_acc,
        'val_f1_score': val_f1,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }

def evaluate_model(model, test_loader, device):
    """评估模型在测试集上的性能"""
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            logits = model(input_ids, attention_mask)
            _, predicted = torch.max(logits.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # 分类报告
    label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
    report = classification_report(
        all_labels, all_preds,
        target_names=list(label_mapping.values()),
        output_dict=True
    )
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'test_accuracy': test_acc,
        'test_f1_score': test_f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

def train_neural_networks(datasets):
    """训练神经网络模型"""
    print("\n🧠 开始训练神经网络模型...")
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   显存: {gpu_memory:.1f} GB")
    
    # 准备数据
    train_texts, train_labels = datasets['train']
    val_texts, val_labels = datasets['val']
    test_texts, test_labels = datasets['test']
    
    # 预处理文本
    print("🔧 预处理文本数据...")
    train_texts = [preprocess_text(text) for text in train_texts]
    val_texts = [preprocess_text(text) for text in val_texts]
    test_texts = [preprocess_text(text) for text in test_texts]
    
    # 构建词汇表
    all_texts = train_texts + val_texts + test_texts
    vocab = build_vocabulary(train_texts, min_freq=2, max_vocab_size=3000)
    
    print(f"✅ 预处理完成")
    print(f"   训练集: {len(train_texts)} 样本")
    print(f"   验证集: {len(val_texts)} 样本")
    print(f"   测试集: {len(test_texts)} 样本")
    print(f"   词汇表大小: {len(vocab)}")
    
    # 创建数据集和数据加载器
    max_length = 64  # 适中的序列长度
    batch_size = 16
    
    train_dataset = TextDataset(train_texts, train_labels, vocab, max_length)
    val_dataset = TextDataset(val_texts, val_labels, vocab, max_length)
    test_dataset = TextDataset(test_texts, test_labels, vocab, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 定义模型
    vocab_size = len(vocab)
    models = {
        'TextCNN': TextCNN(vocab_size=vocab_size, embedding_dim=128, num_filters=64, num_classes=3),
        'BiLSTM': BiLSTM(vocab_size=vocab_size, embedding_dim=128, hidden_dim=64, num_classes=3),
        'TextRCNN': TextRCNN(vocab_size=vocab_size, embedding_dim=128, hidden_dim=64, num_classes=3)
    }
    
    # 移动模型到设备并显示参数量
    for name, model in models.items():
        model.to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   {name}: {param_count:,} 参数")
    
    results = {}
    
    # 训练每个模型
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"训练模型: {model_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 训练模型
        train_result = train_single_model(
            model, train_loader, val_loader, device,
            epochs=12, learning_rate=0.001
        )
        
        # 测试评估
        test_result = evaluate_model(model, test_loader, device)
        
        training_time = time.time() - start_time
        
        # 合并结果
        results[model_name] = {
            **train_result,
            **test_result,
            'training_time': training_time,
            'model_parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"✅ {model_name} 训练完成 (耗时: {training_time:.1f}秒)")
        print(f"   验证准确率: {train_result['best_val_accuracy']:.2f}%")
        print(f"   测试准确率: {test_result['test_accuracy']:.2f}%")
        print(f"   测试F1分数: {test_result['test_f1_score']:.4f}")
    
    return results, vocab, models

def save_results(results, vocab, models):
    """保存训练结果"""
    print("\n💾 保存训练结果...")
    
    # 创建输出目录
    output_dir = code_root / 'outputs' / 'neural_network_demo'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存词汇表
    vocab_file = output_dir / 'vocabulary.pkl'
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"✅ 保存词汇表: {vocab_file}")
    
    # 保存模型
    for model_name, model in models.items():
        model_file = output_dir / f'{model_name.lower()}_model.pth'
        torch.save(model.state_dict(), model_file)
        print(f"✅ 保存模型: {model_file}")
    
    # 保存结果JSON
    results_file = output_dir / 'training_results.json'
    
    # 转换numpy类型为可序列化类型
    serializable_results = {}
    for model_name, result in results.items():
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, (list, np.ndarray)):
                serializable_result[key] = list(value) if hasattr(value, 'tolist') else value
            else:
                serializable_result[key] = value
        serializable_results[model_name] = serializable_result
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 保存结果: {results_file}")
    
    # 保存CSV比较
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Val_Accuracy': result['best_val_accuracy'],
            'Test_Accuracy': result['test_accuracy'],
            'Val_F1': result['val_f1_score'],
            'Test_F1': result['test_f1_score'],
            'Parameters': result['model_parameters'],
            'Training_Time': result['training_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = output_dir / 'model_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"✅ 保存比较: {comparison_file}")
    
    return output_dir

def display_final_results(results):
    """显示最终结果"""
    print(f"\n📊 {'='*80}")
    print("神经网络模型性能对比")
    print(f"{'='*80}")
    
    print(f"{'模型':<12} {'验证准确率':<10} {'测试准确率':<10} {'验证F1':<8} {'测试F1':<8} {'参数量':<10} {'训练时间':<10}")
    print("-" * 80)
    
    # 按测试F1分数排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_f1_score'], reverse=True)
    
    for model_name, result in sorted_results:
        print(f"{model_name:<12} "
              f"{result['best_val_accuracy']:<10.2f} "
              f"{result['test_accuracy']:<10.2f} "
              f"{result['val_f1_score']:<8.4f} "
              f"{result['test_f1_score']:<8.4f} "
              f"{result['model_parameters']:<10,} "
              f"{result['training_time']:<10.1f}")
    
    # 最佳模型
    best_model_name, best_result = sorted_results[0]
    print(f"\n🏆 最佳模型: {best_model_name}")
    print(f"   测试准确率: {best_result['test_accuracy']:.2f}%")
    print(f"   测试F1分数: {best_result['test_f1_score']:.4f}")
    print(f"   训练时间: {best_result['training_time']:.1f}秒")
    
    # 显示详细分类报告
    print(f"\n📋 {best_model_name} 详细分类报告:")
    report = best_result['classification_report']
    
    label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
    
    print(f"{'类别':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<8}")
    print("-" * 50)
    
    for i, label_name in label_mapping.items():
        if str(i) in report:
            metrics = report[str(i)]
            print(f"{label_name:<12} "
                  f"{metrics['precision']:<8.3f} "
                  f"{metrics['recall']:<8.3f} "
                  f"{metrics['f1-score']:<8.3f} "
                  f"{int(metrics['support']):<8}")
    
    # 总体指标
    if 'macro avg' in report:
        macro_avg = report['macro avg']
        print("-" * 50)
        print(f"{'Macro Avg':<12} "
              f"{macro_avg['precision']:<8.3f} "
              f"{macro_avg['recall']:<8.3f} "
              f"{macro_avg['f1-score']:<8.3f} "
              f"{int(macro_avg['support']):<8}")

def analyze_results(results):
    """分析结果并提供建议"""
    print(f"\n🔍 结果分析:")
    
    # 获取所有F1分数
    f1_scores = [result['test_f1_score'] for result in results.values()]
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"   平均测试F1分数: {avg_f1:.4f} (±{std_f1:.4f})")
    
    if avg_f1 > 0.8:
        print("   🎉 整体性能优秀!")
    elif avg_f1 > 0.7:
        print("   👍 整体性能良好!")
    elif avg_f1 > 0.6:
        print("   💡 性能中等，有提升空间")
    else:
        print("   ⚠️  性能较低，需要优化")
    
    print(f"\n💡 优化建议:")
    if avg_f1 < 0.8:
        print("   1. 增加训练轮数")
        print("   2. 调整学习率和批次大小") 
        print("   3. 使用预训练词嵌入 (Word2Vec, GloVe)")
        print("   4. 尝试更深的网络结构")
        print("   5. 添加正则化技术 (BatchNorm, LayerNorm)")
    
    print("   6. 尝试预训练Transformer模型 (BERT, RoBERTa)")
    print("   7. 实验多模态融合 (文本+图像)")

def main():
    """主函数"""
    print("欢迎使用神经网络模型训练演示!")
    print("本演示将使用真实MR2数据训练神经网络模型\n")
    
    # 1. 检查数据
    if not check_real_data():
        print("❌ 请确保真实数据文件存在于 data/ 目录")
        return
    
    # 2. 加载真实数据
    datasets = load_real_mr2_data()
    
    # 3. 训练模型
    start_time = time.time()
    results, vocab, models = train_neural_networks(datasets)
    total_time = time.time() - start_time
    
    # 4. 保存结果
    output_dir = save_results(results, vocab, models)
    
    # 5. 显示结果
    display_final_results(results)
    
    # 6. 分析结果
    analyze_results(results)
    
    # 7. 总结
    print(f"\n🎉 === 神经网络模型训练完成 ===")
    print(f"✅ 总训练时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"✅ 共训练 {len(models)} 个模型")
    print(f"📁 结果已保存到: {output_dir}")
    
    print(f"\n🚀 下一步建议:")
    print("   1. 查看生成的CSV文件分析详细结果")
    print("   2. 尝试调整超参数提升性能")
    print("   3. 与传统ML模型结果进行对比")
    print("   4. 实验预训练模型和多模态融合")

if __name__ == "__main__":
    main()