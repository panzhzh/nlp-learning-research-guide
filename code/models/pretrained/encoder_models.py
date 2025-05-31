#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/pretrained/encoder_models.py

"""
预训练编码器模型实现
支持BERT、RoBERTa、ALBERT、DeBERTa等主流预训练模型
完全复用现有的数据加载和训练框架
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel,
    AlbertTokenizer, AlbertModel,
    DebertaTokenizer, DebertaModel
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import json
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


class PretrainedTextDataset:
    """预训练模型专用的文本数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 预训练模型的tokenizer
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用tokenizer编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class PretrainedClassifier(nn.Module):
    """通用的预训练模型分类器"""
    
    def __init__(self, model_name: str, num_classes: int = 3, dropout: float = 0.1):
        """
        初始化预训练分类器
        
        Args:
            model_name: 预训练模型名称
            num_classes: 分类数量
            dropout: dropout概率
        """
        super(PretrainedClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # 加载预训练模型和配置
        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
        except Exception as e:
            print(f"⚠️  加载预训练模型失败，使用备用模型: {e}")
            # 备用模型配置
            self.config = AutoConfig.from_pretrained('bert-base-uncased')
            self.bert = AutoModel.from_pretrained('bert-base-uncased')
        
        # 分类头
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化分类头权重"""
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            
        Returns:
            分类logits
        """
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS]token的表示
        pooled_output = outputs.pooler_output
        
        # 应用dropout和分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class PretrainedModelTrainer:
    """预训练模型训练器"""
    
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
        self.models = {}
        self.tokenizers = {}
        self.results = {}
        
        # 设置输出目录
        if USE_PROJECT_MODULES:
            config_manager = get_config_manager()
            self.output_dir = get_output_path('models', 'pretrained')
        else:
            self.output_dir = Path('outputs/models/pretrained')
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 标签映射
        self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
        
        # 支持的模型配置
        self.model_configs = {
            'bert-base-uncased': {
                'name': 'bert-base-uncased',
                'tokenizer_class': BertTokenizer,
                'model_class': BertModel,
                'description': 'BERT Base (Uncased)'
            },
            'roberta-base': {
                'name': 'roberta-base',
                'tokenizer_class': RobertaTokenizer,
                'model_class': RobertaModel,
                'description': 'RoBERTa Base'
            },
            'albert-base-v2': {
                'name': 'albert-base-v2',
                'tokenizer_class': AlbertTokenizer,
                'model_class': AlbertModel,
                'description': 'ALBERT Base v2'
            },
            'chinese-bert-wwm': {
                'name': 'hfl/chinese-bert-wwm-ext',
                'tokenizer_class': BertTokenizer,
                'model_class': BertModel,
                'description': 'Chinese BERT (Whole Word Masking)'
            }
        }
        
        print(f"🤖 预训练模型训练器初始化完成")
        print(f"   数据目录: {self.data_dir}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   支持模型: {list(self.model_configs.keys())}")
    
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
            "这是一个关于科技进步的真实新闻报道，包含了详细的技术细节和权威来源",
            "This is a fake news about celebrity scandal without any credible sources or verification",
            "未经证实的传言需要进一步调查验证，目前无法确定真伪，请等待官方消息",
            "Breaking news: Major breakthrough in artificial intelligence technology announced by leading researchers",
            "网传某地发生重大事故，但官方尚未确认，请以权威媒体报道为准",
            "Scientists discover new species in deep ocean using advanced submarine technology and equipment",
            "谣传某知名公司即将倒闭，但公司官方已辟谣，股价保持稳定",
            "Weather alert: Severe storm approaching coastal areas according to national meteorological department",
            "社交媒体广泛流传的未证实消息引发公众关注，专家建议理性对待",
            "Economic indicators show positive growth trends in multiple sectors this quarter",
            "新研究表明气候变化对全球生态系统产生深远影响，需要采取紧急行动",
            "Unverified claims about health benefits of new supplement spread online without scientific backing",
            "政府发布官方声明澄清网络传言并提供准确信息和数据支持",
            "False information about vaccine side effects causes public concern among healthcare professionals",
            "专家呼吁公众理性对待网络信息，避免传播谣言和虚假消息",
            "Technology companies announce new privacy policies following recent data security incidents",
            "教育部发布新政策支持在线教育发展，提高教学质量和覆盖面",
            "International cooperation strengthens global efforts to combat climate change effectively",
            "网络安全专家警告新型网络攻击手段，建议用户提高防范意识",
            "Medical research shows promising results for new treatment methods in clinical trials"
        ]
        
        demo_labels = [0, 1, 2, 0, 2, 0, 1, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 0, 2, 0]
        
        # 扩展数据
        extended_texts = demo_texts * 6
        extended_labels = demo_labels * 6
        
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
    
    def create_model(self, model_key: str):
        """创建预训练模型"""
        print(f"🧠 创建模型: {model_key}")
        
        if model_key not in self.model_configs:
            raise ValueError(f"不支持的模型: {model_key}")
        
        config = self.model_configs[model_key]
        model_name = config['name']
        
        try:
            # 创建tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 创建模型
            model = PretrainedClassifier(
                model_name=model_name,
                num_classes=3,
                dropout=0.1
            ).to(self.device)
            
            self.tokenizers[model_key] = tokenizer
            self.models[model_key] = model
            
            # 打印模型参数量
            param_count = sum(p.numel() for p in model.parameters())
            print(f"✅ {config['description']}: {param_count:,} 参数")
            
        except Exception as e:
            print(f"❌ 创建模型失败: {model_key}, 错误: {e}")
            # 创建备用简单模型
            self._create_fallback_model(model_key)
    
    def _create_fallback_model(self, model_key: str):
        """创建备用简单模型"""
        print(f"🔄 创建备用模型: {model_key}")
        
        try:
            # 使用bert-base-uncased作为备用
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = PretrainedClassifier(
                model_name='bert-base-uncased',
                num_classes=3
            ).to(self.device)
            
            self.tokenizers[model_key] = tokenizer
            self.models[model_key] = model
            
            print(f"✅ 备用模型创建成功: {model_key}")
            
        except Exception as e:
            print(f"❌ 备用模型创建失败: {e}")
    
    def train_single_model(self, model_key: str, train_loader: DataLoader, 
                          val_loader: DataLoader, epochs: int = 3, 
                          learning_rate: float = 2e-5) -> Dict[str, Any]:
        """训练单个模型"""
        print(f"🏋️ 训练 {model_key} 模型...")
        
        model = self.models[model_key]
        
        # 设置优化器
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
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
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
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
        
        # 计算最终F1分数
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        result = {
            'model_name': model_key,
            'model_description': self.model_configs[model_key]['description'],
            'best_val_accuracy': best_val_acc,
            'val_f1_score': val_f1,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'epochs_trained': epochs
        }
        
        print(f"✅ {model_key} 训练完成:")
        print(f"   最佳验证准确率: {best_val_acc:.2f}%")
        print(f"   验证F1分数: {val_f1:.4f}")
        
        return result
    
    def evaluate_model(self, model_key: str, test_loader: DataLoader) -> Dict[str, Any]:
        """评估模型在测试集上的性能"""
        print(f"📊 评估 {model_key} 模型...")
        
        model = self.models[model_key]
        model.eval()
        
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
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
        
        print(f"✅ {model_key} 测试结果:")
        print(f"   测试准确率: {test_acc:.2f}%")
        print(f"   测试F1分数: {test_f1:.4f}")
        
        return result
    
    def train_all_models(self, model_keys: Optional[List[str]] = None, 
                        epochs: int = 3, batch_size: int = 16, 
                        learning_rate: float = 2e-5, max_length: int = 512):
        """训练所有或指定的预训练模型"""
        print("🚀 开始训练预训练模型...")
        
        # 默认训练所有模型
        if model_keys is None:
            model_keys = list(self.model_configs.keys())
        
        # 加载数据
        data = self.load_data()
        
        # 为每个模型创建数据加载器
        for model_key in model_keys:
            print(f"\n{'='*60}")
            print(f"训练模型: {model_key.upper()}")
            print(f"{'='*60}")
            
            try:
                # 创建模型
                self.create_model(model_key)
                
                if model_key not in self.models:
                    print(f"❌ 模型创建失败，跳过: {model_key}")
                    continue
                
                tokenizer = self.tokenizers[model_key]
                
                # 创建数据集
                train_dataset = PretrainedTextDataset(
                    data['train'][0], data['train'][1], tokenizer, max_length
                )
                val_dataset = PretrainedTextDataset(
                    data['val'][0], data['val'][1], tokenizer, max_length
                )
                test_dataset = PretrainedTextDataset(
                    data['test'][0], data['test'][1], tokenizer, max_length
                )
                
                # 创建数据加载器
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # 训练模型
                train_result = self.train_single_model(
                    model_key, train_loader, val_loader, epochs, learning_rate
                )
                
                # 测试评估
                test_result = self.evaluate_model(model_key, test_loader)
                
                # 合并结果
                self.results[model_key] = {**train_result, **test_result}
                
            except Exception as e:
                print(f"❌ 训练模型失败: {model_key}, 错误: {e}")
                continue
        
        # 保存结果
        self.save_models_and_results()
        
        # 显示最终对比
        self.print_model_comparison()
    
    def save_models_and_results(self):
        """保存训练好的模型和结果"""
        print("\n💾 保存模型和结果...")
        
        # 保存每个模型
        for model_key, model in self.models.items():
            model_file = self.output_dir / f'{model_key}_model.pth'
            torch.save(model.state_dict(), model_file)
            print(f"✅ 保存模型: {model_file}")
            
            # 保存tokenizer
            if model_key in self.tokenizers:
                tokenizer_dir = self.output_dir / f'{model_key}_tokenizer'
                tokenizer_dir.mkdir(exist_ok=True)
                self.tokenizers[model_key].save_pretrained(tokenizer_dir)
                print(f"✅ 保存tokenizer: {tokenizer_dir}")
        
        # 保存结果
        results_file = self.output_dir / 'training_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✅ 保存训练结果: {results_file}")
    
    def print_model_comparison(self):
        """打印模型比较结果"""
        print(f"\n📊 {'='*80}")
        print("预训练模型性能对比")
        print(f"{'='*80}")
        
        if not self.results:
            print("⚠️  没有训练结果可显示")
            return
        
        print(f"{'模型':<20} {'描述':<25} {'验证准确率':<10} {'测试准确率':<10} {'测试F1':<8}")
        print("-" * 80)
        
        # 按测试F1分数排序
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1].get('test_f1_score', 0), 
                              reverse=True)
        
        for model_key, result in sorted_results:
            description = result.get('model_description', model_key)[:24]
            val_acc = result.get('best_val_accuracy', 0)
            test_acc = result.get('test_accuracy', 0)
            test_f1 = result.get('test_f1_score', 0)
            
            print(f"{model_key:<20} {description:<25} "
                  f"{val_acc:<10.2f} {test_acc:<10.2f} {test_f1:<8.4f}")
        
        if sorted_results:
            best_model = sorted_results[0]
            print(f"\n🏆 最佳模型: {best_model[0]}")
            print(f"   测试F1分数: {best_model[1].get('test_f1_score', 0):.4f}")
            print(f"   测试准确率: {best_model[1].get('test_accuracy', 0):.2f}%")


def main():
    """主函数，演示预训练模型训练流程"""
    print("🚀 预训练模型训练演示")
    
    # 创建训练器
    trainer = PretrainedModelTrainer(data_dir="data")
    
    # 选择要训练的模型（可根据需要调整）
    models_to_train = [
        'bert-base-uncased',
        'roberta-base',
        # 'albert-base-v2',  # 可选择性启用
        # 'chinese-bert-wwm'  # 中文模型，可选择性启用
    ]
    
    # 训练模型
    trainer.train_all_models(
        model_keys=models_to_train,
        epochs=3,           # 较少的训练轮数
        batch_size=8,       # 较小的批次大小，适应显存限制
        learning_rate=2e-5, # 标准的预训练模型学习率
        max_length=256      # 较短的序列长度，加快训练
    )
    
    print("\n✅ 预训练模型训练演示完成!")
    print(f"📁 模型和结果已保存到: {trainer.output_dir}")


if __name__ == "__main__":
    main()