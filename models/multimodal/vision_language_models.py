#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/multimodal/vision_language_models.py

"""
多模态视觉-语言模型实现 - 修复图像路径问题
解决图像加载失败的问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 快速路径设置
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入项目模块
try:
    from data_utils.data_loaders import create_all_dataloaders
    from utils.config_manager import get_config_manager, get_output_path
    from preprocessing.text_processing import TextProcessor
    from preprocessing.image_processing import ImageProcessor
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块")
except ImportError as e:
    print(f"⚠️  导入项目模块失败: {e}")
    USE_PROJECT_MODULES = False

# 尝试导入CLIP
try:
    import clip
    HAS_CLIP = True
    print("✅ CLIP可用")
except ImportError:
    print("⚠️  CLIP未安装，将使用简化实现")
    HAS_CLIP = False

# 尝试导入transformers用于BLIP
try:
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    HAS_BLIP = True
    print("✅ BLIP可用")
except ImportError:
    print("⚠️  BLIP不可用，将使用CLIP或简化实现")
    HAS_BLIP = False

import logging
logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """多模态数据集类 - 修复图像路径处理"""
    
    def __init__(self, texts: List[str], image_paths: List[str], labels: List[int], 
                 data_dir: str, text_processor, image_processor, max_text_length: int = 77):
        """
        初始化多模态数据集
        
        Args:
            texts: 文本列表
            image_paths: 图像路径列表
            labels: 标签列表
            data_dir: 数据根目录路径
            text_processor: 文本处理器
            image_processor: 图像处理器
            max_text_length: 最大文本长度
        """
        self.texts = texts
        self.image_paths = image_paths
        self.labels = labels
        self.data_dir = Path(data_dir)  # 保存数据根目录
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.max_text_length = max_text_length
        
        # 确保数据长度一致
        min_length = min(len(texts), len(image_paths), len(labels))
        self.texts = texts[:min_length]
        self.image_paths = image_paths[:min_length]
        self.labels = labels[:min_length]
        
        # 验证并修复图像路径
        self._fix_image_paths()
        
        print(f"📊 多模态数据集初始化: {len(self.texts)} 样本")
    
    def _fix_image_paths(self):
        """修复和验证图像路径"""
        print(f"🔧 修复图像路径...")
        
        fixed_paths = []
        valid_count = 0
        
        for i, img_path in enumerate(self.image_paths):
            if not img_path or img_path == "":
                # 空路径，保持为空
                fixed_paths.append("")
                continue
                
            # 构建可能的图像路径
            # 根据你的实际路径结构: data/split/img/index.jpg
            possible_paths = [
                self.data_dir / img_path,  # 直接相对于数据目录
                Path(img_path),  # 原始路径（如果是绝对路径）
            ]
            
            # 如果路径中不包含目录分隔符，可能只是文件名
            if "/" not in str(img_path) and "\\" not in str(img_path):
                # 尝试在各个split的img目录中查找
                for split in ['train', 'val', 'test']:
                    possible_paths.append(self.data_dir / split / 'img' / img_path)
            
            # 查找存在的路径
            found_path = None
            for test_path in possible_paths:
                if test_path.exists() and test_path.is_file():
                    # 将绝对路径转换为相对于data_dir的路径
                    try:
                        relative_path = test_path.relative_to(self.data_dir)
                        found_path = str(relative_path)
                        break
                    except ValueError:
                        # 如果无法转换为相对路径，使用绝对路径
                        found_path = str(test_path)
                        break
            
            if found_path:
                fixed_paths.append(found_path)
                valid_count += 1
            else:
                # 如果找不到，记录调试信息
                if i < 5:  # 只打印前5个失败案例
                    print(f"⚠️  找不到图像文件 {i}: {img_path}")
                    print(f"   尝试的路径:")
                    for j, p in enumerate(possible_paths[:3]):  # 只显示前3个尝试
                        print(f"     {p} - 存在: {p.exists()}")
                fixed_paths.append("")
        
        self.image_paths = fixed_paths
        print(f"✅ 图像路径修复完成: {valid_count}/{len(self.image_paths)} 个有效图像")
        
        # 显示前几个有效路径作为示例
        valid_examples = [path for path in fixed_paths[:10] if path]
        if valid_examples:
            print(f"   图像路径示例: {valid_examples[:3]}")
        
        return valid_count > 0
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 处理文本
        if hasattr(self.text_processor, 'tokenize') and callable(self.text_processor.tokenize):
            # 使用CLIP tokenizer
            try:
                text_tokens = self.text_processor.tokenize(text)
            except:
                # CLIP tokenizer失败，使用简单处理
                words = text.split()[:self.max_text_length]
                text_tokens = torch.zeros(self.max_text_length, dtype=torch.long)
                for i, word in enumerate(words):
                    text_tokens[i] = hash(word) % 30000
        else:
            # 简单文本处理
            words = text.split()[:self.max_text_length]
            text_tokens = torch.zeros(self.max_text_length, dtype=torch.long)
            for i, word in enumerate(words):
                text_tokens[i] = hash(word) % 30000  # 简单hash编码
        
        # 处理图像 - 修复图像加载逻辑
        try:
            if image_path and Path(image_path).exists():
                # 使用项目的图像处理器
                if hasattr(self.image_processor, 'process_single_image'):
                    image_tensor = self.image_processor.process_single_image(
                        image_path, transform_type='val'
                    )
                    if image_tensor is None:
                        raise ValueError("图像处理器返回None")
                else:
                    # 使用PIL直接处理
                    image = Image.open(image_path).convert('RGB')
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    image_tensor = transform(image)
            else:
                # 没有图像文件或文件不存在，创建空tensor
                image_tensor = torch.zeros(3, 224, 224)
                
        except Exception as e:
            # 处理任何图像加载错误，创建空tensor
            image_tensor = torch.zeros(3, 224, 224)
        
        return {
            'text': text_tokens,
            'image': image_tensor,
            'labels': torch.tensor(label, dtype=torch.long),
            'text_raw': text,
            'image_path': str(image_path) if image_path else ""
        }


class SimpleCLIPModel(nn.Module):
    """简化的CLIP风格模型"""
    
    def __init__(self, vocab_size: int = 30000, text_embed_dim: int = 512, 
                 image_embed_dim: int = 512, projection_dim: int = 256, 
                 num_classes: int = 3):
        """
        初始化简化CLIP模型
        
        Args:
            vocab_size: 词汇表大小
            text_embed_dim: 文本嵌入维度
            image_embed_dim: 图像嵌入维度
            projection_dim: 投影维度
            num_classes: 分类数量
        """
        super(SimpleCLIPModel, self).__init__()
        
        # 文本编码器
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(text_embed_dim, 8, dim_feedforward=2048),
            num_layers=6
        )
        self.text_projection = nn.Linear(text_embed_dim, projection_dim)
        
        # 图像编码器（简化的CNN）
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, image_embed_dim),
            nn.ReLU()
        )
        self.image_projection = nn.Linear(image_embed_dim, projection_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # 温度参数（用于对比学习）
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def encode_text(self, text):
        """编码文本"""
        # 文本嵌入
        x = self.text_embedding(text)  # [batch, seq_len, embed_dim]
        x = x.transpose(0, 1)  # [seq_len, batch, embed_dim]
        
        # Transformer编码
        x = self.text_transformer(x)
        
        # 全局平均池化
        x = x.mean(dim=0)  # [batch, embed_dim]
        
        # 投影
        x = self.text_projection(x)
        
        return F.normalize(x, dim=-1)
    
    def encode_image(self, image):
        """编码图像"""
        x = self.image_encoder(image)
        x = self.image_projection(x)
        return F.normalize(x, dim=-1)
    
    def forward(self, text, image):
        """前向传播"""
        text_features = self.encode_text(text)
        image_features = self.encode_image(image)
        
        # 融合特征
        fused_features = torch.cat([text_features, image_features], dim=1)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits, text_features, image_features


class CLIPBasedClassifier(nn.Module):
    """基于官方CLIP的分类器"""
    
    def __init__(self, clip_model_name: str = "ViT-B/32", num_classes: int = 3):
        """
        初始化CLIP分类器
        
        Args:
            clip_model_name: CLIP模型名称
            num_classes: 分类数量
        """
        super(CLIPBasedClassifier, self).__init__()
        
        if HAS_CLIP:
            # 加载预训练CLIP模型
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
            
            # 冻结CLIP参数
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            # 获取特征维度
            with torch.no_grad():
                dummy_text = clip.tokenize(["hello world"])
                dummy_image = torch.randn(1, 3, 224, 224)
                text_features = self.clip_model.encode_text(dummy_text).float()
                image_features = self.clip_model.encode_image(dummy_image).float()
                feature_dim = text_features.shape[-1] + image_features.shape[-1]
            
            # 分类头
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
            
            self.use_clip = True
            print("✅ 使用官方CLIP模型")
        else:
            # 使用简化实现
            self.clip_model = SimpleCLIPModel(num_classes=num_classes)
            self.use_clip = False
            print("⚠️  使用简化CLIP实现")
    
    def forward(self, text, image):
        """前向传播"""
        if self.use_clip and HAS_CLIP:
            # 使用官方CLIP
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text).float()
                image_features = self.clip_model.encode_image(image).float()
            
            # 融合特征
            fused_features = torch.cat([text_features, image_features], dim=1)
            logits = self.classifier(fused_features)
            
            return logits
        else:
            # 使用简化实现
            return self.clip_model(text, image)[0]


class MultiModalTrainer:
    """多模态模型训练器 - 修复数据加载"""
    
    def __init__(self, data_dir: str = "data", device: str = "auto"):
        """
        初始化训练器
        
        Args:
            data_dir: 数据目录路径
            device: 计算设备
        """
        self.data_dir = Path(data_dir).resolve()  # 转换为绝对路径
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  使用设备: {self.device}")
        
        # 初始化组件
        self.models = {}
        self.results = {}
        
        # 初始化处理器
        if USE_PROJECT_MODULES:
            self.text_processor = TextProcessor(language='mixed')
            self.image_processor = ImageProcessor(target_size=(224, 224))
        else:
            self.text_processor = None
            self.image_processor = None
        
        # 设置输出目录
        if USE_PROJECT_MODULES:
            config_manager = get_config_manager()
            self.output_dir = get_output_path('models', 'multimodal')
        else:
            self.output_dir = Path('outputs/models/multimodal')
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 标签映射
        self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
        
        print(f"🎭 多模态模型训练器初始化完成")
        print(f"   数据目录: {self.data_dir}")
        print(f"   输出目录: {self.output_dir}")
    
    def load_data(self) -> Dict[str, Tuple[List[str], List[str], List[int]]]:
        """加载MR2多模态数据集 - 改进版本，优先使用直接加载"""
        print("📚 加载MR2多模态数据集...")
        
        # 优先尝试直接从JSON文件加载，更可靠
        try:
            data = self._load_data_directly()
            if data and len(data) > 0:
                print("✅ 使用直接文件加载成功")
                return data
        except Exception as e:
            print(f"⚠️  直接文件加载失败: {e}")
        
        # 备用：尝试使用数据加载器
        if USE_PROJECT_MODULES:
            try:
                print("🔄 尝试使用数据加载器...")
                dataloaders = create_all_dataloaders(
                    batch_sizes={'train': 16, 'val': 16, 'test': 16}
                )
                
                data = {}
                for split, dataloader in dataloaders.items():
                    texts = []
                    image_paths = []
                    labels = []
                    
                    print(f"📂 处理 {split} 数据集...")
                    
                    for batch_idx, batch in enumerate(dataloader):
                        # 提取文本
                        if 'text' in batch:
                            batch_texts = batch['text']
                        elif 'caption' in batch:
                            batch_texts = batch['caption']
                        else:
                            print(f"⚠️  批次 {batch_idx} 没有文本字段")
                            continue
                        
                        texts.extend(batch_texts)
                        
                        # 提取标签
                        if 'labels' in batch:
                            batch_labels = batch['labels'].tolist()
                        elif 'label' in batch:
                            batch_labels = batch['label']
                        else:
                            print(f"⚠️  批次 {batch_idx} 没有标签字段")
                            batch_labels = [0] * len(batch_texts)
                        
                        labels.extend(batch_labels)
                        
                        # 修复：检查数据集是否有图像路径信息
                        batch_size = len(batch_texts)
                        batch_image_paths = []
                        
                        # 尝试从batch中获取图像路径
                        if 'image_path' in batch:
                            batch_image_paths = batch['image_path']
                        else:
                            # 如果batch中没有image_path，尝试构建路径
                            # 根据数据目录结构：data/split/img/index.jpg
                            start_idx = batch_idx * dataloader.batch_size
                            for i in range(batch_size):
                                img_path = f"{split}/img/{start_idx + i}.jpg"
                                batch_image_paths.append(img_path)
                        
                        image_paths.extend(batch_image_paths)
                    
                    # 确保三个列表长度一致
                    min_length = min(len(texts), len(image_paths), len(labels))
                    if min_length < len(texts):
                        print(f"⚠️  {split} 数据长度不一致，截断到 {min_length}")
                        texts = texts[:min_length]
                        image_paths = image_paths[:min_length]
                        labels = labels[:min_length]
                    
                    data[split] = (texts, image_paths, labels)
                    print(f"✅ 加载 {split}: {len(texts)} 样本")
                
                return data
                
            except Exception as e:
                print(f"❌ 使用项目数据加载器失败: {e}")
        
        # 最后备用：创建演示数据
        print("🔄 使用演示数据...")
        return self._create_demo_data()

    def _load_data_directly(self) -> Dict[str, Tuple[List[str], List[str], List[int]]]:
        """直接从JSON文件加载数据 - 修复版本"""
        print("📂 直接从JSON文件加载数据...")
        data = {}
        
        for split in ['train', 'val', 'test']:
            dataset_file = self.data_dir / f'dataset_items_{split}.json'
            
            if not dataset_file.exists():
                print(f"⚠️  数据文件不存在: {dataset_file}")
                continue
            
            try:
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    dataset_items = json.load(f)
                
                texts = []
                image_paths = []
                labels = []
                
                for item_id, item_data in dataset_items.items():
                    # 修复：优先检查caption字段，然后是text字段
                    text = item_data.get('caption', '') or item_data.get('text', '')
                    if not text or not text.strip():
                        print(f"⚠️  跳过样本 {item_id}: 没有有效文本内容")
                        continue
                    
                    texts.append(text.strip())
                    
                    # 提取标签
                    label = item_data.get('label', 0)
                    labels.append(label)
                    
                    # 使用数据中提供的图像路径
                    img_path = item_data.get('image_path', '')
                    if not img_path:
                        # 如果没有image_path，尝试构建路径
                        img_path = f"{split}/img/{item_id}.jpg"
                    
                    image_paths.append(img_path)
                
                data[split] = (texts, image_paths, labels)
                print(f"✅ 直接加载 {split}: {len(texts)} 样本")
                
                # 验证前几个图像路径是否存在
                if image_paths:
                    valid_images = 0
                    check_count = min(len(image_paths), 5)
                    for i, img_path in enumerate(image_paths[:check_count]):
                        full_path = self.data_dir / img_path
                        if full_path.exists():
                            valid_images += 1
                        elif i < 3:  # 只打印前3个失败的
                            print(f"   图像路径示例 {i}: {img_path} -> {full_path} (存在: {full_path.exists()})")
                    
                    print(f"   前{check_count}个图像中有效的: {valid_images}/{check_count}")
                
            except Exception as e:
                print(f"❌ 直接加载 {split} 失败: {e}")
                continue
        
        return data

    def _create_demo_data(self) -> Dict[str, Tuple[List[str], List[str], List[int]]]:
        """创建演示数据"""
        print("🔧 创建多模态演示数据...")
        print("⚠️  注意：由于没有真实数据，将使用空图像路径")
        
        demo_texts = [
            "这是一个关于科技进步的真实新闻报道",
            "This is fake news about celebrity scandal",
            "未经证实的传言需要进一步调查验证",
            "Breaking: Major breakthrough in AI technology",
            "网传某地发生重大事故，官方尚未确认",
            "Scientists discover new species in deep ocean",
            "谣传某公司倒闭，实际情况有待核实",
            "Weather alert: Severe storm approaching",
            "社交媒体流传的未证实消息引发关注",
            "Economic indicators show positive trends"
        ]
        
        demo_labels = [0, 1, 2, 0, 2, 0, 1, 0, 2, 0]
        demo_image_paths = [""] * len(demo_texts)
        
        # 扩展数据
        extended_texts = demo_texts * 8
        extended_image_paths = demo_image_paths * 8
        extended_labels = demo_labels * 8
        
        # 按比例分割数据
        total_size = len(extended_texts)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        
        print(f"📝 创建演示数据: {total_size} 个样本（无图像）")
        
        return {
            'train': (extended_texts[:train_size], 
                     extended_image_paths[:train_size],
                     extended_labels[:train_size]),
            'val': (extended_texts[train_size:train_size+val_size], 
                   extended_image_paths[train_size:train_size+val_size],
                   extended_labels[train_size:train_size+val_size]),
            'test': (extended_texts[train_size+val_size:], 
                    extended_image_paths[train_size+val_size:],
                    extended_labels[train_size+val_size:])
        }

    def create_models(self):
        """创建多模态模型"""
        print("🎭 创建多模态模型...")
        
        # CLIP模型
        self.models['clip'] = CLIPBasedClassifier(
            clip_model_name="ViT-B/32",
            num_classes=3
        ).to(self.device)
        
        # 简化多模态模型
        self.models['simple_multimodal'] = SimpleCLIPModel(
            vocab_size=30000,
            text_embed_dim=256,
            image_embed_dim=256,
            projection_dim=128,
            num_classes=3
        ).to(self.device)
        
        print(f"✅ 创建了 {len(self.models)} 个多模态模型")
        
        # 打印模型参数量
        for name, model in self.models.items():
            param_count = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   {name}: {param_count:,} 参数 ({trainable_params:,} 可训练)")
    
    def train_single_model(self, model_name: str, train_loader: DataLoader, 
                          val_loader: DataLoader, epochs: int = 5, 
                          learning_rate: float = 1e-4) -> Dict[str, Any]:
        """训练单个模型"""
        print(f"🏋️ 训练 {model_name} 模型...")
        
        model = self.models[model_name]
        
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
                text = batch['text'].to(self.device)
                image = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # 根据模型类型调用不同的前向传播
                if isinstance(model, CLIPBasedClassifier):
                    logits = model(text, image)
                else:
                    logits = model(text, image)[0]  # SimpleCLIPModel返回多个值
                
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
                    text = batch['text'].to(self.device)
                    image = batch['image'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    if isinstance(model, CLIPBasedClassifier):
                        logits = model(text, image)
                    else:
                        logits = model(text, image)[0]
                    
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
                text = batch['text'].to(self.device)
                image = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if isinstance(model, CLIPBasedClassifier):
                    logits = model(text, image)
                else:
                    logits = model(text, image)[0]
                
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
    
    def train_all_models(self, epochs: int = 5, batch_size: int = 8, learning_rate: float = 1e-4):
        """训练所有多模态模型"""
        print("🚀 开始训练所有多模态模型...")
        
        # 加载数据
        data = self.load_data()
        
        # 创建模型
        self.create_models()
        
        # 处理文本的tokenizer
        if HAS_CLIP:
            text_processor = clip.tokenize
        else:
            text_processor = self.text_processor
        
        # 创建数据集
        train_dataset = MultiModalDataset(
            data['train'][0], data['train'][1], data['train'][2],
            str(self.data_dir), text_processor, self.image_processor
        )
        val_dataset = MultiModalDataset(
            data['val'][0], data['val'][1], data['val'][2],
            str(self.data_dir), text_processor, self.image_processor
        )
        test_dataset = MultiModalDataset(
            data['test'][0], data['test'][1], data['test'][2],
            str(self.data_dir), text_processor, self.image_processor
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练每个模型
        for model_name in self.models.keys():
            print(f"\n{'='*60}")
            print(f"训练模型: {model_name.upper()}")
            print(f"{'='*60}")
            
            try:
                # 训练模型
                train_result = self.train_single_model(
                    model_name, train_loader, val_loader, epochs, learning_rate
                )
                
                # 测试评估
                test_result = self.evaluate_model(model_name, test_loader)
                
                # 合并结果
                self.results[model_name] = {**train_result, **test_result}
                
            except Exception as e:
                print(f"❌ 训练模型失败: {model_name}, 错误: {e}")
                continue
        
        # 保存模型和结果
        self.save_models_and_results()
        
        # 显示最终对比
        self.print_model_comparison()
    
    def save_models_and_results(self):
        """保存训练好的模型和结果"""
        print("\n💾 保存模型和结果...")
        
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
    
    def print_model_comparison(self):
        """打印模型比较结果"""
        print(f"\n📊 {'='*70}")
        print("多模态模型性能对比")
        print(f"{'='*70}")
        
        if not self.results:
            print("⚠️  没有训练结果可显示")
            return
        
        print(f"{'模型':<20} {'验证准确率':<10} {'测试准确率':<10} {'验证F1':<8} {'测试F1':<8}")
        print("-" * 70)
        
        # 按测试F1分数排序
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1].get('test_f1_score', 0), 
                              reverse=True)
        
        for model_name, result in sorted_results:
            val_acc = result.get('best_val_accuracy', 0)
            test_acc = result.get('test_accuracy', 0)
            val_f1 = result.get('val_f1_score', 0)
            test_f1 = result.get('test_f1_score', 0)
            
            print(f"{model_name:<20} "
                  f"{val_acc:<10.2f} "
                  f"{test_acc:<10.2f} "
                  f"{val_f1:<8.4f} "
                  f"{test_f1:<8.4f}")
        
        if sorted_results:
            best_model = sorted_results[0]
            print(f"\n🏆 最佳模型: {best_model[0]}")
            print(f"   测试F1分数: {best_model[1].get('test_f1_score', 0):.4f}")
            print(f"   测试准确率: {best_model[1].get('test_accuracy', 0):.2f}%")


def main():
    """主函数，演示多模态模型训练流程"""
    print("🚀 多模态模型训练演示")
    
    # 创建训练器
    trainer = MultiModalTrainer(data_dir="data")
    
    # 训练所有模型
    trainer.train_all_models(
        epochs=5,           # 训练轮数
        batch_size=8,       # 小批次，适应显存限制
        learning_rate=1e-4  # 多模态模型学习率
    )
    
    print("\n✅ 多模态模型训练演示完成!")
    print(f"📁 模型和结果已保存到: {trainer.output_dir}")


if __name__ == "__main__":
    main()