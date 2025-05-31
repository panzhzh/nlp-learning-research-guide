#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# datasets/mr2_dataset.py

"""
MR2多模态谣言检测数据集类
PyTorch Dataset实现，支持：
- 文本和图像的多模态数据加载
- 检索增强信息的整合
- 灵活的数据增强策略
- 批量数据处理
- 缓存机制优化
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_file = Path(__file__).resolve()
code_root = current_file.parent.parent
sys.path.append(str(code_root))

# 导入自定义模块
try:
    from preprocessing.text_processing import TextProcessor
    from preprocessing.image_processing import ImageProcessor
    from utils.config_manager import get_data_config, get_data_dir, get_label_mapping
    USE_CUSTOM_MODULES = True
except ImportError as e:
    print(f"⚠️  导入自定义模块失败: {e}")
    USE_CUSTOM_MODULES = False

import logging
logger = logging.getLogger(__name__)


class MR2Dataset(Dataset):
    """
    MR2多模态谣言检测数据集
    
    数据格式:
    - 文本: 中英文混合的声明文本
    - 图像: 与声明相关的图像
    - 标签: 0=Non-rumor, 1=Rumor, 2=Unverified
    - 检索信息: 直接检索和反向检索的验证信息
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 split: str = 'train',
                 transform_type: str = 'train',
                 load_retrieval_info: bool = True,
                 use_cache: bool = True,
                 target_size: Tuple[int, int] = (224, 224),
                 max_text_length: int = 512):
        """
        初始化MR2数据集
        
        Args:
            data_dir: 数据目录路径
            split: 数据划分 ('train', 'val', 'test')
            transform_type: 图像变换类型 ('train', 'val')
            load_retrieval_info: 是否加载检索信息
            use_cache: 是否使用缓存
            target_size: 目标图像尺寸
            max_text_length: 最大文本长度
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform_type = transform_type
        self.load_retrieval_info = load_retrieval_info
        self.use_cache = use_cache
        self.target_size = target_size
        self.max_text_length = max_text_length
        
        # 初始化处理器
        self.setup_processors()
        
        # 加载配置
        self.load_config()
        
        # 加载数据
        self.load_dataset()
        
        # 初始化缓存
        if self.use_cache:
            self.image_cache = {}
            self.text_cache = {}
        
        print(f"📚 MR2数据集初始化完成")
        print(f"   数据划分: {self.split}")
        print(f"   样本数量: {len(self.items)}")
        print(f"   标签分布: {self.get_label_distribution()}")
    
    def setup_processors(self):
        """设置文本和图像处理器"""
        if USE_CUSTOM_MODULES:
            # 使用自定义处理器
            self.text_processor = TextProcessor(language='mixed')
            self.image_processor = ImageProcessor(target_size=self.target_size)
        else:
            # 使用基本处理器
            self.text_processor = None
            self.image_processor = None
            
            # 设置基本图像变换
            if self.transform_type == 'train':
                self.image_transforms = transforms.Compose([
                    transforms.Resize(self.target_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.image_transforms = transforms.Compose([
                    transforms.Resize(self.target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
    
    def load_config(self):
        """加载配置信息"""
        if USE_CUSTOM_MODULES:
            try:
                # 使用配置管理器
                self.label_mapping = get_label_mapping()
                data_config = get_data_config()
                self.dataset_config = data_config.get('dataset', {})
            except:
                # 默认配置
                self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
                self.dataset_config = {}
        else:
            # 默认配置
            self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
            self.dataset_config = {}
    
    def load_dataset(self):
        """加载数据集文件"""
        # 加载主数据文件
        dataset_file = self.data_dir / f'dataset_items_{self.split}.json'
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"数据集文件不存在: {dataset_file}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # 构建数据项列表
        self.items = []
        self.item_ids = []
        
        for item_id, item_data in self.raw_data.items():
            # 验证必要字段
            if 'caption' in item_data and 'label' in item_data:
                self.items.append(item_data)
                self.item_ids.append(item_id)
        
        print(f"📂 加载 {self.split} 数据: {len(self.items)} 个有效样本")
        
        # 加载检索信息
        if self.load_retrieval_info:
            self.load_retrieval_annotations()
    
    def load_retrieval_annotations(self):
        """加载检索标注信息"""
        self.direct_annotations = {}
        self.inverse_annotations = {}
        
        # 加载直接检索标注
        direct_file = self.data_dir / self.split / 'img_html_news' / 'direct_annotation.json'
        if direct_file.exists():
            try:
                with open(direct_file, 'r', encoding='utf-8') as f:
                    self.direct_annotations = json.load(f)
                print(f"✅ 加载直接检索标注: {len(self.direct_annotations)} 条")
            except Exception as e:
                logger.warning(f"加载直接检索标注失败: {e}")
        
        # 加载反向检索标注
        inverse_file = self.data_dir / self.split / 'inverse_search' / 'inverse_annotation.json'
        if inverse_file.exists():
            try:
                with open(inverse_file, 'r', encoding='utf-8') as f:
                    self.inverse_annotations = json.load(f)
                print(f"✅ 加载反向检索标注: {len(self.inverse_annotations)} 条")
            except Exception as e:
                logger.warning(f"加载反向检索标注失败: {e}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含文本、图像、标签等信息的字典
        """
        if idx >= len(self.items):
            raise IndexError(f"索引超出范围: {idx} >= {len(self.items)}")
        
        item = self.items[idx]
        item_id = self.item_ids[idx]
        
        # 构建基本数据项
        data_item = {
            'item_id': item_id,
            'text': item.get('caption', ''),
            'label': item.get('label', -1),
            'language': item.get('language', 'unknown'),
            'source': item.get('source', 'unknown')
        }
        
        # 处理文本
        processed_text = self.process_text(data_item['text'])
        data_item.update(processed_text)
        
        # 处理图像
        if 'image_path' in item:
            processed_image = self.process_image(item['image_path'], item_id)
            data_item.update(processed_image)
        else:
            # 如果没有图像，创建空tensor
            data_item['image'] = torch.zeros(3, *self.target_size)
            data_item['has_image'] = False
        
        # 添加检索信息
        if self.load_retrieval_info:
            retrieval_info = self.get_retrieval_info(item_id)
            data_item.update(retrieval_info)
        
        return data_item
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        处理文本数据
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本信息字典
        """
        if USE_CUSTOM_MODULES and self.text_processor:
            # 使用缓存
            if self.use_cache and text in self.text_cache:
                return self.text_cache[text]
            
            # 处理文本
            cleaned_text = self.text_processor.clean_text(text)
            tokens = self.text_processor.tokenize(text)
            features = self.text_processor.extract_features(text)
            
            processed = {
                'text_cleaned': cleaned_text,
                'text_tokens': tokens,
                'text_length': len(text),
                'token_count': len(tokens),
                'detected_language': features.get('language', 'unknown'),
                'text_features': features
            }
            
            # 缓存结果
            if self.use_cache:
                self.text_cache[text] = processed
            
            return processed
        else:
            # 基本文本处理
            return {
                'text_cleaned': text.strip(),
                'text_tokens': text.split(),
                'text_length': len(text),
                'token_count': len(text.split()),
                'detected_language': 'unknown',
                'text_features': {}
            }
    
    def process_image(self, image_path: str, item_id: str) -> Dict[str, Any]:
        """
        处理图像数据
        
        Args:
            image_path: 图像相对路径
            item_id: 数据项ID
            
        Returns:
            处理后的图像信息字典
        """
        # 构建完整路径
        full_image_path = self.data_dir / image_path
        
        # 使用缓存
        if self.use_cache and item_id in self.image_cache:
            return self.image_cache[item_id]
        
        if not full_image_path.exists():
            logger.warning(f"图像文件不存在: {full_image_path}")
            return {
                'image': torch.zeros(3, *self.target_size),
                'has_image': False,
                'image_path': str(full_image_path)
            }
        
        try:
            if USE_CUSTOM_MODULES and self.image_processor:
                # 使用自定义图像处理器
                image_tensor = self.image_processor.process_single_image(
                    full_image_path, 
                    transform_type=self.transform_type
                )
                
                if image_tensor is None:
                    raise Exception("图像处理失败")
                
                # 获取图像信息
                image_info = self.image_processor.get_image_info(full_image_path)
                
                processed = {
                    'image': image_tensor,
                    'has_image': True,
                    'image_path': str(full_image_path),
                    'image_info': image_info
                }
            else:
                # 基本图像处理
                image = Image.open(full_image_path).convert('RGB')
                image_tensor = self.image_transforms(image)
                
                processed = {
                    'image': image_tensor,
                    'has_image': True,
                    'image_path': str(full_image_path),
                    'image_info': {
                        'size': image.size,
                        'mode': image.mode
                    }
                }
            
            # 缓存结果
            if self.use_cache:
                self.image_cache[item_id] = processed
            
            return processed
            
        except Exception as e:
            logger.error(f"处理图像失败 {full_image_path}: {e}")
            return {
                'image': torch.zeros(3, *self.target_size),
                'has_image': False,
                'image_path': str(full_image_path)
            }
    
    def get_retrieval_info(self, item_id: str) -> Dict[str, Any]:
        """
        获取检索信息
        
        Args:
            item_id: 数据项ID
            
        Returns:
            检索信息字典
        """
        retrieval_info = {
            'has_direct_retrieval': False,
            'has_inverse_retrieval': False,
            'direct_info': {},
            'inverse_info': {}
        }
        
        # 直接检索信息
        if item_id in self.direct_annotations:
            retrieval_info['has_direct_retrieval'] = True
            retrieval_info['direct_info'] = self.direct_annotations[item_id]
        
        # 反向检索信息
        if item_id in self.inverse_annotations:
            retrieval_info['has_inverse_retrieval'] = True
            retrieval_info['inverse_info'] = self.inverse_annotations[item_id]
        
        return retrieval_info
    
    def get_label_distribution(self) -> Dict[str, int]:
        """获取标签分布"""
        label_counts = {}
        for item in self.items:
            label = item.get('label', -1)
            label_name = self.label_mapping.get(label, f'Unknown({label})')
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        return label_counts
    
    def get_sample_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取样本
        
        Args:
            item_id: 数据项ID
            
        Returns:
            样本数据或None
        """
        try:
            idx = self.item_ids.index(item_id)
            return self[idx]
        except ValueError:
            return None
    
    def get_samples_by_label(self, label: int, max_samples: int = 10) -> List[Dict[str, Any]]:
        """
        获取指定标签的样本
        
        Args:
            label: 标签值
            max_samples: 最大样本数
            
        Returns:
            样本列表
        """
        samples = []
        for idx, item in enumerate(self.items):
            if item.get('label') == label and len(samples) < max_samples:
                samples.append(self[idx])
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.items),
            'label_distribution': self.get_label_distribution(),
            'language_distribution': {},
            'has_image_count': 0,
            'has_retrieval_count': 0
        }
        
        # 统计语言分布和其他信息
        for item in self.items:
            # 语言分布
            language = item.get('language', 'unknown')
            stats['language_distribution'][language] = stats['language_distribution'].get(language, 0) + 1
            
            # 图像统计
            if 'image_path' in item:
                stats['has_image_count'] += 1
        
        # 检索信息统计
        if self.load_retrieval_info:
            stats['direct_retrieval_count'] = len(self.direct_annotations)
            stats['inverse_retrieval_count'] = len(self.inverse_annotations)
        
        return stats
    
    def save_processed_data(self, output_dir: Optional[str] = None):
        """
        保存预处理后的数据
        
        Args:
            output_dir: 输出目录，默认为data/processed
        """
        if output_dir is None:
            output_dir = self.data_dir / 'processed'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理所有数据
        processed_data = []
        print(f"🔄 处理 {len(self.items)} 个样本...")
        
        for idx in range(len(self.items)):
            if idx % 50 == 0:
                print(f"  已处理 {idx}/{len(self.items)}")
            
            data_item = self[idx]
            processed_data.append(data_item)
        
        # 保存处理后的数据
        output_file = output_dir / f'{self.split}_processed.json'
        
        # 由于包含tensor，需要特殊处理
        serializable_data = []
        for item in processed_data:
            # 创建可序列化的副本
            serializable_item = {}
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    # 保存tensor的形状信息
                    serializable_item[f'{key}_shape'] = list(value.shape)
                elif key == 'text_features' and isinstance(value, dict):
                    # 保存文本特征
                    serializable_item[key] = value
                elif not isinstance(value, (torch.Tensor, type(lambda: None))):
                    serializable_item[key] = value
            
            serializable_data.append(serializable_item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 处理后的数据已保存到: {output_file}")


def create_mr2_dataloader(data_dir: Union[str, Path],
                         split: str = 'train',
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         **dataset_kwargs) -> DataLoader:
    """
    创建MR2数据加载器
    
    Args:
        data_dir: 数据目录
        split: 数据划分
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        **dataset_kwargs: 数据集额外参数
        
    Returns:
        DataLoader对象
    """
    # 创建数据集
    dataset = MR2Dataset(data_dir=data_dir, split=split, **dataset_kwargs)
    
    # 自定义collate function处理可变长度数据
    def collate_fn(batch):
        """自定义批处理函数"""
        # 收集所有字段
        batch_data = {}
        
        # 处理每个字段
        for key in batch[0].keys():
            if key == 'image':
                # 堆叠图像tensor
                batch_data[key] = torch.stack([item[key] for item in batch])
            elif key in ['text_tokens']:
                # 保持列表形式
                batch_data[key] = [item[key] for item in batch]
            elif isinstance(batch[0][key], (int, float, str, bool)):
                # 简单类型直接收集
                batch_data[key] = [item[key] for item in batch]
            elif isinstance(batch[0][key], dict):
                # 字典类型
                batch_data[key] = [item[key] for item in batch]
            else:
                # 其他类型
                batch_data[key] = [item[key] for item in batch]
        
        return batch_data
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


# 使用示例和测试代码
if __name__ == "__main__":
    print("📚 测试MR2数据集类")
    
    # 设置数据目录
    data_dir = "data"  # 根据实际情况修改
    
    try:
        # 创建数据集
        print(f"\n📂 加载训练数据集...")
        train_dataset = MR2Dataset(
            data_dir=data_dir,
            split='train',
            transform_type='train',
            load_retrieval_info=True,
            use_cache=True
        )
        
        print(f"✅ 数据集创建成功")
        print(f"   数据集大小: {len(train_dataset)}")
        
        # 获取统计信息
        stats = train_dataset.get_statistics()
        print(f"\n📊 数据集统计:")
        print(f"   总样本数: {stats['total_samples']}")
        print(f"   标签分布: {stats['label_distribution']}")
        print(f"   语言分布: {stats['language_distribution']}")
        print(f"   有图像样本: {stats['has_image_count']}")
        
        # 测试单个样本
        print(f"\n🔍 测试单个样本:")
        sample = train_dataset[0]
        print(f"   样本ID: {sample['item_id']}")
        print(f"   文本长度: {sample['text_length']}")
        print(f"   标签: {sample['label']}")
        print(f"   有图像: {sample['has_image']}")
        if sample['has_image']:
            print(f"   图像形状: {sample['image'].shape}")
        
        # 测试按标签获取样本
        print(f"\n🏷️  测试按标签获取样本:")
        rumor_samples = train_dataset.get_samples_by_label(label=1, max_samples=3)
        print(f"   获取到 {len(rumor_samples)} 个谣言样本")
        
        # 创建数据加载器
        print(f"\n🔄 创建数据加载器...")
        dataloader = create_mr2_dataloader(
            data_dir=data_dir,
            split='train',
            batch_size=4,
            shuffle=True,
            num_workers=0  # 设置为0避免多进程问题
        )
        
        print(f"✅ 数据加载器创建成功")
        
        # 测试批处理
        print(f"\n📦 测试批处理:")
        for batch_idx, batch in enumerate(dataloader):
            print(f"   批次 {batch_idx}:")
            print(f"     批次大小: {len(batch['item_id'])}")
            print(f"     图像形状: {batch['image'].shape}")
            print(f"     标签: {batch['label']}")
            
            if batch_idx >= 2:  # 只测试前3个批次
                break
        
        # 保存处理后的数据（可选）
        print(f"\n💾 保存处理后的数据...")
        train_dataset.save_processed_data()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print(f"请确保数据目录 '{data_dir}' 存在且包含必要的文件")
        
        # 提供调试信息
        data_path = Path(data_dir)
        if data_path.exists():
            print(f"\n🔍 数据目录内容:")
            for item in data_path.iterdir():
                print(f"   {item.name}")
        else:
            print(f"\n❌ 数据目录不存在: {data_path}")
    
    print(f"\n✅ MR2数据集测试完成")