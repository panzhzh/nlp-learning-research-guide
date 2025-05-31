#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# datasets/mr2_dataset.py

"""
简化的MR2数据集类
专注于核心功能，易于理解和调试
"""

import json
import torch
from torch.utils.data import Dataset
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

# 导入配置管理
try:
    from utils.config_manager import get_data_config, get_data_dir, get_label_mapping
    USE_CONFIG = True
except ImportError:
    print("⚠️  配置管理器不可用，使用默认配置")
    USE_CONFIG = False

import logging
logger = logging.getLogger(__name__)


class SimpleMR2Dataset(Dataset):
    """
    简化的MR2多模态谣言检测数据集
    
    功能:
    - 加载文本和图像数据
    - 基本的数据预处理
    - 错误处理和调试信息
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 split: str = 'train',
                 transform_type: str = 'train',
                 target_size: Tuple[int, int] = (224, 224),
                 load_images: bool = True):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            split: 数据划分 ('train', 'val', 'test')
            transform_type: 图像变换类型 ('train', 'val')
            target_size: 目标图像尺寸
            load_images: 是否加载图像
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform_type = transform_type
        self.target_size = target_size
        self.load_images = load_images
        
        # 加载配置
        self.setup_config()
        
        # 设置图像变换
        self.setup_transforms()
        
        # 加载数据
        self.load_dataset()
        
        print(f"📚 简化MR2数据集初始化完成")
        print(f"   数据划分: {self.split}")
        print(f"   样本数量: {len(self.items)}")
        print(f"   加载图像: {self.load_images}")
        if self.items:
            print(f"   标签分布: {self.get_label_distribution()}")
    
    def setup_config(self):
        """设置配置"""
        if USE_CONFIG:
            try:
                self.label_mapping = get_label_mapping()
                data_config = get_data_config()
                self.dataset_config = data_config.get('dataset', {})
            except:
                self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
                self.dataset_config = {}
        else:
            self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
            self.dataset_config = {}
    
    def setup_transforms(self):
        """设置图像变换"""
        # 训练时的变换
        if self.transform_type == 'train':
            self.image_transforms = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.3),  # 减少随机性
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # 验证/测试时的变换
            self.image_transforms = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def load_dataset(self):
        """加载数据集文件"""
        # 加载主数据文件
        dataset_file = self.data_dir / f'dataset_items_{self.split}.json'
        
        if not dataset_file.exists():
            print(f"❌ 数据集文件不存在: {dataset_file}")
            print(f"   请确保文件路径正确")
            print(f"   当前数据目录: {self.data_dir}")
            
            # 创建空数据集用于演示
            self.items = []
            self.item_ids = []
            return
        
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            print(f"✅ 成功加载数据文件: {dataset_file}")
        except Exception as e:
            print(f"❌ 加载数据文件失败: {e}")
            self.items = []
            self.item_ids = []
            return
        
        # 构建数据项列表
        self.items = []
        self.item_ids = []
        
        for item_id, item_data in self.raw_data.items():
            # 验证必要字段
            if 'caption' in item_data and 'label' in item_data:
                self.items.append(item_data)
                self.item_ids.append(item_id)
        
        print(f"📂 加载 {self.split} 数据: {len(self.items)} 个有效样本")
    
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
            'text_length': len(item.get('caption', '')),
            'token_count': len(item.get('caption', '').split())
        }
        
        # 处理图像
        if self.load_images and 'image_path' in item:
            image_result = self.load_image_safe(item['image_path'])
            data_item.update(image_result)
        else:
            # 如果不加载图像或没有图像路径，创建空tensor
            data_item.update({
                'image': torch.zeros(3, *self.target_size),
                'has_image': False,
                'image_path': None
            })
        
        return data_item
    
    def load_image_safe(self, image_path: str) -> Dict[str, Any]:
        """
        安全地加载图像
        
        Args:
            image_path: 图像相对路径
            
        Returns:
            图像数据字典
        """
        # 构建完整路径
        full_image_path = self.data_dir / image_path
        
        try:
            # 检查文件是否存在
            if not full_image_path.exists():
                logger.warning(f"图像文件不存在: {full_image_path}")
                return self.create_empty_image_result(str(full_image_path))
            
            # 尝试加载图像
            with Image.open(full_image_path) as image:
                # 转换为RGB模式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 应用变换
                image_tensor = self.image_transforms(image)
                
                return {
                    'image': image_tensor,
                    'has_image': True,
                    'image_path': str(full_image_path),
                    'image_size': image.size
                }
                
        except Exception as e:
            logger.error(f"处理图像失败 {full_image_path}: {e}")
            return self.create_empty_image_result(str(full_image_path))
    
    def create_empty_image_result(self, image_path: str) -> Dict[str, Any]:
        """创建空图像结果"""
        return {
            'image': torch.zeros(3, *self.target_size),
            'has_image': False,
            'image_path': image_path
        }
    
    def get_label_distribution(self) -> Dict[str, int]:
        """获取标签分布"""
        if not self.items:
            return {}
        
        label_counts = {}
        for item in self.items:
            label = item.get('label', -1)
            label_name = self.label_mapping.get(label, f'Unknown({label})')
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        return label_counts
    
    def get_sample_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取样本"""
        try:
            idx = self.item_ids.index(item_id)
            return self[idx]
        except ValueError:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self.items:
            return {'total_samples': 0}
        
        stats = {
            'total_samples': len(self.items),
            'label_distribution': self.get_label_distribution(),
            'has_image_count': 0,
            'text_length_stats': {}
        }
        
        # 统计图像和文本信息
        text_lengths = []
        for item in self.items:
            # 图像统计
            if 'image_path' in item:
                image_path = self.data_dir / item['image_path']
                if image_path.exists():
                    stats['has_image_count'] += 1
            
            # 文本长度统计
            text = item.get('caption', '')
            text_lengths.append(len(text))
        
        # 文本长度统计
        if text_lengths:
            stats['text_length_stats'] = {
                'min': min(text_lengths),
                'max': max(text_lengths),
                'mean': np.mean(text_lengths),
                'std': np.std(text_lengths)
            }
        
        return stats
    
    def print_sample_info(self, idx: int = 0):
        """打印样本信息用于调试"""
        if idx >= len(self.items):
            print(f"❌ 索引 {idx} 超出范围")
            return
        
        sample = self[idx]
        print(f"\n🔍 样本 {idx} 信息:")
        print(f"   ID: {sample['item_id']}")
        print(f"   文本: {sample['text'][:50]}...")
        print(f"   标签: {sample['label']} ({self.label_mapping.get(sample['label'], 'Unknown')})")
        print(f"   文本长度: {sample['text_length']}")
        print(f"   有图像: {sample['has_image']}")
        if sample['has_image']:
            print(f"   图像路径: {sample['image_path']}")
            print(f"   图像张量形状: {sample['image'].shape}")


# 兼容性别名
MR2Dataset = SimpleMR2Dataset


def create_demo_dataset(data_dir: str, split: str = 'train'):
    """
    创建演示数据集（当真实数据不可用时）
    
    Args:
        data_dir: 数据目录
        split: 数据划分
    """
    print(f"🔧 创建演示数据集: {split}")
    
    # 确保数据目录存在
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # 创建演示数据
    demo_data = {}
    texts = [
        "这是一个谣言检测的演示文本",
        "This is a demo text for rumor detection",
        "混合语言的演示文本 mixed language demo",
        "另一个中文演示样本",
        "Another English demo sample"
    ]
    
    for i in range(len(texts)):
        demo_data[str(i)] = {
            'caption': texts[i % len(texts)],
            'label': i % 3,  # 0, 1, 2 循环
            'language': 'mixed',
            'image_path': f'{split}/img/{i}.jpg'
        }
    
    # 保存到文件
    demo_file = data_path / f'dataset_items_{split}.json'
    with open(demo_file, 'w', encoding='utf-8') as f:
        json.dump(demo_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 演示数据集已创建: {demo_file}")
    return demo_file


# 测试和演示代码
def test_dataset():
    """测试数据集功能"""
    print("📚 测试简化MR2数据集")
    
    # 设置数据目录
    data_dir = "data"
    
    try:
        # 尝试创建数据集
        print(f"\n📂 尝试加载数据集...")
        dataset = SimpleMR2Dataset(
            data_dir=data_dir,
            split='train',
            transform_type='val',  # 使用验证模式，减少随机性
            load_images=True
        )
        
        if len(dataset) == 0:
            print("❌ 数据集为空，创建演示数据...")
            create_demo_dataset(data_dir, 'train')
            
            # 重新加载
            dataset = SimpleMR2Dataset(
                data_dir=data_dir,
                split='train',
                load_images=False  # 演示数据没有真实图像
            )
        
        print(f"✅ 数据集创建成功，大小: {len(dataset)}")
        
        # 获取统计信息
        stats = dataset.get_statistics()
        print(f"\n📊 数据集统计:")
        print(f"   总样本数: {stats['total_samples']}")
        print(f"   标签分布: {stats['label_distribution']}")
        if 'text_length_stats' in stats and stats['text_length_stats']:
            print(f"   文本长度: 平均 {stats['text_length_stats']['mean']:.1f}, "
                  f"范围 {stats['text_length_stats']['min']}-{stats['text_length_stats']['max']}")
        
        # 测试样本访问
        if len(dataset) > 0:
            dataset.print_sample_info(0)
            
            # 测试多个样本
            print(f"\n🧪 测试多个样本:")
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"   样本 {i}: 标签={sample['label']}, 文本长度={sample['text_length']}, 有图像={sample['has_image']}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None


if __name__ == "__main__":
    # 运行测试
    dataset = test_dataset()
    
    if dataset and len(dataset) > 0:
        print(f"\n✅ 简化MR2数据集测试完成")
        print(f"数据集可以正常使用，包含 {len(dataset)} 个样本")
    else:
        print(f"\n⚠️  数据集测试未完全成功")
        print(f"请检查数据文件是否存在")
    
    print(f"\n📝 使用说明:")
    print(f"1. 确保数据文件存在于 data/ 目录下")
    print(f"2. 如果没有真实数据，程序会自动创建演示数据")
    print(f"3. 可以通过设置 load_images=False 来跳过图像加载")