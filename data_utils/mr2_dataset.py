#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# data_utils/mr2_dataset.py

"""
严格的MR2数据集类
只支持真实数据集，不再提供演示数据fallback
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
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入配置管理
try:
    from utils.config_manager import get_data_config, get_data_dir, get_label_mapping, check_data_requirements
    USE_CONFIG = True
    print("✅ 成功导入配置管理器")
except ImportError as e:
    print(f"❌ 无法导入配置管理器: {e}")
    raise ImportError("❌ 必须导入配置管理器才能继续")

import logging
logger = logging.getLogger(__name__)


class MR2Dataset(Dataset):
    """
    严格的MR2多模态谣言检测数据集
    
    功能:
    - 只加载真实数据集
    - 找不到数据就报错
    - 严格的数据验证
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
            
        Raises:
            FileNotFoundError: 数据文件不存在
            ValueError: 数据格式错误或为空
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform_type = transform_type
        self.target_size = target_size
        self.load_images = load_images
        
        # 验证数据要求
        try:
            check_data_requirements()
        except Exception as e:
            raise RuntimeError(f"❌ 数据要求检查失败: {e}")
        
        # 加载配置
        self.setup_config()
        
        # 设置图像变换
        self.setup_transforms()
        
        # 加载数据集
        self.load_dataset()
        
        # 验证数据集
        self.validate_dataset()
        
        print(f"📚 MR2数据集初始化完成")
        print(f"   数据划分: {self.split}")
        print(f"   样本数量: {len(self.items)}")
        print(f"   加载图像: {self.load_images}")
        print(f"   标签分布: {self.get_label_distribution()}")
    
    def setup_config(self):
        """设置配置"""
        try:
            self.label_mapping = get_label_mapping()
            data_config = get_data_config()
            self.dataset_config = data_config.get('dataset', {})
        except Exception as e:
            raise RuntimeError(f"❌ 配置加载失败: {e}")
    
    def setup_transforms(self):
        """设置图像变换"""
        # 训练时的变换
        if self.transform_type == 'train':
            self.image_transforms = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.3),
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
        # 构建数据文件路径
        dataset_file = self.data_dir / f'dataset_items_{self.split}.json'
        
        # 检查文件是否存在
        if not dataset_file.exists():
            raise FileNotFoundError(
                f"❌ 数据集文件不存在: {dataset_file}\n"
                f"请确保MR2数据集已下载并解压到: {self.data_dir}\n"
                f"下载链接: https://pan.baidu.com/s/1sfUwsaeV2nfl54OkrfrKVw?pwd=jxhc"
            )
        
        # 加载JSON文件
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            print(f"✅ 成功加载数据文件: {dataset_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ JSON文件格式错误: {dataset_file}, 错误: {e}")
        except Exception as e:
            raise RuntimeError(f"❌ 加载数据文件失败: {dataset_file}, 错误: {e}")
        
        # 构建数据项列表
        self.items = []
        self.item_ids = []
        
        # 验证数据格式并构建数据项
        for item_id, item_data in self.raw_data.items():
            # 验证必要字段
            if not isinstance(item_data, dict):
                logger.warning(f"跳过无效数据项 {item_id}: 不是字典格式")
                continue
                
            if 'caption' not in item_data:
                logger.warning(f"跳过数据项 {item_id}: 缺少caption字段")
                continue
                
            if 'label' not in item_data:
                logger.warning(f"跳过数据项 {item_id}: 缺少label字段")
                continue
            
            # 验证标签值
            label = item_data['label']
            if not isinstance(label, int) or label not in self.label_mapping:
                logger.warning(f"跳过数据项 {item_id}: 无效标签 {label}")
                continue
            
            # 验证文本内容
            caption = item_data['caption']
            if not isinstance(caption, str) or len(caption.strip()) == 0:
                logger.warning(f"跳过数据项 {item_id}: 无效文本内容")
                continue
            
            # 添加有效数据项
            self.items.append(item_data)
            self.item_ids.append(item_id)
        
        print(f"📂 加载 {self.split} 数据: {len(self.items)} 个有效样本")
    
    def validate_dataset(self):
        """验证数据集"""
        # 检查数据集是否为空
        if len(self.items) == 0:
            raise ValueError(f"❌ {self.split} 数据集为空，无法继续")
        
        # 检查最小样本数要求
        min_samples = self.dataset_config.get('requirements', {}).get('min_samples_per_split', 10)
        if len(self.items) < min_samples:
            raise ValueError(f"❌ {self.split} 数据集样本数不足: {len(self.items)} < {min_samples}")
        
        # 验证标签分布
        label_counts = {}
        for item in self.items:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # 检查是否包含所有标签类别
        expected_labels = set(self.label_mapping.keys())
        found_labels = set(label_counts.keys())
        
        if not found_labels.issubset(expected_labels):
            invalid_labels = found_labels - expected_labels
            raise ValueError(f"❌ 发现无效标签: {invalid_labels}")
        
        # 警告缺失的标签类别
        missing_labels = expected_labels - found_labels
        if missing_labels:
            missing_names = [self.label_mapping[label] for label in missing_labels]
            logger.warning(f"⚠️  {self.split} 数据集缺少标签类别: {missing_names}")
        
        print(f"✅ {self.split} 数据集验证通过")
    
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
            
        Raises:
            IndexError: 索引超出范围
        """
        if idx >= len(self.items):
            raise IndexError(f"索引超出范围: {idx} >= {len(self.items)}")
        
        item = self.items[idx]
        item_id = self.item_ids[idx]
        
        # 构建基本数据项
        data_item = {
            'item_id': item_id,
            'text': item.get('caption', ''),
            'caption': item.get('caption', ''),  # 兼容性
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
                'image_path': item.get('image_path', None)
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
SimpleMR2Dataset = MR2Dataset


# 测试和演示代码
def test_dataset():
    """测试数据集功能"""
    print("📚 测试严格MR2数据集")
    
    try:
        # 获取数据目录
        data_dir = get_data_dir()
        
        # 尝试创建数据集
        print(f"\n📂 尝试加载数据集...")
        dataset = MR2Dataset(
            data_dir=data_dir,
            split='train',
            transform_type='val',  # 使用验证模式，减少随机性
            load_images=True
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
        print("\n💡 解决方案:")
        print("1. 确保MR2数据集已下载")
        print("2. 检查数据文件路径是否正确")
        print("3. 验证数据文件格式是否完整")
        raise


if __name__ == "__main__":
    # 运行测试
    try:
        dataset = test_dataset()
        
        if dataset and len(dataset) > 0:
            print(f"\n✅ 严格MR2数据集测试完成")
            print(f"数据集可以正常使用，包含 {len(dataset)} 个样本")
        
    except Exception as e:
        print(f"\n❌ 数据集测试失败: {e}")
        print("请确保MR2数据集已正确安装")
        sys.exit(1)
    
    print(f"\n📝 使用说明:")
    print(f"1. 必须确保数据文件存在于正确路径")
    print(f"2. 不再支持演示数据，必须使用真实数据集") 
    print(f"3. 数据集会进行严格验证，确保数据质量")