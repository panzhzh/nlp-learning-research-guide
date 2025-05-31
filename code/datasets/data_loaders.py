#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# datasets/data_loaders.py

"""
MR2数据加载器配置模块
提供灵活的数据加载配置，支持：
- 多种批处理策略
- 自适应数据增强
- 多进程数据加载
- 内存优化
- 实验重现性配置
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import sys
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_file = Path(__file__).resolve()
code_root = current_file.parent.parent
sys.path.append(str(code_root))

# 导入自定义模块
try:
    # 尝试多种导入方式
    try:
        from datasets.mr2_dataset import MR2Dataset
    except ImportError:
        # 如果上面失败，尝试直接导入
        import sys
        from pathlib import Path
        current_file = Path(__file__).resolve()
        datasets_dir = current_file.parent
        sys.path.insert(0, str(datasets_dir))
        from mr2_dataset import MR2Dataset
    
    from utils.config_manager import get_training_config, get_data_config
    USE_CUSTOM_MODULES = True
    print("✅ 成功导入自定义模块")
except ImportError as e:
    print(f"⚠️  导入自定义模块失败: {e}")
    USE_CUSTOM_MODULES = False
    # 定义MR2Dataset占位符
    class MR2Dataset:
        def __init__(self, *args, **kwargs):
            self.items = []
        
        def __len__(self):
            return len(self.items)
        
        def __getitem__(self, idx):
            return {'item_id': 'dummy', 'label': 0}

import logging
logger = logging.getLogger(__name__)


class MR2DataLoaderConfig:
    """
    MR2数据加载器配置类
    管理数据加载的各种参数和策略
    """
    
    def __init__(self):
        """初始化配置"""
        self.load_config()
        self.setup_default_config()
    
    def load_config(self):
        """加载配置文件"""
        if USE_CUSTOM_MODULES:
            try:
                self.training_config = get_training_config()
                self.data_config = get_data_config()
            except:
                self.training_config = {}
                self.data_config = {}
        else:
            self.training_config = {}
            self.data_config = {}
    
    def setup_default_config(self):
        """设置默认配置"""
        # 通用数据加载配置
        general_config = self.training_config.get('general', {}).get('data', {})
        
        self.default_config = {
            # 批次配置
            'train_batch_size': general_config.get('train_batch_size', 32),
            'eval_batch_size': general_config.get('eval_batch_size', 64),
            'test_batch_size': general_config.get('eval_batch_size', 64),
            
            # 工作进程配置
            'num_workers': general_config.get('data_workers', 4),
            'pin_memory': general_config.get('pin_memory', True),
            'persistent_workers': general_config.get('persistent_workers', True),
            
            # 采样配置
            'shuffle_train': True,
            'shuffle_val': False,
            'drop_last': False,
            
            # 数据增强配置
            'use_augmentation': True,
            'augmentation_prob': 0.5,
            
            # 内存优化
            'prefetch_factor': 2,
            'timeout': 0,
            
            # 分布式训练
            'distributed': False,
            'world_size': 1,
            'rank': 0
        }


class AdvancedCollateFunction:
    """
    高级批处理函数
    支持多模态数据的智能批处理
    """
    
    def __init__(self, 
                 max_text_length: int = 512,
                 pad_token_id: int = 0,
                 return_attention_mask: bool = True):
        """
        初始化批处理函数
        
        Args:
            max_text_length: 最大文本长度
            pad_token_id: 填充token ID
            return_attention_mask: 是否返回注意力掩码
        """
        self.max_text_length = max_text_length
        self.pad_token_id = pad_token_id
        self.return_attention_mask = return_attention_mask
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批处理函数
        
        Args:
            batch: 批次数据列表
            
        Returns:
            批处理后的数据字典
        """
        batch_size = len(batch)
        batch_data = {}
        
        # 处理基本字段
        for key in ['item_id', 'label', 'language', 'source', 'has_image']:
            if key in batch[0]:
                batch_data[key] = [item[key] for item in batch]
        
        # 处理数值型标签
        if 'label' in batch_data:
            batch_data['labels'] = torch.tensor(batch_data['label'], dtype=torch.long)
        
        # 处理图像数据
        if 'image' in batch[0]:
            images = []
            for item in batch:
                if item.get('has_image', False):
                    images.append(item['image'])
                else:
                    # 创建空图像tensor
                    images.append(torch.zeros_like(batch[0]['image']))
            
            batch_data['images'] = torch.stack(images)
            batch_data['image_mask'] = torch.tensor([item.get('has_image', False) for item in batch])
        
        # 处理文本数据
        if 'text' in batch[0]:
            texts = [item['text'] for item in batch]
            batch_data['texts'] = texts
            
            # 处理文本tokens（如果存在）
            if 'text_tokens' in batch[0]:
                batch_data['text_tokens'] = [item['text_tokens'] for item in batch]
        
        # 处理文本特征
        if 'text_features' in batch[0]:
            # 收集数值型特征
            feature_keys = []
            if batch[0]['text_features']:
                feature_keys = [k for k, v in batch[0]['text_features'].items() 
                              if isinstance(v, (int, float))]
            
            if feature_keys:
                feature_matrix = []
                for item in batch:
                    features = item.get('text_features', {})
                    feature_vector = [features.get(k, 0) for k in feature_keys]
                    feature_matrix.append(feature_vector)
                
                batch_data['text_feature_matrix'] = torch.tensor(feature_matrix, dtype=torch.float)
                batch_data['text_feature_names'] = feature_keys
        
        # 处理检索信息
        retrieval_fields = ['has_direct_retrieval', 'has_inverse_retrieval']
        for field in retrieval_fields:
            if field in batch[0]:
                batch_data[field] = torch.tensor([item.get(field, False) for item in batch])
        
        # 处理图像信息
        if 'image_info' in batch[0]:
            batch_data['image_info'] = [item.get('image_info', {}) for item in batch]
        
        return batch_data


class BalancedBatchSampler:
    """
    平衡批次采样器
    确保每个批次中各类别样本相对平衡
    """
    
    def __init__(self, 
                 dataset: MR2Dataset,
                 batch_size: int,
                 samples_per_class: Optional[int] = None):
        """
        初始化平衡采样器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            samples_per_class: 每个类别的样本数
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        
        # 构建标签到索引的映射
        self.label_to_indices = {}
        for idx, item in enumerate(dataset.items):
            label = item.get('label', -1)
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        self.labels = list(self.label_to_indices.keys())
        self.num_classes = len(self.labels)
        
        # 计算每个类别在批次中的样本数
        if self.samples_per_class is None:
            self.samples_per_class = max(1, batch_size // self.num_classes)
        
        print(f"🎯 平衡采样器初始化:")
        print(f"   类别数: {self.num_classes}")
        print(f"   每类样本数: {self.samples_per_class}")
        for label in self.labels:
            print(f"   标签 {label}: {len(self.label_to_indices[label])} 样本")
    
    def __iter__(self):
        """迭代器"""
        while True:
            batch_indices = []
            
            for label in self.labels:
                # 随机选择该类别的样本
                available_indices = self.label_to_indices[label]
                if len(available_indices) >= self.samples_per_class:
                    selected_indices = np.random.choice(
                        available_indices, 
                        size=self.samples_per_class, 
                        replace=False
                    )
                else:
                    # 如果样本不足，使用重复采样
                    selected_indices = np.random.choice(
                        available_indices, 
                        size=self.samples_per_class, 
                        replace=True
                    )
                
                batch_indices.extend(selected_indices)
            
            # 随机打乱批次内的顺序
            np.random.shuffle(batch_indices)
            
            # 确保批次大小
            if len(batch_indices) > self.batch_size:
                batch_indices = batch_indices[:self.batch_size]
            elif len(batch_indices) < self.batch_size:
                # 随机补充样本
                all_indices = list(range(len(self.dataset)))
                additional_needed = self.batch_size - len(batch_indices)
                additional_indices = np.random.choice(
                    all_indices, 
                    size=additional_needed, 
                    replace=True
                )
                batch_indices.extend(additional_indices)
            
            yield batch_indices
    
    def __len__(self):
        """返回一个epoch的批次数"""
        return len(self.dataset) // self.batch_size


class WeightedSamplerCreator:
    """
    加权采样器创建器
    根据类别分布创建加权采样器
    """
    
    @staticmethod
    def create_weighted_sampler(dataset: MR2Dataset, 
                               sampling_strategy: str = 'inverse_freq') -> WeightedRandomSampler:
        """
        创建加权随机采样器
        
        Args:
            dataset: 数据集
            sampling_strategy: 采样策略 ('inverse_freq', 'sqrt_inv_freq', 'balanced')
            
        Returns:
            WeightedRandomSampler对象
        """
        # 统计标签分布
        labels = [item.get('label', -1) for item in dataset.items]
        label_counts = Counter(labels)
        
        # 计算权重
        if sampling_strategy == 'inverse_freq':
            # 逆频率权重
            total_samples = len(labels)
            weights = {label: total_samples / count for label, count in label_counts.items()}
        
        elif sampling_strategy == 'sqrt_inv_freq':
            # 平方根逆频率权重
            total_samples = len(labels)
            weights = {label: np.sqrt(total_samples / count) for label, count in label_counts.items()}
        
        elif sampling_strategy == 'balanced':
            # 平衡权重
            max_count = max(label_counts.values())
            weights = {label: max_count / count for label, count in label_counts.items()}
        
        else:
            raise ValueError(f"不支持的采样策略: {sampling_strategy}")
        
        # 为每个样本分配权重
        sample_weights = [weights[label] for label in labels]
        
        print(f"📊 加权采样器统计:")
        print(f"   采样策略: {sampling_strategy}")
        for label, count in label_counts.items():
            weight = weights[label]
            print(f"   标签 {label}: {count} 样本, 权重 {weight:.3f}")
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


class DataLoaderFactory:
    """
    数据加载器工厂类
    根据不同需求创建合适的数据加载器
    """
    
    def __init__(self, config: Optional[MR2DataLoaderConfig] = None):
        """
        初始化工厂
        
        Args:
            config: 数据加载器配置
        """
        self.config = config if config is not None else MR2DataLoaderConfig()
    
    def create_basic_dataloader(self,
                               dataset: MR2Dataset,
                               split: str = 'train',
                               batch_size: Optional[int] = None,
                               shuffle: Optional[bool] = None,
                               **kwargs) -> DataLoader:
        """
        创建基础数据加载器
        
        Args:
            dataset: 数据集
            split: 数据划分
            batch_size: 批次大小
            shuffle: 是否打乱
            **kwargs: 其他参数
            
        Returns:
            DataLoader对象
        """
        # 确定批次大小
        if batch_size is None:
            if split == 'train':
                batch_size = self.config.default_config['train_batch_size']
            else:
                batch_size = self.config.default_config['eval_batch_size']
        
        # 确定是否打乱
        if shuffle is None:
            shuffle = split == 'train' and self.config.default_config['shuffle_train']
        
        # 创建批处理函数
        collate_fn = AdvancedCollateFunction()
        
        # 合并配置
        dataloader_config = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': self.config.default_config['num_workers'],
            'pin_memory': self.config.default_config['pin_memory'],
            'drop_last': self.config.default_config['drop_last'],
            'collate_fn': collate_fn,
            'persistent_workers': self.config.default_config['persistent_workers'] and self.config.default_config['num_workers'] > 0
        }
        
        # 更新用户提供的参数
        dataloader_config.update(kwargs)
        
        return DataLoader(dataset, **dataloader_config)
    
    def create_balanced_dataloader(self,
                                  dataset: MR2Dataset,
                                  batch_size: int,
                                  sampling_strategy: str = 'weighted',
                                  **kwargs) -> DataLoader:
        """
        创建平衡数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            sampling_strategy: 平衡策略 ('weighted', 'batch_balanced')
            **kwargs: 其他参数
            
        Returns:
            DataLoader对象
        """
        collate_fn = AdvancedCollateFunction()
        
        if sampling_strategy == 'weighted':
            # 使用加权采样
            sampler = WeightedSamplerCreator.create_weighted_sampler(dataset)
            
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=self.config.default_config['num_workers'],
                pin_memory=self.config.default_config['pin_memory'],
                collate_fn=collate_fn,
                **kwargs
            )
        
        elif sampling_strategy == 'batch_balanced':
            # 使用批次平衡采样
            batch_sampler = BalancedBatchSampler(dataset, batch_size)
            
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.config.default_config['num_workers'],
                pin_memory=self.config.default_config['pin_memory'],
                collate_fn=collate_fn,
                **kwargs
            )
        
        else:
            raise ValueError(f"不支持的平衡策略: {sampling_strategy}")
    
    def create_distributed_dataloader(self,
                                     dataset: MR2Dataset,
                                     batch_size: int,
                                     world_size: int,
                                     rank: int,
                                     **kwargs) -> DataLoader:
        """
        创建分布式数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            world_size: 总进程数
            rank: 当前进程rank
            **kwargs: 其他参数
            
        Returns:
            DataLoader对象
        """
        # 创建分布式采样器
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        collate_fn = AdvancedCollateFunction()
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.config.default_config['num_workers'],
            pin_memory=self.config.default_config['pin_memory'],
            collate_fn=collate_fn,
            **kwargs
        )
    
    def create_evaluation_dataloader(self,
                                    dataset: MR2Dataset,
                                    batch_size: Optional[int] = None,
                                    **kwargs) -> DataLoader:
        """
        创建评估数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            **kwargs: 其他参数
            
        Returns:
            DataLoader对象
        """
        if batch_size is None:
            batch_size = self.config.default_config['eval_batch_size']
        
        return self.create_basic_dataloader(
            dataset=dataset,
            split='eval',
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )


def create_mr2_dataloaders(data_dir: Union[str, Path],
                          batch_sizes: Optional[Dict[str, int]] = None,
                          sampling_strategy: str = 'basic',
                          num_workers: int = 4,
                          **dataset_kwargs) -> Dict[str, DataLoader]:
    """
    创建完整的MR2数据加载器集合
    
    Args:
        data_dir: 数据目录
        batch_sizes: 各数据集的批次大小
        sampling_strategy: 采样策略
        num_workers: 工作进程数
        **dataset_kwargs: 数据集参数
        
    Returns:
        包含train/val/test数据加载器的字典
    """
    if batch_sizes is None:
        batch_sizes = {'train': 32, 'val': 64, 'test': 64}
    
    # 创建工厂和配置
    config = MR2DataLoaderConfig()
    config.default_config['num_workers'] = num_workers
    factory = DataLoaderFactory(config)
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            print(f"\n📂 创建 {split} 数据加载器...")
            
            # 创建数据集
            dataset = MR2Dataset(
                data_dir=data_dir,
                split=split,
                transform_type='train' if split == 'train' else 'val',
                **dataset_kwargs
            )
            
            # 根据策略创建数据加载器
            batch_size = batch_sizes.get(split, 32)
            
            if split == 'train' and sampling_strategy == 'balanced':
                dataloader = factory.create_balanced_dataloader(
                    dataset=dataset,
                    batch_size=batch_size,
                    sampling_strategy='weighted'
                )
            elif split == 'train' and sampling_strategy == 'batch_balanced':
                dataloader = factory.create_balanced_dataloader(
                    dataset=dataset,
                    batch_size=batch_size,
                    sampling_strategy='batch_balanced'
                )
            else:
                dataloader = factory.create_basic_dataloader(
                    dataset=dataset,
                    split=split,
                    batch_size=batch_size
                )
            
            dataloaders[split] = dataloader
            print(f"✅ {split} 数据加载器创建成功 (批次大小: {batch_size})")
            
        except Exception as e:
            logger.warning(f"创建 {split} 数据加载器失败: {e}")
            continue
    
    return dataloaders


def analyze_dataloader_performance(dataloader: DataLoader, 
                                  num_batches: int = 10) -> Dict[str, Any]:
    """
    分析数据加载器性能
    
    Args:
        dataloader: 数据加载器
        num_batches: 分析的批次数
        
    Returns:
        性能分析结果
    """
    import time
    
    print(f"🔍 分析数据加载器性能...")
    
    start_time = time.time()
    batch_times = []
    batch_sizes = []
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()
        
        # 获取批次信息
        if isinstance(batch, dict):
            batch_size = len(batch.get('item_id', []))
            has_images = batch.get('images') is not None
            has_texts = batch.get('texts') is not None
        else:
            batch_size = len(batch)
            has_images = False
            has_texts = False
        
        batch_end = time.time()
        batch_time = batch_end - batch_start
        
        batch_times.append(batch_time)
        batch_sizes.append(batch_size)
        
        if batch_idx >= num_batches - 1:
            break
    
    total_time = time.time() - start_time
    
    analysis = {
        'total_time': total_time,
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'avg_batch_size': np.mean(batch_sizes),
        'total_samples': sum(batch_sizes),
        'samples_per_second': sum(batch_sizes) / total_time,
        'batches_analyzed': len(batch_times)
    }
    
    print(f"📊 性能分析结果:")
    print(f"   总时间: {analysis['total_time']:.2f}s")
    print(f"   平均批次时间: {analysis['avg_batch_time']:.3f}s")
    print(f"   平均批次大小: {analysis['avg_batch_size']:.1f}")
    print(f"   处理速度: {analysis['samples_per_second']:.1f} 样本/秒")
    
    return analysis


# 使用示例和测试代码
if __name__ == "__main__":
    print("🔄 测试数据加载器配置")
    
    # 设置数据目录
    data_dir = "data"
    
    if not USE_CUSTOM_MODULES:
        print("⚠️  自定义模块不可用，跳过测试")
        print("✅ 数据加载器测试完成")
        exit()
    
    try:
        # 测试基础数据加载器
        print(f"\n📦 === 测试基础数据加载器 ===")
        
        # 创建单个数据加载器
        config = MR2DataLoaderConfig()
        factory = DataLoaderFactory(config)
        
        # 创建训练数据集
        train_dataset = MR2Dataset(
            data_dir=data_dir,
            split='train',
            transform_type='train',
            use_cache=True
        )
        
        # 基础数据加载器
        basic_loader = factory.create_basic_dataloader(
            dataset=train_dataset,
            split='train',
            batch_size=8
        )
        
        print(f"✅ 基础数据加载器创建成功")
        
        # 测试批次
        print(f"\n🧪 测试批次数据:")
        for batch_idx, batch in enumerate(basic_loader):
            print(f"   批次 {batch_idx}:")
            print(f"     批次大小: {len(batch['item_id'])}")
            print(f"     图像形状: {batch['images'].shape}")
            print(f"     标签: {batch['labels']}")
            
            if batch_idx >= 2:
                break
        
        print(f"✅ 数据加载器测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print(f"建议使用简化版本: python datasets/simple_mr2_dataset.py")