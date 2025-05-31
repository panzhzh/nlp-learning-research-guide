#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# datasets/data_loaders.py

"""
简化的MR2数据加载器
专注于核心功能，易于理解和使用
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# 添加项目路径
current_file = Path(__file__).resolve()
code_root = current_file.parent.parent
sys.path.append(str(code_root))

# 导入必要模块 - 修复导入问题
try:
    # 方法1: 尝试相对导入
    from .mr2_dataset import MR2Dataset
    USE_CUSTOM_MODULES = True
    print("✅ 成功导入自定义模块 (相对导入)")
except ImportError:
    try:
        # 方法2: 尝试直接导入
        from mr2_dataset import MR2Dataset
        USE_CUSTOM_MODULES = True
        print("✅ 成功导入自定义模块 (直接导入)")
    except ImportError:
        try:
            # 方法3: 尝试绝对路径导入
            import sys
            datasets_path = current_file.parent
            sys.path.insert(0, str(datasets_path))
            from mr2_dataset import MR2Dataset
            USE_CUSTOM_MODULES = True
            print("✅ 成功导入自定义模块 (绝对路径导入)")
        except ImportError as e:
            print(f"⚠️  导入自定义模块失败: {e}")
            USE_CUSTOM_MODULES = False
            
            # 创建简单的数据集占位符
            class MR2Dataset:
                def __init__(self, *args, **kwargs):
                    self.items = []
                    print("使用简化版MR2Dataset")
                
                def __len__(self):
                    return len(self.items)
                
                def __getitem__(self, idx):
                    return {'item_id': 'demo', 'label': 0, 'text': 'demo text'}

# 尝试导入配置管理器
try:
    from utils.config_manager import get_training_config, get_data_config
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False


class SimpleDataLoaderConfig:
    """简化的数据加载器配置"""
    
    def __init__(self):
        """初始化配置"""
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if USE_CONFIG:
            try:
                training_config = get_training_config()
                data_config = get_data_config()
                
                # 从配置中提取关键参数
                general_config = training_config.get('general', {}).get('data', {})
                return {
                    'train_batch_size': general_config.get('train_batch_size', 32),
                    'eval_batch_size': general_config.get('eval_batch_size', 64),
                    'num_workers': general_config.get('data_workers', 4),
                    'pin_memory': general_config.get('pin_memory', True)
                }
            except Exception as e:
                print(f"⚠️  加载配置失败: {e}")
        
        # 默认配置
        return {
            'train_batch_size': 32,
            'eval_batch_size': 64,
            'num_workers': 4,
            'pin_memory': True
        }


class SimpleCollateFunction:
    """简化的批处理函数"""
    
    def __init__(self):
        """初始化"""
        pass
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        简单的批处理函数
        
        Args:
            batch: 批次数据列表
            
        Returns:
            批处理后的数据字典
        """
        if not batch:
            return {}
        
        batch_data = {}
        
        # 处理基本字段 - 修复为支持caption字段
        for key in ['item_id', 'label', 'text', 'caption']:
            if key in batch[0]:
                batch_data[key] = [item.get(key) for item in batch]
        
        # 处理标签（转换为tensor）
        if 'label' in batch_data:
            batch_data['labels'] = torch.tensor(batch_data['label'], dtype=torch.long)
        
        # 处理图像（如果存在）
        if 'image' in batch[0]:
            try:
                images = []
                for item in batch:
                    if 'image' in item and item['image'] is not None:
                        images.append(item['image'])
                    else:
                        # 创建空图像tensor
                        images.append(torch.zeros(3, 224, 224))
                
                if images:
                    batch_data['images'] = torch.stack(images)
            except Exception as e:
                print(f"⚠️  处理图像失败: {e}")
                batch_data['images'] = torch.zeros(len(batch), 3, 224, 224)
        
        return batch_data


def create_simple_dataloader(data_dir: str,
                            split: str = 'train',
                            batch_size: Optional[int] = None,
                            shuffle: Optional[bool] = None,
                            num_workers: int = 4) -> Optional[DataLoader]:
    """
    创建简单的MR2数据加载器
    
    Args:
        data_dir: 数据目录
        split: 数据划分 ('train', 'val', 'test')
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        
    Returns:
        DataLoader对象或None
    """
    try:
        # 加载配置
        config = SimpleDataLoaderConfig()
        
        # 设置默认参数
        if batch_size is None:
            batch_size = config.config['train_batch_size'] if split == 'train' else config.config['eval_batch_size']
        
        if shuffle is None:
            shuffle = (split == 'train')
        
        print(f"📂 创建 {split} 数据加载器...")
        print(f"   批次大小: {batch_size}")
        print(f"   是否打乱: {shuffle}")
        
        if not USE_CUSTOM_MODULES:
            print("⚠️  使用占位符数据集，仅用于测试")
            dataset = MR2Dataset()
            # 创建一些虚拟数据
            dataset.items = [{'item_id': f'demo_{i}', 'label': i % 3, 'text': f'Demo text {i}'} for i in range(10)]
        else:
            # 创建真实数据集
            print(f"✅ 使用真实MR2数据集")
            dataset = MR2Dataset(
                data_dir=data_dir,
                split=split,
                transform_type='train' if split == 'train' else 'val',
                load_images=True  # 尝试加载图像
            )
        
        # 创建collate函数
        collate_fn = SimpleCollateFunction()
        
        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0 if not USE_CUSTOM_MODULES else min(num_workers, 4),  # 简化：减少进程数
            collate_fn=collate_fn,
            pin_memory=config.config['pin_memory'] and torch.cuda.is_available()
        )
        
        print(f"✅ 数据加载器创建成功")
        print(f"   数据集大小: {len(dataset)}")
        
        return dataloader
        
    except Exception as e:
        print(f"❌ 创建数据加载器失败: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return None


def create_all_dataloaders(data_dir: str,
                          batch_sizes: Optional[Dict[str, int]] = None) -> Dict[str, DataLoader]:
    """
    创建所有数据加载器
    
    Args:
        data_dir: 数据目录
        batch_sizes: 各数据集的批次大小
        
    Returns:
        数据加载器字典
    """
    if batch_sizes is None:
        batch_sizes = {'train': 32, 'val': 64, 'test': 64}
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\n📂 创建 {split} 数据加载器...")
        
        batch_size = batch_sizes.get(split, 32)
        dataloader = create_simple_dataloader(
            data_dir=data_dir,
            split=split,
            batch_size=batch_size,
            num_workers=2  # 简化：使用较少的工作进程
        )
        
        if dataloader is not None:
            dataloaders[split] = dataloader
            print(f"✅ {split} 数据加载器创建成功")
        else:
            print(f"❌ {split} 数据加载器创建失败")
    
    return dataloaders


def test_dataloader(dataloader: DataLoader, max_batches: int = 3):
    """
    测试数据加载器
    
    Args:
        dataloader: 要测试的数据加载器
        max_batches: 最大测试批次数
    """
    print(f"\n🧪 测试数据加载器 (最多 {max_batches} 个批次)")
    
    try:
        for batch_idx, batch in enumerate(dataloader):
            print(f"  批次 {batch_idx}:")
            
            if isinstance(batch, dict):
                print(f"    数据键: {list(batch.keys())}")
                
                if 'item_id' in batch:
                    print(f"    批次大小: {len(batch['item_id'])}")
                
                if 'labels' in batch:
                    print(f"    标签: {batch['labels']}")
                elif 'label' in batch:
                    print(f"    标签: {batch['label']}")
                
                if 'images' in batch:
                    print(f"    图像形状: {batch['images'].shape}")
                
                # 显示文本样例
                if 'text' in batch and batch['text']:
                    print(f"    文本样例: {batch['text'][0][:50]}...")
                elif 'caption' in batch and batch['caption']:
                    print(f"    文本样例: {batch['caption'][0][:50]}...")
            else:
                print(f"    批次类型: {type(batch)}")
            
            if batch_idx >= max_batches - 1:
                break
        
        print("✅ 数据加载器测试完成")
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")


def demo_usage():
    """演示用法"""
    print("🔄 演示数据加载器使用方法")
    
    # 设置数据目录
    data_dir = "data"  # 相对于code目录
    
    # 创建单个数据加载器
    print(f"\n📚 === 创建单个数据加载器 ===")
    train_loader = create_simple_dataloader(
        data_dir=data_dir,
        split='train',
        batch_size=8,
        shuffle=True
    )
    
    if train_loader:
        test_dataloader(train_loader, max_batches=2)
    
    # 创建所有数据加载器
    print(f"\n📚 === 创建所有数据加载器 ===")
    all_loaders = create_all_dataloaders(
        data_dir=data_dir,
        batch_sizes={'train': 4, 'val': 8, 'test': 8}
    )
    
    print(f"\n📊 数据加载器摘要:")
    for split, loader in all_loaders.items():
        print(f"  {split}: {len(loader.dataset)} 样本, 批次大小 {loader.batch_size}")


# 主执行代码
if __name__ == "__main__":
    print("🔄 测试数据加载器配置")
    
    if not USE_CUSTOM_MODULES:
        print("⚠️  自定义模块不可用，使用简化版本进行演示")
    else:
        print("✅ 自定义模块可用，使用真实数据集")
    
    # 运行演示
    demo_usage()
    
    print("\n✅ 数据加载器测试完成")