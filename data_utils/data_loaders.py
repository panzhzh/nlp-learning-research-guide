#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# data_utils/data_loaders.py

"""
强制使用真实数据集的数据加载器
不再支持演示数据，找不到数据集就报错
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入必要模块
try:
    from .mr2_dataset import MR2Dataset
    USE_CUSTOM_MODULES = True
    print("✅ 成功导入自定义模块 (相对导入)")
except ImportError:
    try:
        from mr2_dataset import MR2Dataset
        USE_CUSTOM_MODULES = True
        print("✅ 成功导入自定义模块 (直接导入)")
    except ImportError:
        try:
            import sys
            data_utils_path = current_file.parent
            sys.path.insert(0, str(data_utils_path))
            from mr2_dataset import MR2Dataset
            USE_CUSTOM_MODULES = True
            print("✅ 成功导入自定义模块 (路径导入)")
        except ImportError as e:
            print(f"❌ 无法导入MR2Dataset: {e}")
            print("❌ 系统要求必须使用真实数据集，不支持占位符数据")
            raise ImportError("❌ 必须导入MR2Dataset模块才能继续")

# 导入配置管理器
try:
    from utils.config_manager import get_config_manager, get_data_dir, check_data_requirements
    USE_CONFIG = True
    print("✅ 成功导入配置管理器")
except ImportError as e:
    print(f"❌ 无法导入配置管理器: {e}")
    raise ImportError("❌ 必须导入配置管理器才能继续")


class StrictDataLoaderConfig:
    """严格的数据加载器配置 - 只支持真实数据"""
    
    def __init__(self):
        """初始化配置"""
        # 检查数据要求
        try:
            check_data_requirements()
            print("✅ 数据要求检查通过")
        except Exception as e:
            print(f"❌ 数据要求检查失败: {e}")
            raise
        
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置"""
        try:
            config_manager = get_config_manager()
            training_config = config_manager.get_training_config()
            
            # 从配置中提取关键参数
            general_config = training_config.get('general', {}).get('data', {})
            return {
                'train_batch_size': general_config.get('train_batch_size', 32),
                'eval_batch_size': general_config.get('eval_batch_size', 64),
                'num_workers': general_config.get('data_workers', 4),
                'pin_memory': general_config.get('pin_memory', True)
            }
        except Exception as e:
            print(f"⚠️  加载配置失败，使用默认配置: {e}")
            return {
                'train_batch_size': 32,
                'eval_batch_size': 64,
                'num_workers': 4,
                'pin_memory': True
            }


class StrictCollateFunction:
    """严格的批处理函数 - 要求数据完整"""
    
    def __init__(self):
        """初始化"""
        pass
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        严格的批处理函数
        
        Args:
            batch: 批次数据列表
            
        Returns:
            批处理后的数据字典
        """
        if not batch:
            raise ValueError("❌ 批次数据为空，无法进行批处理")
        
        batch_data = {}
        
        # 处理基本字段
        for key in ['item_id', 'label', 'text', 'caption']:
            if key in batch[0]:
                batch_data[key] = [item.get(key) for item in batch]
        
        # 验证必要字段
        if 'label' not in batch_data and 'caption' not in batch_data and 'text' not in batch_data:
            raise ValueError("❌ 批次数据缺少必要字段 (label, text/caption)")
        
        # 处理标签（转换为tensor）
        if 'label' in batch_data:
            try:
                batch_data['labels'] = torch.tensor(batch_data['label'], dtype=torch.long)
            except Exception as e:
                raise ValueError(f"❌ 标签转换失败: {e}")
        
        # 处理图像（如果存在）
        if 'image' in batch[0]:
            try:
                images = []
                for item in batch:
                    if 'image' in item and item['image'] is not None:
                        images.append(item['image'])
                    else:
                        # 对于缺失图像，创建零tensor
                        images.append(torch.zeros(3, 224, 224))
                
                if images:
                    batch_data['images'] = torch.stack(images)
            except Exception as e:
                print(f"⚠️  图像批处理失败: {e}")
                batch_data['images'] = torch.zeros(len(batch), 3, 224, 224)
        
        return batch_data


def create_strict_dataloader(split: str = 'train',
                           batch_size: Optional[int] = None,
                           shuffle: Optional[bool] = None,
                           num_workers: int = 4) -> DataLoader:
    """
    创建严格的数据加载器 - 必须使用真实数据集
    
    Args:
        split: 数据划分 ('train', 'val', 'test')
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        
    Returns:
        DataLoader对象
        
    Raises:
        FileNotFoundError: 找不到数据文件
        ValueError: 数据集为空或不符合要求
    """
    try:
        # 加载配置
        config = StrictDataLoaderConfig()
        
        # 获取数据目录
        data_dir = get_data_dir()
        print(f"📂 使用数据目录: {data_dir}")
        
        # 设置默认参数
        if batch_size is None:
            batch_size = config.config['train_batch_size'] if split == 'train' else config.config['eval_batch_size']
        
        if shuffle is None:
            shuffle = (split == 'train')
        
        print(f"📂 创建 {split} 数据加载器...")
        print(f"   批次大小: {batch_size}")
        print(f"   是否打乱: {shuffle}")
        
        # 创建真实数据集
        dataset = MR2Dataset(
            data_dir=str(data_dir),
            split=split,
            transform_type='train' if split == 'train' else 'val',
            load_images=True
        )
        
        # 严格验证数据集
        if len(dataset) == 0:
            raise ValueError(f"❌ {split} 数据集为空，无法创建数据加载器")
        
        # 检查最小样本数要求
        data_config = get_config_manager().get_data_config()
        min_samples = data_config.get('dataset', {}).get('requirements', {}).get('min_samples_per_split', 10)
        
        if len(dataset) < min_samples:
            raise ValueError(f"❌ {split} 数据集样本数不足: {len(dataset)} < {min_samples}")
        
        print(f"✅ {split} 数据集验证通过: {len(dataset)} 样本")
        
        # 创建collate函数
        collate_fn = StrictCollateFunction()
        
        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=min(num_workers, 4),
            collate_fn=collate_fn,
            pin_memory=config.config['pin_memory'] and torch.cuda.is_available(),
            drop_last=False
        )
        
        print(f"✅ {split} 数据加载器创建成功")
        
        return dataloader
        
    except FileNotFoundError as e:
        print(f"❌ 数据文件不存在: {e}")
        print("请确保MR2数据集已下载并解压到正确位置")
        raise
    except ValueError as e:
        print(f"❌ 数据验证失败: {e}")
        raise
    except Exception as e:
        print(f"❌ 创建数据加载器失败: {e}")
        raise


def create_all_dataloaders(batch_sizes: Optional[Dict[str, int]] = None) -> Dict[str, DataLoader]:
    """
    创建所有数据加载器 - 严格模式
    
    Args:
        batch_sizes: 各数据集的批次大小
        
    Returns:
        数据加载器字典
        
    Raises:
        Exception: 任何数据加载失败都会抛出异常
    """
    if batch_sizes is None:
        batch_sizes = {'train': 32, 'val': 64, 'test': 64}
    
    dataloaders = {}
    errors = []
    
    for split in ['train', 'val', 'test']:
        print(f"\n📂 创建 {split} 数据加载器...")
        
        try:
            batch_size = batch_sizes.get(split, 32)
            dataloader = create_strict_dataloader(
                split=split,
                batch_size=batch_size,
                num_workers=2  # 减少工作进程避免问题
            )
            
            dataloaders[split] = dataloader
            print(f"✅ {split} 数据加载器创建成功")
            
        except Exception as e:
            error_msg = f"❌ {split} 数据加载器创建失败: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    # 如果有任何错误，抛出异常
    if errors:
        error_summary = "\n".join(errors)
        raise RuntimeError(
            f"❌ 数据加载器创建失败:\n{error_summary}\n\n"
            f"请检查:\n"
            f"1. MR2数据集是否已下载并解压\n"
            f"2. 数据文件是否在正确路径: {get_data_dir()}\n"
            f"3. 数据文件格式是否正确\n"
            f"下载链接: https://pan.baidu.com/s/1sfUwsaeV2nfl54OkrfrKVw?pwd=jxhc"
        )
    
    # 验证所有数据加载器都已创建
    required_splits = ['train', 'val', 'test']
    missing_splits = [split for split in required_splits if split not in dataloaders]
    
    if missing_splits:
        raise RuntimeError(f"❌ 缺少必要的数据分割: {missing_splits}")
    
    print(f"\n✅ 所有数据加载器创建成功")
    for split, loader in dataloaders.items():
        print(f"   {split}: {len(loader.dataset)} 样本, 批次大小 {loader.batch_size}")
    
    return dataloaders


def test_dataloader(dataloader: DataLoader, max_batches: int = 3):
    """
    测试数据加载器 - 严格验证
    
    Args:
        dataloader: 要测试的数据加载器
        max_batches: 最大测试批次数
    """
    print(f"\n🧪 测试数据加载器 (最多 {max_batches} 个批次)")
    
    try:
        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            print(f"  批次 {batch_idx}:")
            
            if not isinstance(batch, dict):
                raise ValueError(f"❌ 批次数据类型错误: {type(batch)}")
            
            print(f"    数据键: {list(batch.keys())}")
            
            # 验证必要字段
            if 'labels' not in batch and 'label' not in batch:
                raise ValueError("❌ 批次缺少标签字段")
            
            if 'labels' in batch:
                if not isinstance(batch['labels'], torch.Tensor):
                    raise ValueError("❌ 标签不是tensor类型")
                print(f"    标签: {batch['labels']}")
                batch_size = len(batch['labels'])
            elif 'label' in batch:
                print(f"    标签: {batch['label']}")
                batch_size = len(batch['label'])
            
            print(f"    批次大小: {batch_size}")
            
            # 验证文本字段
            if 'text' in batch:
                if len(batch['text']) != batch_size:
                    raise ValueError(f"❌ 文本数量与批次大小不匹配: {len(batch['text'])} != {batch_size}")
                print(f"    文本样例: {batch['text'][0][:50]}...")
            elif 'caption' in batch:
                if len(batch['caption']) != batch_size:
                    raise ValueError(f"❌ 文本数量与批次大小不匹配: {len(batch['caption'])} != {batch_size}")
                print(f"    文本样例: {batch['caption'][0][:50]}...")
            
            # 验证图像字段
            if 'images' in batch:
                if not isinstance(batch['images'], torch.Tensor):
                    raise ValueError("❌ 图像不是tensor类型")
                print(f"    图像形状: {batch['images'].shape}")
                expected_shape = (batch_size, 3, 224, 224)
                if batch['images'].shape != expected_shape:
                    print(f"    ⚠️  图像形状与预期不符: {batch['images'].shape} != {expected_shape}")
            
            batch_count += 1
            if batch_count >= max_batches:
                break
        
        print("✅ 数据加载器测试通过")
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        raise


def demo_usage():
    """演示严格数据加载器的使用方法"""
    print("🔄 演示严格数据加载器使用方法")
    print("⚠️  注意: 此版本要求必须有真实数据集")
    
    try:
        # 检查数据要求
        check_data_requirements()
        
        # 创建单个数据加载器
        print(f"\n📚 === 创建单个数据加载器 ===")
        train_loader = create_strict_dataloader(
            split='train',
            batch_size=8,
            shuffle=True
        )
        
        if train_loader:
            test_dataloader(train_loader, max_batches=2)
        
        # 创建所有数据加载器
        print(f"\n📚 === 创建所有数据加载器 ===")
        all_loaders = create_all_dataloaders(
            batch_sizes={'train': 4, 'val': 8, 'test': 8}
        )
        
        print(f"\n📊 数据加载器摘要:")
        for split, loader in all_loaders.items():
            print(f"  {split}: {len(loader.dataset)} 样本, 批次大小 {loader.batch_size}")
        
        # 测试一个数据加载器
        print(f"\n🧪 测试验证集数据加载器:")
        test_dataloader(all_loaders['val'], max_batches=1)
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        print("\n💡 解决方案:")
        print("1. 确保MR2数据集已下载")
        print("2. 检查数据文件路径是否正确")
        print("3. 验证数据文件格式是否完整")
        raise


# 为了向后兼容，保留原有函数名
def create_simple_dataloader(*args, **kwargs):
    """向后兼容的函数名"""
    return create_strict_dataloader(*args, **kwargs)

def create_mr2_dataloaders(*args, **kwargs):
    """向后兼容的函数名"""
    return create_all_dataloaders(*args, **kwargs)


# 主执行代码
if __name__ == "__main__":
    print("🔄 测试严格数据加载器配置")
    
    try:
        # 运行演示
        demo_usage()
        print("\n✅ 严格数据加载器测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("请确保MR2数据集已正确安装")
        sys.exit(1)