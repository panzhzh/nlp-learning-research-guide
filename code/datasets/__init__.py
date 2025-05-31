#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# datasets/__init__.py

"""
数据集模块
包含MR2数据集相关的所有组件
更新版本：适配简化的模块结构
"""

# 尝试导入简化版本的模块
try:
    # 优先导入简化版本
    from .mr2_dataset import SimpleMR2Dataset as MR2Dataset
    print("✅ 导入简化版MR2Dataset")
except ImportError:
    try:
        # fallback到原版本
        from .mr2_dataset import MR2Dataset
        print("✅ 导入原版MR2Dataset")
    except ImportError as e:
        print(f"❌ 导入MR2Dataset失败: {e}")
        MR2Dataset = None

# 尝试导入数据加载器
try:
    # 优先导入简化版本
    from .data_loaders import create_simple_dataloader, create_all_dataloaders, SimpleDataLoaderConfig
    print("✅ 导入简化版数据加载器")
    
    # 为了向后兼容，创建别名
    create_mr2_dataloaders = create_all_dataloaders
    DataLoaderFactory = SimpleDataLoaderConfig
    
except ImportError:
    try:
        # fallback到原版本
        from .data_loaders import create_mr2_dataloaders, DataLoaderFactory
        print("✅ 导入原版数据加载器")
    except ImportError as e:
        print(f"❌ 导入数据加载器失败: {e}")
        create_mr2_dataloaders = None
        DataLoaderFactory = None

# 导出的公共接口
__all__ = [
    'MR2Dataset',
    'create_mr2_dataloaders', 
    'DataLoaderFactory'
]

# 添加便捷函数
def get_available_components():
    """获取可用的组件信息"""
    components = {
        'MR2Dataset': MR2Dataset is not None,
        'create_mr2_dataloaders': create_mr2_dataloaders is not None,
        'DataLoaderFactory': DataLoaderFactory is not None
    }
    return components

def print_component_status():
    """打印组件可用状态"""
    print("📦 数据集模块组件状态:")
    components = get_available_components()
    for name, available in components.items():
        status = "✅ 可用" if available else "❌ 不可用"
        print(f"   {name}: {status}")

# 如果直接运行此模块，显示组件状态
if __name__ == "__main__":
    print_component_status()