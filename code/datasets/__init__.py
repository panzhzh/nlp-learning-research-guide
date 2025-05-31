#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# datasets/__init__.py

"""
数据集模块
包含MR2数据集相关的所有组件
"""

try:
    from .mr2_dataset import MR2Dataset
    from .data_loaders import create_mr2_dataloaders, DataLoaderFactory
except ImportError as e:
    print(f"⚠️  导入数据集模块失败: {e}")