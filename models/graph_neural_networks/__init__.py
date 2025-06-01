#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/graph_neural_networks/__init__.py

"""
图神经网络模块
包含基础GNN层、社交图构建、多模态GNN等功能
专门为MR2数据集和谣言检测任务设计
"""

from .basic_gnn_layers import (
    GCNLayer, GATLayer, GraphSAGELayer, GINLayer,
    BasicGNN, GNNClassifier
)

from .social_graph_builder import (
    SocialGraphBuilder, GraphFeatureExtractor
)

from .multimodal_gnn import (
    MultimodalGNN, MultimodalGraphClassifier
)

__version__ = "1.0.0"
__author__ = "ipanzhzh"

__all__ = [
    # 基础GNN层
    'GCNLayer',
    'GATLayer', 
    'GraphSAGELayer',
    'GINLayer',
    'BasicGNN',
    'GNNClassifier',
    
    # 社交图构建
    'SocialGraphBuilder',
    'GraphFeatureExtractor',
    
    # 多模态GNN
    'MultimodalGNN',
    'MultimodalGraphClassifier'
]

# 模块信息
SUPPORTED_GNN_TYPES = {
    'gcn': 'Graph Convolutional Network',
    'gat': 'Graph Attention Network', 
    'graphsage': 'Graph Sample and Aggregate',
    'gin': 'Graph Isomorphism Network'
}

SUPPORTED_FUSION_METHODS = {
    'early': '早期融合 - 特征级融合',
    'late': '后期融合 - 决策级融合',
    'attention': '注意力融合 - 动态权重',
    'cross_modal': '跨模态融合 - 交互建模'
}

def get_gnn_info():
    """获取图神经网络模块信息"""
    return {
        'version': __version__,
        'author': __author__,
        'supported_gnn_types': SUPPORTED_GNN_TYPES,
        'supported_fusion_methods': SUPPORTED_FUSION_METHODS,
        'description': '图神经网络模块，支持多模态社交媒体分析'
    }

def list_available_models():
    """列出可用的GNN模型"""
    print("🔗 可用的图神经网络模型:")
    for gnn_type, description in SUPPORTED_GNN_TYPES.items():
        print(f"  • {gnn_type.upper()}: {description}")
    
    print("\n🔀 可用的融合方法:")
    for fusion_method, description in SUPPORTED_FUSION_METHODS.items():
        print(f"  • {fusion_method}: {description}")

if __name__ == "__main__":
    print("📊 图神经网络模块信息:")
    info = get_gnn_info()
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    - {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    list_available_models()