#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/graph_neural_networks/basic_gnn_layers.py

"""
基础图神经网络层实现
包含GCN、GAT、GraphSAGE、GIN等经典GNN架构
支持多模态特征和MR2数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch_geometric
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入项目模块
try:
    from utils.config_manager import get_config_manager, get_output_path
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块")
except ImportError as e:
    print(f"⚠️  导入项目模块失败: {e}")
    USE_PROJECT_MODULES = False

import logging
logger = logging.getLogger(__name__)


class GCNLayer(nn.Module):
    """图卷积网络层"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 dropout: float = 0.5, bias: bool = True):
        """
        初始化GCN层
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            dropout: Dropout概率
            bias: 是否使用偏置
        """
        super(GCNLayer, self).__init__()
        
        self.conv = GCNConv(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges] (可选)
            
        Returns:
            输出特征 [num_nodes, output_dim]
        """
        x = self.conv(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class GATLayer(nn.Module):
    """图注意力网络层"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 heads: int = 8, dropout: float = 0.5, 
                 concat: bool = True, bias: bool = True):
        """
        初始化GAT层
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            heads: 注意力头数
            dropout: Dropout概率
            concat: 是否拼接多头结果
            bias: 是否使用偏置
        """
        super(GATLayer, self).__init__()
        
        self.heads = heads
        self.concat = concat
        
        if concat:
            assert output_dim % heads == 0
            self.head_dim = output_dim // heads
        else:
            self.head_dim = output_dim
            
        self.conv = GATConv(
            input_dim, self.head_dim, heads=heads,
            dropout=dropout, concat=concat, bias=bias
        )
        self.dropout = nn.Dropout(dropout)
        
        if not concat:
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x, edge_index, return_attention_weights=False):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            return_attention_weights: 是否返回注意力权重
            
        Returns:
            输出特征和可选的注意力权重
        """
        if return_attention_weights:
            x, attention_weights = self.conv(x, edge_index, return_attention_weights=True)
            x = self.activation(x)
            x = self.dropout(x)
            return x, attention_weights
        else:
            x = self.conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
            return x


class GraphSAGELayer(nn.Module):
    """GraphSAGE层"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 aggr: str = 'mean', dropout: float = 0.5, 
                 bias: bool = True, normalize: bool = True):
        """
        初始化GraphSAGE层
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            aggr: 聚合方法 ('mean', 'max', 'add')
            dropout: Dropout概率
            bias: 是否使用偏置
            normalize: 是否L2标准化
        """
        super(GraphSAGELayer, self).__init__()
        
        self.conv = SAGEConv(
            input_dim, output_dim, aggr=aggr, 
            bias=bias, normalize=normalize
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_index):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            输出特征 [num_nodes, output_dim]
        """
        x = self.conv(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class GINLayer(nn.Module):
    """图同构网络层"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dim: Optional[int] = None, dropout: float = 0.5,
                 eps: float = 0.0, train_eps: bool = False):
        """
        初始化GIN层
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            hidden_dim: 隐藏层维度（默认与output_dim相同）
            dropout: Dropout概率
            eps: epsilon参数
            train_eps: 是否训练epsilon
        """
        super(GINLayer, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = output_dim
        
        # 构建MLP
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.conv = GINConv(mlp, eps=eps, train_eps=train_eps)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_index):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            输出特征 [num_nodes, output_dim]
        """
        x = self.conv(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class BasicGNN(nn.Module):
    """基础GNN模型，支持多种GNN架构"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, gnn_type: str = 'gcn',
                 dropout: float = 0.5, **kwargs):
        """
        初始化基础GNN模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出特征维度
            gnn_type: GNN类型 ('gcn', 'gat', 'graphsage', 'gin')
            dropout: Dropout概率
            **kwargs: 其他参数
        """
        super(BasicGNN, self).__init__()
        
        self.gnn_type = gnn_type.lower()
        self.num_layers = len(hidden_dims)
        
        # 构建GNN层
        self.gnn_layers = nn.ModuleList()
        
        # 第一层
        first_layer = self._create_gnn_layer(
            input_dim, hidden_dims[0], dropout, **kwargs
        )
        self.gnn_layers.append(first_layer)
        
        # 中间层
        for i in range(1, self.num_layers):
            layer = self._create_gnn_layer(
                hidden_dims[i-1], hidden_dims[i], dropout, **kwargs
            )
            self.gnn_layers.append(layer)
        
        # 输出层
        if output_dim > 0:
            self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        else:
            self.output_layer = None
        
        # Batch Normalization (可选)
        self.use_bn = kwargs.get('use_bn', False)
        if self.use_bn:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(dim) for dim in hidden_dims
            ])
        
        print(f"🔗 创建 {gnn_type.upper()} 模型:")
        print(f"   输入维度: {input_dim}")
        print(f"   隐藏维度: {hidden_dims}")
        print(f"   输出维度: {output_dim}")
        print(f"   层数: {self.num_layers + (1 if output_dim > 0 else 0)}")
    
    def _create_gnn_layer(self, input_dim: int, output_dim: int, 
                         dropout: float, **kwargs):
        """创建GNN层"""
        if self.gnn_type == 'gcn':
            return GCNLayer(input_dim, output_dim, dropout)
        elif self.gnn_type == 'gat':
            heads = kwargs.get('heads', 8)
            concat = kwargs.get('concat', True)
            return GATLayer(input_dim, output_dim, heads, dropout, concat)
        elif self.gnn_type == 'graphsage':
            aggr = kwargs.get('aggr', 'mean')
            normalize = kwargs.get('normalize', True)
            return GraphSAGELayer(input_dim, output_dim, aggr, dropout, normalize=normalize)
        elif self.gnn_type == 'gin':
            hidden_dim = kwargs.get('gin_hidden_dim', output_dim)
            eps = kwargs.get('eps', 0.0)
            return GINLayer(input_dim, output_dim, hidden_dim, dropout, eps)
        else:
            raise ValueError(f"不支持的GNN类型: {self.gnn_type}")
    
    def forward(self, x, edge_index, batch=None, return_embeddings=False):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引 [num_nodes] (图级任务需要)
            return_embeddings: 是否返回节点嵌入
            
        Returns:
            输出特征或图级表示
        """
        # 通过GNN层
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.gnn_type == 'gat' and i == 0:
                # GAT可能返回注意力权重
                if hasattr(gnn_layer, 'return_attention_weights'):
                    x = gnn_layer(x, edge_index)
                else:
                    x = gnn_layer(x, edge_index)
            else:
                x = gnn_layer(x, edge_index)
            
            # Batch Normalization
            if self.use_bn and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
        
        # 保存节点嵌入
        node_embeddings = x.clone() if return_embeddings else None
        
        # 图级池化（如果有batch参数）
        if batch is not None:
            # 图级任务：池化得到图表示
            x = global_mean_pool(x, batch)  # 也可以使用global_max_pool
        
        # 输出层
        if self.output_layer is not None:
            x = self.output_layer(x)
        
        if return_embeddings:
            return x, node_embeddings
        else:
            return x
    
    def get_embeddings(self, x, edge_index, batch=None):
        """获取节点嵌入"""
        with torch.no_grad():
            _, embeddings = self.forward(x, edge_index, batch, return_embeddings=True)
            return embeddings


class GNNClassifier(nn.Module):
    """GNN分类器，专门用于谣言检测任务"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 num_classes: int = 3, gnn_type: str = 'gat',
                 dropout: float = 0.5, use_residual: bool = True, **kwargs):
        """
        初始化GNN分类器
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            num_classes: 分类类别数
            gnn_type: GNN类型
            dropout: Dropout概率
            use_residual: 是否使用残差连接
            **kwargs: 其他参数
        """
        super(GNNClassifier, self).__init__()
        
        self.use_residual = use_residual
        self.num_classes = num_classes
        
        # GNN主体
        self.gnn = BasicGNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=0,  # 不需要额外的输出层
            gnn_type=gnn_type,
            dropout=dropout,
            **kwargs
        )
        
        # 分类头
        classifier_input_dim = hidden_dims[-1]
        
        # 可选的特征融合层
        self.feature_fusion = kwargs.get('feature_fusion', False)
        if self.feature_fusion:
            self.fusion_layer = nn.Linear(classifier_input_dim * 2, classifier_input_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(classifier_input_dim // 2, num_classes)
        )
        
        # 残差连接（如果输入输出维度匹配）
        if use_residual and input_dim == hidden_dims[-1]:
            self.residual_projection = None
        elif use_residual:
            self.residual_projection = nn.Linear(input_dim, hidden_dims[-1])
        else:
            self.residual_projection = None
        
        print(f"🎯 创建GNN分类器:")
        print(f"   GNN类型: {gnn_type.upper()}")
        print(f"   输入维度: {input_dim}")
        print(f"   隐藏维度: {hidden_dims}")
        print(f"   分类数: {num_classes}")
        print(f"   残差连接: {use_residual}")
    
    def forward(self, x, edge_index, batch=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引 [num_nodes] (图级任务需要)
            
        Returns:
            分类logits [batch_size, num_classes] 或 [num_nodes, num_classes]
        """
        # 保存输入用于残差连接
        residual = x
        
        # 通过GNN
        x = self.gnn(x, edge_index, batch)
        
        # 残差连接
        if self.use_residual:
            if self.residual_projection is not None:
                residual = self.residual_projection(residual)
            
            # 如果是图级任务，需要对residual进行池化
            if batch is not None:
                residual = global_mean_pool(residual, batch)
            
            x = x + residual
        
        # 分类
        logits = self.classifier(x)
        
        return logits
    
    def predict(self, x, edge_index, batch=None):
        """预测"""
        with torch.no_grad():
            logits = self.forward(x, edge_index, batch)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            return predictions, probabilities
    
    def get_parameter_count(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GraphPooling(nn.Module):
    """图池化层"""
    
    def __init__(self, pooling_type: str = 'mean'):
        """
        初始化图池化层
        
        Args:
            pooling_type: 池化类型 ('mean', 'max', 'add', 'attention')
        """
        super(GraphPooling, self).__init__()
        self.pooling_type = pooling_type
        
        if pooling_type == 'attention':
            # 注意力池化需要额外的参数
            self.attention_weights = None
    
    def forward(self, x, batch):
        """
        图池化
        
        Args:
            x: 节点特征 [num_nodes, feature_dim]
            batch: 批次索引 [num_nodes]
            
        Returns:
            图级特征 [batch_size, feature_dim]
        """
        if self.pooling_type == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling_type == 'max':
            return global_max_pool(x, batch)
        elif self.pooling_type == 'add':
            return torch_geometric.nn.global_add_pool(x, batch)
        elif self.pooling_type == 'attention':
            # 简单的注意力池化实现
            if self.attention_weights is None:
                self.attention_weights = nn.Linear(x.size(-1), 1)
            
            attention_scores = self.attention_weights(x)
            attention_weights = F.softmax(attention_scores, dim=0)
            
            # 按batch加权求和
            graph_embeddings = []
            for i in range(batch.max().item() + 1):
                mask = batch == i
                if mask.sum() > 0:
                    graph_x = x[mask]
                    graph_weights = attention_weights[mask]
                    graph_emb = (graph_x * graph_weights).sum(dim=0)
                    graph_embeddings.append(graph_emb)
            
            return torch.stack(graph_embeddings)
        else:
            raise ValueError(f"不支持的池化类型: {self.pooling_type}")


def create_gnn_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建GNN模型
    
    Args:
        model_config: 模型配置字典
        
    Returns:
        GNN模型实例
    """
    model_type = model_config.get('model_type', 'classifier')
    gnn_type = model_config.get('gnn_type', 'gat')
    
    if model_type == 'classifier':
        model = GNNClassifier(
            input_dim=model_config.get('input_dim', 768),
            hidden_dims=model_config.get('hidden_dims', [128, 64]),
            num_classes=model_config.get('num_classes', 3),
            gnn_type=gnn_type,
            dropout=model_config.get('dropout', 0.5),
            use_residual=model_config.get('use_residual', True),
            **model_config.get('gnn_params', {})
        )
    elif model_type == 'basic':
        model = BasicGNN(
            input_dim=model_config.get('input_dim', 768),
            hidden_dims=model_config.get('hidden_dims', [128, 64]),
            output_dim=model_config.get('output_dim', 3),
            gnn_type=gnn_type,
            dropout=model_config.get('dropout', 0.5),
            **model_config.get('gnn_params', {})
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model


# 使用示例和测试代码
if __name__ == "__main__":
    print("🔗 测试基础图神经网络层")
    
    # 创建示例数据
    num_nodes = 100
    num_edges = 200
    input_dim = 768
    num_classes = 3
    
    # 节点特征
    x = torch.randn(num_nodes, input_dim)
    
    # 边索引
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # 批次索引（图级任务）
    batch = torch.zeros(num_nodes, dtype=torch.long)  # 单图
    
    print(f"📊 测试数据:")
    print(f"   节点数: {num_nodes}")
    print(f"   边数: {num_edges}")
    print(f"   特征维度: {input_dim}")
    print(f"   节点特征形状: {x.shape}")
    print(f"   边索引形状: {edge_index.shape}")
    
    # 测试各种GNN层
    gnn_types = ['gcn', 'gat', 'graphsage', 'gin']
    
    for gnn_type in gnn_types:
        print(f"\n🧪 测试 {gnn_type.upper()} 模型:")
        
        try:
            # 创建分类模型
            model = GNNClassifier(
                input_dim=input_dim,
                hidden_dims=[128, 64],
                num_classes=num_classes,
                gnn_type=gnn_type,
                dropout=0.5
            )
            
            # 前向传播
            with torch.no_grad():
                logits = model(x, edge_index, batch)
                predictions, probabilities = model.predict(x, edge_index, batch)
                
                print(f"   ✅ 输出形状: {logits.shape}")
                print(f"   ✅ 预测形状: {predictions.shape}")
                print(f"   ✅ 参数数量: {model.get_parameter_count():,}")
                
                # 节点级任务测试
                node_logits = model(x, edge_index)  # 不提供batch
                print(f"   ✅ 节点级输出形状: {node_logits.shape}")
                
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
    
    # 测试模型配置创建
    print(f"\n🔧 测试模型配置创建:")
    model_config = {
        'model_type': 'classifier',
        'gnn_type': 'gat',
        'input_dim': input_dim,
        'hidden_dims': [256, 128, 64],
        'num_classes': num_classes,
        'dropout': 0.3,
        'use_residual': True,
        'gnn_params': {
            'heads': 4,
            'concat': True,
            'use_bn': True
        }
    }
    
    try:
        model = create_gnn_model(model_config)
        print(f"   ✅ 配置模型创建成功")
        print(f"   ✅ 参数数量: {model.get_parameter_count():,}")
    except Exception as e:
        print(f"   ❌ 配置模型创建失败: {e}")
    
    print("\n✅ 基础图神经网络层测试完成")