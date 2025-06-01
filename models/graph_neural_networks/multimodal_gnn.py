#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/graph_neural_networks/multimodal_gnn.py

"""
多模态图神经网络模块
结合文本、图像、图结构的联合建模
专门为MR2数据集和谣言检测任务设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
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
    from .basic_gnn_layers import BasicGNN, GNNClassifier
    from .social_graph_builder import SocialGraphBuilder, GraphFeatureExtractor
    from utils.config_manager import get_config_manager, get_output_path
    from data_utils.data_loaders import create_all_dataloaders
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块 (相对导入)")
except ImportError as e:
    print(f"⚠️  相对导入失败: {e}")
    try:
        from basic_gnn_layers import BasicGNN, GNNClassifier
        from social_graph_builder import SocialGraphBuilder, GraphFeatureExtractor
        # 尝试导入项目配置
        try:
            from utils.config_manager import get_config_manager, get_output_path
            from data_utils.data_loaders import create_all_dataloaders
            USE_PROJECT_MODULES = True
            print("✅ 成功导入本地模块 (包含项目配置)")
        except ImportError:
            USE_PROJECT_MODULES = False
            print("✅ 成功导入本地模块 (不含项目配置)")
    except ImportError as e2:
        print(f"❌ 导入模块失败: {e2}")
        print("创建简化版本...")
        USE_PROJECT_MODULES = False
        
        # 创建简化的类定义用于独立运行
        class BasicGNN(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim, gnn_type='simple', **kwargs):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dims[0]),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_dims[0], output_dim)
                )
            
            def forward(self, x, edge_index, batch=None):
                if batch is not None:
                    # 图级任务：取平均
                    x = torch.mean(x, dim=0, keepdim=True)
                    return self.layers(x)
                else:
                    # 节点级任务
                    return self.layers(torch.mean(x, dim=0, keepdim=True).repeat(x.size(0), 1))
        
        class GNNClassifier(nn.Module):
            def __init__(self, input_dim, hidden_dims, num_classes, **kwargs):
                super().__init__()
                self.gnn = BasicGNN(input_dim, hidden_dims, hidden_dims[-1])
                self.classifier = nn.Linear(hidden_dims[-1], num_classes)
                
            def forward(self, x, edge_index, batch=None):
                x = self.gnn(x, edge_index, batch)
                return self.classifier(x)
        
        class SocialGraphBuilder:
            def __init__(self, *args, **kwargs):
                print("⚠️  使用简化版社交图构建器")
                
        class GraphFeatureExtractor:
            def __init__(self, *args, **kwargs):
                print("⚠️  使用简化版图特征提取器")

import logging
logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """文本编码器"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 output_dim: int = 768, freeze_bert: bool = False):
        """
        初始化文本编码器
        
        Args:
            model_name: 预训练模型名称
            output_dim: 输出维度
            freeze_bert: 是否冻结BERT参数
        """
        super(TextEncoder, self).__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        try:
            # 尝试加载预训练模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
            
            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
            
            # 投影层
            bert_dim = self.bert.config.hidden_size
            if bert_dim != output_dim:
                self.projection = nn.Linear(bert_dim, output_dim)
            else:
                self.projection = nn.Identity()
                
            self.use_pretrained = True
            print(f"✅ 加载预训练文本编码器: {model_name}")
            
        except Exception as e:
            print(f"⚠️  加载预训练模型失败: {e}")
            print("使用简单的文本编码器")
            
            # 简单的文本编码器
            self.vocab_size = 10000
            self.embedding_dim = 256
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(self.embedding_dim, output_dim // 2, 
                              batch_first=True, bidirectional=True)
            self.projection = nn.Identity()
            self.use_pretrained = False
    
    def forward(self, text_inputs):
        """
        前向传播
        
        Args:
            text_inputs: 文本输入（字符串列表或token ids）
            
        Returns:
            文本特征 [batch_size, output_dim]
        """
        if self.use_pretrained:
            if isinstance(text_inputs, list):
                # 字符串列表
                encoded = self.tokenizer(
                    text_inputs, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,
                    return_tensors='pt'
                )
                
                if next(self.bert.parameters()).is_cuda:
                    encoded = {k: v.cuda() for k, v in encoded.items()}
                
                with torch.no_grad() if not self.training else torch.enable_grad():
                    outputs = self.bert(**encoded)
                
                # 使用[CLS] token的表示
                text_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, bert_dim]
            else:
                # 已经是tensor
                if len(text_inputs.shape) == 2:
                    # [batch_size, seq_len]
                    attention_mask = (text_inputs != 0).long()
                    outputs = self.bert(input_ids=text_inputs, attention_mask=attention_mask)
                    text_features = outputs.last_hidden_state[:, 0, :]
                else:
                    raise ValueError(f"不支持的输入格式: {text_inputs.shape}")
            
            # 投影到目标维度
            text_features = self.projection(text_features)
            
        else:
            # 简单编码器
            if isinstance(text_inputs, list):
                # 简单的词汇表映射
                max_len = 128
                batch_size = len(text_inputs)
                input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
                
                for i, text in enumerate(text_inputs):
                    tokens = text.lower().split()[:max_len]
                    for j, token in enumerate(tokens):
                        # 简单的hash映射到词汇表
                        token_id = hash(token) % (self.vocab_size - 1) + 1
                        input_ids[i, j] = token_id
                
                if next(self.embedding.parameters()).is_cuda:
                    input_ids = input_ids.cuda()
                
                text_inputs = input_ids
            
            # 嵌入和LSTM
            embedded = self.embedding(text_inputs)  # [batch_size, seq_len, embedding_dim]
            lstm_out, (hidden, _) = self.lstm(embedded)
            
            # 使用最后一个隐藏状态
            text_features = torch.cat([hidden[0], hidden[1]], dim=1)  # [batch_size, output_dim]
        
        return text_features


class ImageEncoder(nn.Module):
    """图像编码器"""
    
    def __init__(self, model_name: str = 'resnet50', output_dim: int = 768, 
                 pretrained: bool = True, freeze_backbone: bool = False):
        """
        初始化图像编码器
        
        Args:
            model_name: 模型名称
            output_dim: 输出维度
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结主干网络
        """
        super(ImageEncoder, self).__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        try:
            # 加载预训练模型
            if model_name == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
                backbone_dim = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()  # 移除分类头
            elif model_name == 'resnet18':
                self.backbone = models.resnet18(pretrained=pretrained)
                backbone_dim = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            elif model_name == 'vgg16':
                self.backbone = models.vgg16(pretrained=pretrained)
                backbone_dim = self.backbone.classifier[0].in_features
                self.backbone.classifier = nn.Identity()
            else:
                raise ValueError(f"不支持的模型: {model_name}")
            
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            # 投影层
            if backbone_dim != output_dim:
                self.projection = nn.Sequential(
                    nn.Linear(backbone_dim, output_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            else:
                self.projection = nn.Identity()
                
            self.use_pretrained = True
            print(f"✅ 加载预训练图像编码器: {model_name}")
            
        except Exception as e:
            print(f"⚠️  加载预训练模型失败: {e}")
            print("使用简单的图像编码器")
            
            # 简单的CNN编码器
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.projection = nn.Linear(256, output_dim)
            self.use_pretrained = False
    
    def forward(self, image_inputs):
        """
        前向传播
        
        Args:
            image_inputs: 图像输入 [batch_size, 3, H, W]
            
        Returns:
            图像特征 [batch_size, output_dim]
        """
        if image_inputs.dim() != 4:
            raise ValueError(f"期望4D输入，得到{image_inputs.dim()}D")
        
        # 通过主干网络
        image_features = self.backbone(image_inputs)
        
        # 投影到目标维度
        image_features = self.projection(image_features)
        
        return image_features


class CrossModalAttention(nn.Module):
    """跨模态注意力模块"""
    
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int = 256):
        """
        初始化跨模态注意力
        
        Args:
            text_dim: 文本特征维度
            image_dim: 图像特征维度
            hidden_dim: 隐藏层维度
        """
        super(CrossModalAttention, self).__init__()
        
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        self.text_output = nn.Linear(hidden_dim, text_dim)
        self.image_output = nn.Linear(hidden_dim, image_dim)
        
        self.layer_norm_text = nn.LayerNorm(text_dim)
        self.layer_norm_image = nn.LayerNorm(image_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, text_features, image_features):
        """
        跨模态注意力计算
        
        Args:
            text_features: 文本特征 [batch_size, text_dim]
            image_features: 图像特征 [batch_size, image_dim]
            
        Returns:
            融合后的文本和图像特征
        """
        batch_size = text_features.size(0)
        
        # 投影到相同维度
        text_proj = self.text_proj(text_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        image_proj = self.image_proj(image_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 文本-图像注意力
        text_attended, _ = self.attention(text_proj, image_proj, image_proj)
        text_attended = text_attended.squeeze(1)  # [batch_size, hidden_dim]
        
        # 图像-文本注意力
        image_attended, _ = self.attention(image_proj, text_proj, text_proj)
        image_attended = image_attended.squeeze(1)  # [batch_size, hidden_dim]
        
        # 输出投影
        text_output = self.text_output(self.dropout(text_attended))
        image_output = self.image_output(self.dropout(image_attended))
        
        # 残差连接和层标准化
        text_fused = self.layer_norm_text(text_features + text_output)
        image_fused = self.layer_norm_image(image_features + image_output)
        
        return text_fused, image_fused


class MultimodalFusion(nn.Module):
    """多模态融合模块"""
    
    def __init__(self, text_dim: int, image_dim: int, graph_dim: int,
                 fusion_method: str = 'attention', output_dim: int = 512):
        """
        初始化多模态融合
        
        Args:
            text_dim: 文本特征维度
            image_dim: 图像特征维度
            graph_dim: 图特征维度
            fusion_method: 融合方法 ('concat', 'attention', 'gate', 'cross_modal')
            output_dim: 输出维度
        """
        super(MultimodalFusion, self).__init__()
        
        self.fusion_method = fusion_method
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        
        if fusion_method == 'concat':
            # 简单拼接
            total_dim = text_dim + image_dim + graph_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
        elif fusion_method == 'attention':
            # 注意力融合
            self.text_proj = nn.Linear(text_dim, output_dim)
            self.image_proj = nn.Linear(image_dim, output_dim)
            self.graph_proj = nn.Linear(graph_dim, output_dim)
            
            self.attention_weights = nn.Linear(output_dim, 1)
            self.layer_norm = nn.LayerNorm(output_dim)
            
        elif fusion_method == 'gate':
            # 门控融合
            self.text_gate = nn.Sequential(
                nn.Linear(text_dim, output_dim),
                nn.Sigmoid()
            )
            self.image_gate = nn.Sequential(
                nn.Linear(image_dim, output_dim),
                nn.Sigmoid()
            )
            self.graph_gate = nn.Sequential(
                nn.Linear(graph_dim, output_dim),
                nn.Sigmoid()
            )
            
            self.text_proj = nn.Linear(text_dim, output_dim)
            self.image_proj = nn.Linear(image_dim, output_dim)
            self.graph_proj = nn.Linear(graph_dim, output_dim)
            
        elif fusion_method == 'cross_modal':
            # 跨模态注意力融合
            self.cross_attention = CrossModalAttention(text_dim, image_dim)
            self.graph_proj = nn.Linear(graph_dim, output_dim)
            
            fusion_dim = text_dim + image_dim + output_dim
            self.final_fusion = nn.Sequential(
                nn.Linear(fusion_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")
        
        print(f"🔀 多模态融合初始化: {fusion_method}")
    
    def forward(self, text_features, image_features, graph_features):
        """
        多模态融合
        
        Args:
            text_features: 文本特征 [batch_size, text_dim]
            image_features: 图像特征 [batch_size, image_dim]
            graph_features: 图特征 [batch_size, graph_dim]
            
        Returns:
            融合特征 [batch_size, output_dim]
        """
        if self.fusion_method == 'concat':
            # 拼接融合
            fused = torch.cat([text_features, image_features, graph_features], dim=1)
            fused = self.fusion_layer(fused)
            
        elif self.fusion_method == 'attention':
            # 注意力融合
            text_proj = self.text_proj(text_features)
            image_proj = self.image_proj(image_features)
            graph_proj = self.graph_proj(graph_features)
            
            # 堆叠特征
            features = torch.stack([text_proj, image_proj, graph_proj], dim=1)  # [batch_size, 3, output_dim]
            
            # 计算注意力权重
            attention_scores = self.attention_weights(features).squeeze(-1)  # [batch_size, 3]
            attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 3]
            
            # 加权求和
            fused = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, output_dim]
            fused = self.layer_norm(fused)
            
        elif self.fusion_method == 'gate':
            # 门控融合
            text_gate = self.text_gate(text_features)
            image_gate = self.image_gate(image_features)
            graph_gate = self.graph_gate(graph_features)
            
            text_proj = self.text_proj(text_features)
            image_proj = self.image_proj(image_features)
            graph_proj = self.graph_proj(graph_features)
            
            # 门控机制
            gated_text = text_gate * text_proj
            gated_image = image_gate * image_proj
            gated_graph = graph_gate * graph_proj
            
            fused = gated_text + gated_image + gated_graph
            
        elif self.fusion_method == 'cross_modal':
            # 跨模态注意力融合
            text_fused, image_fused = self.cross_attention(text_features, image_features)
            graph_proj = self.graph_proj(graph_features)
            
            # 最终融合
            all_features = torch.cat([text_fused, image_fused, graph_proj], dim=1)
            fused = self.final_fusion(all_features)
        
        return fused


class MultimodalGNN(nn.Module):
    """多模态图神经网络"""
    
    def __init__(self, text_encoder_config: Dict[str, Any],
                 image_encoder_config: Dict[str, Any],
                 gnn_config: Dict[str, Any],
                 fusion_config: Dict[str, Any],
                 num_classes: int = 3):
        """
        初始化多模态GNN
        
        Args:
            text_encoder_config: 文本编码器配置
            image_encoder_config: 图像编码器配置
            gnn_config: GNN配置
            fusion_config: 融合配置
            num_classes: 分类数量
        """
        super(MultimodalGNN, self).__init__()
        
        self.num_classes = num_classes
        
        # 检查是否有真正的GNN可用
        self._has_real_gnn = USE_PROJECT_MODULES and hasattr(BasicGNN, '__module__')
        
        # 文本编码器
        self.text_encoder = TextEncoder(
            model_name=text_encoder_config.get('model_name', 'bert-base-uncased'),
            output_dim=text_encoder_config.get('output_dim', 768),
            freeze_bert=text_encoder_config.get('freeze_bert', False)
        )
        
        # 图像编码器
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_config.get('model_name', 'resnet50'),
            output_dim=image_encoder_config.get('output_dim', 768),
            pretrained=image_encoder_config.get('pretrained', True),
            freeze_backbone=image_encoder_config.get('freeze_backbone', False)
        )
        
        # 图神经网络
        if USE_PROJECT_MODULES and hasattr(self, '_has_real_gnn'):
            self.gnn = BasicGNN(
                input_dim=gnn_config.get('input_dim', 768),
                hidden_dims=gnn_config.get('hidden_dims', [256, 128]),
                output_dim=gnn_config.get('output_dim', 128),
                gnn_type=gnn_config.get('gnn_type', 'gat'),
                dropout=gnn_config.get('dropout', 0.5),
                **gnn_config.get('gnn_params', {})
            )
        else:
            # 简单的图网络实现
            input_dim = gnn_config.get('input_dim', 768)
            output_dim = gnn_config.get('output_dim', 128)
            self.gnn = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, output_dim)
            )
            print("⚠️  使用简化版GNN实现")
        
        # 多模态融合
        self.fusion = MultimodalFusion(
            text_dim=text_encoder_config.get('output_dim', 768),
            image_dim=image_encoder_config.get('output_dim', 768),
            graph_dim=gnn_config.get('output_dim', 128),
            fusion_method=fusion_config.get('method', 'attention'),
            output_dim=fusion_config.get('output_dim', 512)
        )
        
        # 分类器
        classifier_input_dim = fusion_config.get('output_dim', 512)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_input_dim // 2, num_classes)
        )
        
        print(f"🤖 多模态GNN初始化完成:")
        print(f"   文本编码器: {text_encoder_config.get('model_name', 'simple')}")
        print(f"   图像编码器: {image_encoder_config.get('model_name', 'simple')}")
        print(f"   GNN类型: {gnn_config.get('gnn_type', 'simple')}")
        print(f"   融合方法: {fusion_config.get('method', 'attention')}")
        print(f"   分类数: {num_classes}")
    
    def forward(self, text_inputs, image_inputs, graph_data, return_features=False):
        """
        前向传播
        
        Args:
            text_inputs: 文本输入
            image_inputs: 图像输入 [batch_size, 3, H, W]
            graph_data: 图数据 (PyG Data对象)
            return_features: 是否返回中间特征
            
        Returns:
            分类logits或特征字典
        """
        # 文本编码
        text_features = self.text_encoder(text_inputs)  # [batch_size, text_dim]
        
        # 图像编码
        image_features = self.image_encoder(image_inputs)  # [batch_size, image_dim]
        
        # 图编码
        if self._has_real_gnn and hasattr(self.gnn, 'forward'):
            # 使用真实的GNN
            if hasattr(graph_data, 'batch'):
                graph_features = self.gnn(graph_data.x, graph_data.edge_index, graph_data.batch)
            else:
                # 单图情况，创建batch
                batch_size = text_features.size(0)
                num_nodes = graph_data.x.size(0)
                batch = torch.zeros(num_nodes, dtype=torch.long, device=graph_data.x.device)
                graph_features = self.gnn(graph_data.x, graph_data.edge_index, batch)
                
                # 确保图特征的batch维度与文本、图像一致
                if graph_features.size(0) != batch_size:
                    # 如果是单个图特征，重复以匹配batch_size
                    if graph_features.size(0) == 1:
                        graph_features = graph_features.repeat(batch_size, 1)
                    else:
                        # 如果是多节点输出，取平均作为图表示
                        graph_features = torch.mean(graph_features, dim=0, keepdim=True)
                        graph_features = graph_features.repeat(batch_size, 1)
        else:
            # 使用简单的图特征
            batch_size = text_features.size(0)
            if hasattr(graph_data, 'x'):
                # 取图节点特征的平均值作为图表示
                graph_repr = torch.mean(graph_data.x, dim=0, keepdim=True)  # [1, input_dim]
                graph_features = self.gnn(graph_repr)  # [1, output_dim]
                
                # 重复以匹配batch_size
                graph_features = graph_features.repeat(batch_size, 1)
            else:
                # 创建零图特征
                graph_dim = 128  # 默认图特征维度
                graph_features = torch.zeros(batch_size, graph_dim, device=text_features.device)
        
        # 确保所有特征的batch维度一致
        batch_size = text_features.size(0)
        if image_features.size(0) != batch_size:
            image_features = image_features[:batch_size] if image_features.size(0) > batch_size else image_features.repeat(batch_size, 1)
        if graph_features.size(0) != batch_size:
            graph_features = graph_features[:batch_size] if graph_features.size(0) > batch_size else graph_features.repeat(batch_size, 1)
        
        # 多模态融合
        fused_features = self.fusion(text_features, image_features, graph_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        if return_features:
            return {
                'logits': logits,
                'text_features': text_features,
                'image_features': image_features,
                'graph_features': graph_features,
                'fused_features': fused_features
            }
        else:
            return logits
    
    def predict(self, text_inputs, image_inputs, graph_data):
        """预测"""
        with torch.no_grad():
            logits = self.forward(text_inputs, image_inputs, graph_data)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            return predictions, probabilities
    
    def get_parameter_count(self):
        """获取模型参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


class MultimodalGraphClassifier:
    """多模态图分类器训练器"""
    
    def __init__(self, model_config: Dict[str, Any], device: str = 'auto'):
        """
        初始化多模态图分类器
        
        Args:
            model_config: 模型配置
            device: 计算设备
        """
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  使用设备: {self.device}")
        
        # 创建模型
        self.model = MultimodalGNN(
            text_encoder_config=model_config.get('text_encoder', {}),
            image_encoder_config=model_config.get('image_encoder', {}),
            gnn_config=model_config.get('gnn', {}),
            fusion_config=model_config.get('fusion', {}),
            num_classes=model_config.get('num_classes', 3)
        ).to(self.device)
        
        # 初始化图构建器
        if USE_PROJECT_MODULES:
            self.graph_builder = SocialGraphBuilder()
        else:
            self.graph_builder = None
        
        # 标签映射
        self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
        
        # 训练历史
        self.training_history = {
            'train_losses': [],
            'val_accuracies': [],
            'val_f1_scores': []
        }
        
        param_info = self.model.get_parameter_count()
        print(f"🤖 多模态图分类器初始化完成:")
        print(f"   总参数: {param_info['total']:,}")
        print(f"   可训练参数: {param_info['trainable']:,}")
        print(f"   冻结参数: {param_info['frozen']:,}")
    
    def create_demo_data(self, batch_size: int = 8):
        """创建演示数据 - 修复维度问题"""
        print("🔧 创建演示数据...")
        
        # 文本数据
        demo_texts = [
            "这是一个关于科技的真实新闻",
            "This is fake news about celebrities",
            "未经证实的传言需要验证",
            "Breaking news from reliable sources",
            "网络谣言传播速度很快",
            "Scientific research shows evidence",
            "官方辟谣声明已发布",
            "Verified information from experts"
        ]
        
        texts = demo_texts[:batch_size]
        
        # 图像数据（确保batch维度正确）
        images = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        # 图数据 - 确保能产生正确的图特征维度
        num_nodes = batch_size  # 让节点数等于batch_size，避免维度问题
        node_features = torch.randn(num_nodes, 768).to(self.device)
        
        # 创建简单的环形连接
        edge_index = []
        for i in range(num_nodes):
            edge_index.append([i, (i + 1) % num_nodes])
            edge_index.append([(i + 1) % num_nodes, i])  # 双向边
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        
        from torch_geometric.data import Data
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=num_nodes
        )
        
        # 标签
        labels = torch.randint(0, 3, (batch_size,)).to(self.device)
        
        print(f"✅ 演示数据创建完成:")
        print(f"   文本数量: {len(texts)}")
        print(f"   图像形状: {images.shape}")
        print(f"   图节点数: {num_nodes}")
        print(f"   标签数量: {labels.shape[0]}")
        
        return texts, images, graph_data, labels
    
    def train_step(self, texts, images, graph_data, labels, optimizer, criterion):
        """单步训练"""
        self.model.train()
        
        # 前向传播
        logits = self.model(texts, images, graph_data)
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean()
        
        return loss.item(), accuracy.item()
    
    def evaluate_step(self, texts, images, graph_data, labels, criterion):
        """单步评估"""
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(texts, images, graph_data)
            loss = criterion(logits, labels)
            
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean()
            
            # 计算F1分数（简化版本）
            from sklearn.metrics import f1_score
            f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')
        
        return loss.item(), accuracy.item(), f1
    
    def train_demo(self, epochs: int = 10, learning_rate: float = 1e-4):
        """演示训练"""
        print(f"🚀 开始多模态GNN演示训练...")
        
        # 创建演示数据
        train_texts, train_images, train_graph, train_labels = self.create_demo_data(8)
        val_texts, val_images, val_graph, val_labels = self.create_demo_data(4)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print(f"📊 训练数据: {len(train_texts)} 样本")
        print(f"📊 验证数据: {len(val_texts)} 样本")
        
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_step(
                train_texts, train_images, train_graph, train_labels,
                optimizer, criterion
            )
            
            # 验证
            val_loss, val_acc, val_f1 = self.evaluate_step(
                val_texts, val_images, val_graph, val_labels, criterion
            )
            
            # 记录历史
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_accuracies'].append(val_acc)
            self.training_history['val_f1_scores'].append(val_f1)
            
            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}")
        
        print(f"\n✅ 训练完成!")
        print(f"   最佳验证F1: {best_val_f1:.4f} (Epoch {best_epoch+1})")
        
        return self.training_history
    
    def test_inference(self):
        """测试推理"""
        print(f"\n🧪 测试模型推理...")
        
        # 创建测试数据
        test_texts, test_images, test_graph, test_labels = self.create_demo_data(4)
        
        # 推理
        predictions, probabilities = self.model.predict(test_texts, test_images, test_graph)
        
        print(f"📊 推理结果:")
        for i in range(len(test_texts)):
            pred = predictions[i].item()
            prob = probabilities[i].max().item()
            true_label = test_labels[i].item()
            
            print(f"   样本 {i+1}: 预测={self.label_mapping[pred]} (置信度: {prob:.3f}), "
                  f"真实={self.label_mapping[true_label]}")
        
        # 测试特征提取
        print(f"\n🔍 测试特征提取...")
        features = self.model.forward(test_texts, test_images, test_graph, return_features=True)
        
        for feature_name, feature_tensor in features.items():
            if isinstance(feature_tensor, torch.Tensor):
                print(f"   {feature_name}: {feature_tensor.shape}")


def create_multimodal_gnn_config(gnn_type: str = 'gat', 
                                fusion_method: str = 'attention') -> Dict[str, Any]:
    """
    创建多模态GNN配置
    
    Args:
        gnn_type: GNN类型
        fusion_method: 融合方法
        
    Returns:
        配置字典
    """
    config = {
        'text_encoder': {
            'model_name': 'bert-base-uncased',
            'output_dim': 768,
            'freeze_bert': False
        },
        'image_encoder': {
            'model_name': 'resnet50',
            'output_dim': 768,
            'pretrained': True,
            'freeze_backbone': False
        },
        'gnn': {
            'input_dim': 768,
            'hidden_dims': [256, 128],
            'output_dim': 128,
            'gnn_type': gnn_type,
            'dropout': 0.5,
            'gnn_params': {
                'heads': 8 if gnn_type == 'gat' else None,
                'concat': True if gnn_type == 'gat' else None,
                'use_bn': True
            }
        },
        'fusion': {
            'method': fusion_method,
            'output_dim': 512
        },
        'num_classes': 3
    }
    
    return config


if __name__ == "__main__":
    print("🤖 测试多模态图神经网络")
    
    # 只测试一个配置，确保能运行
    test_config = {'gnn_type': 'gat', 'fusion_method': 'attention'}
    
    print(f"\n{'='*60}")
    print(f"测试配置: GNN={test_config['gnn_type'].upper()}, "
          f"Fusion={test_config['fusion_method']}")
    print(f"{'='*60}")
    
    try:
        # 创建模型配置
        model_config = create_multimodal_gnn_config(
            gnn_type=test_config['gnn_type'],
            fusion_method=test_config['fusion_method']
        )
        
        # 创建分类器
        classifier = MultimodalGraphClassifier(model_config)
        
        # 演示训练（减少epochs）
        history = classifier.train_demo(epochs=3, learning_rate=1e-4)
        
        # 测试推理
        classifier.test_inference()
        
        print(f"✅ 测试成功完成!")
        print(f"   最终训练损失: {history['train_losses'][-1]:.4f}")
        print(f"   最终验证F1: {history['val_f1_scores'][-1]:.4f}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 多模态图神经网络测试完成")