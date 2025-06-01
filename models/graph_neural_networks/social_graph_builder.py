#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/graph_neural_networks/social_graph_builder.py

"""
社交网络图构建模块
从MR2数据集构建社交图，提取图特征，支持多种图结构
专门为谣言检测任务设计
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import sys
import json
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入项目模块
try:
    from datasets.data_loaders import create_all_dataloaders
    from utils.config_manager import get_config_manager, get_data_dir, get_output_path
    from preprocessing.text_processing import TextProcessor
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块")
except ImportError as e:
    print(f"⚠️  导入项目模块失败: {e}")
    USE_PROJECT_MODULES = False

import logging
logger = logging.getLogger(__name__)


class SocialGraphBuilder:
    """社交网络图构建器"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化社交图构建器
        
        Args:
            data_dir: 数据目录路径
        """
        if USE_PROJECT_MODULES:
            self.data_dir = get_data_dir() if data_dir is None else Path(data_dir)
            self.output_dir = get_output_path('graphs', 'social_networks')
        else:
            self.data_dir = Path(data_dir) if data_dir else Path('data')
            self.output_dir = Path('outputs/graphs/social_networks')
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化文本处理器
        if USE_PROJECT_MODULES:
            self.text_processor = TextProcessor(language='mixed')
        else:
            self.text_processor = None
        
        # 图构建参数
        self.min_similarity = 0.1  # 最小相似度阈值
        self.max_edges_per_node = 10  # 每个节点最大边数
        
        # 缓存
        self.node_features_cache = {}
        self.edge_cache = {}
        
        print(f"🔗 社交图构建器初始化完成")
        print(f"   数据目录: {self.data_dir}")
        print(f"   输出目录: {self.output_dir}")
    
    def load_mr2_data(self, split: str = 'train') -> Dict[str, Any]:
        """
        加载MR2数据集
        
        Args:
            split: 数据划分 ('train', 'val', 'test')
            
        Returns:
            数据字典
        """
        dataset_file = self.data_dir / f'dataset_items_{split}.json'
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {dataset_file}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📚 加载 {split} 数据: {len(data)} 条记录")
        return data
    
    def extract_entities_from_annotations(self, item_data: Dict[str, Any]) -> List[str]:
        """
        从标注文件中提取实体
        
        Args:
            item_data: 数据项
            
        Returns:
            实体列表
        """
        entities = set()
        
        # 从inverse search标注提取实体
        if 'inv_path' in item_data:
            inv_annotation_file = self.data_dir / item_data['inv_path'] / 'inverse_annotation.json'
            if inv_annotation_file.exists():
                try:
                    with open(inv_annotation_file, 'r', encoding='utf-8') as f:
                        inv_data = json.load(f)
                    
                    # 提取实体
                    if 'entities' in inv_data:
                        entities.update(inv_data['entities'])
                    
                    # 从best_guess_lbl提取
                    if 'best_guess_lbl' in inv_data:
                        entities.update(inv_data['best_guess_lbl'])
                        
                except Exception as e:
                    logger.warning(f"读取inverse annotation失败: {e}")
        
        # 从direct search标注提取实体
        if 'direct_path' in item_data:
            direct_annotation_file = self.data_dir / item_data['direct_path'] / 'direct_annotation.json'
            if direct_annotation_file.exists():
                try:
                    with open(direct_annotation_file, 'r', encoding='utf-8') as f:
                        direct_data = json.load(f)
                    
                    # 从图像标注中提取实体
                    for img_data in direct_data.get('images_with_captions', []):
                        if 'caption' in img_data:
                            caption_info = img_data['caption']
                            if isinstance(caption_info, dict):
                                for key, value in caption_info.items():
                                    if isinstance(value, str):
                                        # 简单的实体提取（可以改进）
                                        words = value.split()
                                        entities.update(word for word in words if len(word) > 2)
                        
                        # 从域名提取
                        if 'domain' in img_data:
                            entities.add(img_data['domain'])
                            
                except Exception as e:
                    logger.warning(f"读取direct annotation失败: {e}")
        
        return list(entities)
    
    def build_text_similarity_graph(self, data: Dict[str, Any], 
                                  similarity_threshold: float = 0.3) -> Data:
        """
        基于文本相似度构建图
        
        Args:
            data: MR2数据
            similarity_threshold: 相似度阈值
            
        Returns:
            PyG Data对象
        """
        print("🔤 构建文本相似度图...")
        
        # 提取文本和标签
        texts = []
        labels = []
        item_ids = []
        
        for item_id, item_data in data.items():
            if 'caption' in item_data:
                texts.append(item_data['caption'])
                labels.append(item_data.get('label', 0))
                item_ids.append(item_id)
        
        if len(texts) == 0:
            raise ValueError("没有找到文本数据")
        
        # 计算文本特征（简化版本）
        node_features = self._compute_text_features(texts)
        
        # 计算相似度矩阵
        similarity_matrix = self._compute_cosine_similarity(node_features)
        
        # 构建边
        edge_index, edge_weights = self._build_edges_from_similarity(
            similarity_matrix, similarity_threshold
        )
        
        # 创建图数据
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weights,
            y=torch.tensor(labels, dtype=torch.long),
            num_nodes=len(texts)
        )
        
        # 添加额外信息
        graph_data.item_ids = item_ids
        graph_data.texts = texts
        
        print(f"✅ 文本相似度图构建完成:")
        print(f"   节点数: {graph_data.num_nodes}")
        print(f"   边数: {graph_data.edge_index.size(1)}")
        print(f"   特征维度: {graph_data.x.size(1)}")
        
        return graph_data
    
    def build_entity_cooccurrence_graph(self, data: Dict[str, Any]) -> Data:
        """
        基于实体共现构建图
        
        Args:
            data: MR2数据
            
        Returns:
            PyG Data对象
        """
        print("🏷️  构建实体共现图...")
        
        # 提取每个项目的实体
        item_entities = {}
        all_entities = set()
        labels = []
        item_ids = []
        
        for item_id, item_data in data.items():
            entities = self.extract_entities_from_annotations(item_data)
            
            # 从文本中提取更多实体
            if 'caption' in item_data and self.text_processor:
                text_tokens = self.text_processor.tokenize(item_data['caption'])
                # 简单的实体识别：长度大于2的词
                text_entities = [token for token in text_tokens if len(token) > 2]
                entities.extend(text_entities)
            
            if entities:
                item_entities[item_id] = list(set(entities))
                all_entities.update(entities)
                labels.append(item_data.get('label', 0))
                item_ids.append(item_id)
        
        if len(all_entities) == 0:
            raise ValueError("没有找到实体数据")
        
        # 创建实体到索引的映射
        entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}
        
        # 构建实体特征（简单的one-hot编码）
        num_entities = len(all_entities)
        entity_features = torch.eye(num_entities)
        
        # 构建共现边
        edges = []
        edge_weights = []
        entity_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # 统计共现
        for item_id, entities in item_entities.items():
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    if i != j and entity1 in entity_to_idx and entity2 in entity_to_idx:
                        entity_cooccurrence[entity1][entity2] += 1
        
        # 构建边索引
        for entity1, cooccur_dict in entity_cooccurrence.items():
            for entity2, count in cooccur_dict.items():
                if count > 1:  # 至少共现2次
                    idx1 = entity_to_idx[entity1]
                    idx2 = entity_to_idx[entity2]
                    edges.append([idx1, idx2])
                    edge_weights.append(count)
        
        if len(edges) == 0:
            # 如果没有共现边，创建一些基于文本相似度的边
            print("⚠️  没有足够的实体共现，使用随机连接")
            edges = [[i, (i + 1) % num_entities] for i in range(min(num_entities, 10))]
            edge_weights = [1.0] * len(edges)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        # 为每个数据项创建标签（使用实体的平均）
        item_labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(num_entities, dtype=torch.long)
        
        # 创建图数据
        graph_data = Data(
            x=entity_features,
            edge_index=edge_index,
            edge_attr=edge_weights,
            y=item_labels,
            num_nodes=num_entities
        )
        
        # 添加额外信息
        graph_data.entities = list(all_entities)
        graph_data.entity_to_idx = entity_to_idx
        graph_data.item_entities = item_entities
        
        print(f"✅ 实体共现图构建完成:")
        print(f"   实体数: {num_entities}")
        print(f"   边数: {edge_index.size(1)}")
        print(f"   数据项数: {len(item_ids)}")
        
        return graph_data
    
    def build_domain_graph(self, data: Dict[str, Any]) -> Data:
        """
        基于域名构建图
        
        Args:
            data: MR2数据
            
        Returns:
            PyG Data对象
        """
        print("🌐 构建域名图...")
        
        # 提取域名信息
        domains = set()
        item_domains = {}
        labels = []
        item_ids = []
        
        for item_id, item_data in data.items():
            item_domain_list = []
            
            # 从direct annotation提取域名
            if 'direct_path' in item_data:
                direct_annotation_file = self.data_dir / item_data['direct_path'] / 'direct_annotation.json'
                if direct_annotation_file.exists():
                    try:
                        with open(direct_annotation_file, 'r', encoding='utf-8') as f:
                            direct_data = json.load(f)
                        
                        # 提取域名
                        for img_data in direct_data.get('images_with_captions', []):
                            if 'domain' in img_data:
                                domain = img_data['domain']
                                domains.add(domain)
                                item_domain_list.append(domain)
                        
                        for img_data in direct_data.get('images_with_no_captions', []):
                            if 'domain' in img_data:
                                domain = img_data['domain']
                                domains.add(domain)
                                item_domain_list.append(domain)
                                
                    except Exception as e:
                        logger.warning(f"读取direct annotation失败: {e}")
            
            # 从inverse annotation提取域名
            if 'inv_path' in item_data:
                inv_annotation_file = self.data_dir / item_data['inv_path'] / 'inverse_annotation.json'
                if inv_annotation_file.exists():
                    try:
                        with open(inv_annotation_file, 'r', encoding='utf-8') as f:
                            inv_data = json.load(f)
                        
                        # 从匹配结果提取域名
                        for match_data in inv_data.get('fully_matched_no_text', []):
                            if 'domain' in match_data:
                                domain = match_data['domain']
                                domains.add(domain)
                                item_domain_list.append(domain)
                                
                    except Exception as e:
                        logger.warning(f"读取inverse annotation失败: {e}")
            
            if item_domain_list:
                item_domains[item_id] = list(set(item_domain_list))
                labels.append(item_data.get('label', 0))
                item_ids.append(item_id)
        
        if len(domains) == 0:
            raise ValueError("没有找到域名数据")
        
        # 创建域名到索引的映射
        domain_to_idx = {domain: idx for idx, domain in enumerate(domains)}
        
        # 构建域名特征（基于域名的简单特征）
        domain_features = []
        for domain in domains:
            feature = self._extract_domain_features(domain)
            domain_features.append(feature)
        
        domain_features = torch.tensor(domain_features, dtype=torch.float)
        
        # 构建边（域名相似性）
        edges = []
        edge_weights = []
        
        domain_list = list(domains)
        for i, domain1 in enumerate(domain_list):
            for j, domain2 in enumerate(domain_list):
                if i < j:
                    similarity = self._compute_domain_similarity(domain1, domain2)
                    if similarity > 0.1:  # 相似度阈值
                        edges.append([i, j])
                        edges.append([j, i])  # 无向图
                        edge_weights.extend([similarity, similarity])
        
        if len(edges) == 0:
            # 如果没有相似边，创建一些基本连接
            print("⚠️  没有足够的域名相似性，使用基本连接")
            for i in range(min(len(domains), 5)):
                for j in range(i + 1, min(len(domains), 5)):
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_weights.extend([0.5, 0.5])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float) if edge_weights else torch.zeros(0)
        
        # 创建域名级别的标签（基于关联的数据项）
        domain_labels = []
        for domain in domains:
            # 找到与该域名关联的数据项的标签
            related_labels = []
            for item_id, item_domain_list in item_domains.items():
                if domain in item_domain_list:
                    item_idx = item_ids.index(item_id)
                    related_labels.append(labels[item_idx])
            
            if related_labels:
                # 使用众数作为域名标签
                domain_label = max(set(related_labels), key=related_labels.count)
            else:
                domain_label = 0
            domain_labels.append(domain_label)
        
        # 创建图数据
        graph_data = Data(
            x=domain_features,
            edge_index=edge_index,
            edge_attr=edge_weights,
            y=torch.tensor(domain_labels, dtype=torch.long),
            num_nodes=len(domains)
        )
        
        # 添加额外信息
        graph_data.domains = list(domains)
        graph_data.domain_to_idx = domain_to_idx
        graph_data.item_domains = item_domains
        
        print(f"✅ 域名图构建完成:")
        print(f"   域名数: {len(domains)}")
        print(f"   边数: {edge_index.size(1)}")
        print(f"   数据项数: {len(item_ids)}")
        
        return graph_data
    
    def _compute_text_features(self, texts: List[str]) -> torch.Tensor:
        """计算文本特征"""
        if self.text_processor:
            # 使用项目的文本处理器
            features = []
            for text in texts:
                text_features = self.text_processor.extract_features(text)
                # 转换为向量
                feature_vector = [
                    text_features.get('text_length', 0),
                    text_features.get('word_count', 0),
                    text_features.get('char_count', 0),
                    text_features.get('token_count', 0),
                    text_features.get('exclamation_count', 0),
                    text_features.get('question_count', 0),
                    text_features.get('uppercase_ratio', 0),
                    text_features.get('digit_count', 0),
                    text_features.get('url_count', 0),
                    text_features.get('mention_count', 0),
                    text_features.get('hashtag_count', 0),
                    text_features.get('emoji_count', 0)
                ]
                features.append(feature_vector)
        else:
            # 简单的文本特征
            features = []
            for text in texts:
                feature_vector = [
                    len(text),  # 文本长度
                    len(text.split()),  # 词数
                    text.count('!'),  # 感叹号数
                    text.count('?'),  # 问号数
                    sum(1 for c in text if c.isupper()) / max(len(text), 1),  # 大写比例
                    sum(1 for c in text if c.isdigit()),  # 数字数
                    text.count('http'),  # URL数（简单）
                    text.count('@'),  # 提及数
                ]
                features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float)
    
    def _compute_cosine_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """计算余弦相似度矩阵"""
        # L2标准化
        features_norm = F.normalize(features, p=2, dim=1)
        # 计算相似度矩阵
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        return similarity_matrix
    
    def _build_edges_from_similarity(self, similarity_matrix: torch.Tensor, 
                                   threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """从相似度矩阵构建边"""
        num_nodes = similarity_matrix.size(0)
        edges = []
        edge_weights = []
        
        for i in range(num_nodes):
            # 获取与节点i最相似的节点
            similarities = similarity_matrix[i]
            
            # 排除自身
            similarities[i] = -1
            
            # 找到超过阈值的邻居
            valid_neighbors = torch.where(similarities > threshold)[0]
            
            # 限制每个节点的边数
            if len(valid_neighbors) > self.max_edges_per_node:
                _, top_indices = torch.topk(similarities, self.max_edges_per_node)
                valid_neighbors = top_indices[similarities[top_indices] > threshold]
            
            for j in valid_neighbors:
                edges.append([i, j.item()])
                edge_weights.append(similarities[j].item())
        
        if len(edges) == 0:
            # 如果没有满足阈值的边，创建一些基本连接
            print("⚠️  没有满足阈值的边，创建基本连接")
            for i in range(min(num_nodes, 10)):
                j = (i + 1) % num_nodes
                edges.append([i, j])
                edge_weights.append(0.5)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_weights
    
    def _extract_domain_features(self, domain: str) -> List[float]:
        """提取域名特征"""
        features = [
            len(domain),  # 域名长度
            domain.count('.'),  # 点的数量
            1.0 if 'com' in domain else 0.0,  # 是否包含com
            1.0 if 'org' in domain else 0.0,  # 是否包含org
            1.0 if 'net' in domain else 0.0,  # 是否包含net
            1.0 if 'edu' in domain else 0.0,  # 是否包含edu
            1.0 if 'gov' in domain else 0.0,  # 是否包含gov
            1.0 if any(char.isdigit() for char in domain) else 0.0,  # 是否包含数字
        ]
        return features
    
    def _compute_domain_similarity(self, domain1: str, domain2: str) -> float:
        """计算域名相似度"""
        # 基于编辑距离的相似度
        def edit_distance(s1, s2):
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        max_len = max(len(domain1), len(domain2))
        if max_len == 0:
            return 1.0
        
        distance = edit_distance(domain1, domain2)
        similarity = 1.0 - (distance / max_len)
        
        return max(0.0, similarity)
    
    def save_graph(self, graph_data: Data, filename: str):
        """保存图数据 - 修复PyTorch 2.6兼容性"""
        save_path = self.output_dir / filename
        
        # 解决PyTorch 2.6的weights_only=True问题
        try:
            # 尝试使用weights_only=False（适用于可信来源）
            torch.save(graph_data, save_path, _use_new_zipfile_serialization=False)
            print(f"💾 图数据已保存: {save_path}")
        except Exception as e:
            print(f"⚠️  PyTorch保存失败，尝试替代方案: {e}")
            # 使用字典格式保存关键数据
            graph_dict = {
                'x': graph_data.x,
                'edge_index': graph_data.edge_index,
                'edge_attr': graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None,
                'y': graph_data.y if hasattr(graph_data, 'y') else None,
                'num_nodes': graph_data.num_nodes,
                'metadata': {
                    'item_ids': getattr(graph_data, 'item_ids', None),
                    'texts': getattr(graph_data, 'texts', None),
                    'entities': getattr(graph_data, 'entities', None),
                    'domains': getattr(graph_data, 'domains', None)
                }
            }
            
            # 保存为pickle文件
            import pickle
            pickle_path = save_path.with_suffix('.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(graph_dict, f)
            print(f"💾 图数据已保存为pickle: {pickle_path}")
    
    def load_graph(self, filename: str) -> Data:
        """加载图数据 - 修复PyTorch 2.6兼容性"""
        load_path = self.output_dir / filename
        
        if not load_path.exists():
            # 尝试找pickle版本
            pickle_path = load_path.with_suffix('.pkl')
            if pickle_path.exists():
                print(f"📂 加载pickle格式图数据: {pickle_path}")
                import pickle
                with open(pickle_path, 'rb') as f:
                    graph_dict = pickle.load(f)
                
                # 重建Data对象
                graph_data = Data(
                    x=graph_dict['x'],
                    edge_index=graph_dict['edge_index'],
                    edge_attr=graph_dict['edge_attr'],
                    y=graph_dict['y'],
                    num_nodes=graph_dict['num_nodes']
                )
                
                # 添加metadata
                metadata = graph_dict.get('metadata', {})
                for key, value in metadata.items():
                    if value is not None:
                        setattr(graph_data, key, value)
                
                return graph_data
            else:
                raise FileNotFoundError(f"图文件不存在: {load_path} 或 {pickle_path}")
        
        try:
            # 尝试使用weights_only=False加载
            graph_data = torch.load(load_path, weights_only=False)
            print(f"📂 图数据已加载: {load_path}")
            return graph_data
        except Exception as e:
            print(f"⚠️  PyTorch加载失败: {e}")
            raise


class GraphFeatureExtractor:
    """图特征提取器"""
    
    def __init__(self):
        """初始化图特征提取器"""
        pass
    
    def extract_node_features(self, graph_data: Data) -> Dict[str, torch.Tensor]:
        """
        提取节点特征
        
        Args:
            graph_data: 图数据
            
        Returns:
            特征字典
        """
        features = {}
        
        # 转换为NetworkX图以便计算图特征
        nx_graph = to_networkx(graph_data, to_undirected=True)
        
        # 度中心性
        degree_centrality = nx.degree_centrality(nx_graph)
        features['degree_centrality'] = torch.tensor(
            [degree_centrality.get(i, 0.0) for i in range(graph_data.num_nodes)],
            dtype=torch.float
        )
        
        # 介数中心性
        try:
            betweenness_centrality = nx.betweenness_centrality(nx_graph)
            features['betweenness_centrality'] = torch.tensor(
                [betweenness_centrality.get(i, 0.0) for i in range(graph_data.num_nodes)],
                dtype=torch.float
            )
        except:
            features['betweenness_centrality'] = torch.zeros(graph_data.num_nodes)
        
        # 紧密性中心性
        try:
            closeness_centrality = nx.closeness_centrality(nx_graph)
            features['closeness_centrality'] = torch.tensor(
                [closeness_centrality.get(i, 0.0) for i in range(graph_data.num_nodes)],
                dtype=torch.float
            )
        except:
            features['closeness_centrality'] = torch.zeros(graph_data.num_nodes)
        
        # 节点度
        degrees = torch.zeros(graph_data.num_nodes)
        edge_index = graph_data.edge_index
        for i in range(graph_data.num_nodes):
            degrees[i] = (edge_index[0] == i).sum() + (edge_index[1] == i).sum()
        features['degree'] = degrees
        
        # 聚类系数
        try:
            clustering = nx.clustering(nx_graph)
            features['clustering'] = torch.tensor(
                [clustering.get(i, 0.0) for i in range(graph_data.num_nodes)],
                dtype=torch.float
            )
        except:
            features['clustering'] = torch.zeros(graph_data.num_nodes)
        
        return features
    
    def extract_graph_features(self, graph_data: Data) -> Dict[str, float]:
        """
        提取图级特征
        
        Args:
            graph_data: 图数据
            
        Returns:
            特征字典
        """
        features = {}
        
        # 转换为NetworkX图
        nx_graph = to_networkx(graph_data, to_undirected=True)
        
        # 基本统计
        features['num_nodes'] = graph_data.num_nodes
        features['num_edges'] = graph_data.edge_index.size(1) // 2  # 无向图
        features['density'] = nx.density(nx_graph)
        
        # 连通性
        features['num_connected_components'] = nx.number_connected_components(nx_graph)
        features['is_connected'] = float(nx.is_connected(nx_graph))
        
        # 度分布统计
        degrees = [nx_graph.degree(n) for n in nx_graph.nodes()]
        if degrees:
            features['avg_degree'] = np.mean(degrees)
            features['max_degree'] = np.max(degrees)
            features['min_degree'] = np.min(degrees)
            features['degree_std'] = np.std(degrees)
        else:
            features['avg_degree'] = 0.0
            features['max_degree'] = 0.0
            features['min_degree'] = 0.0
            features['degree_std'] = 0.0
        
        # 路径相关特征
        try:
            if nx.is_connected(nx_graph):
                features['avg_shortest_path_length'] = nx.average_shortest_path_length(nx_graph)
                features['diameter'] = nx.diameter(nx_graph)
                features['radius'] = nx.radius(nx_graph)
            else:
                # 对于非连通图，使用最大连通分量
                largest_cc = max(nx.connected_components(nx_graph), key=len)
                subgraph = nx_graph.subgraph(largest_cc)
                features['avg_shortest_path_length'] = nx.average_shortest_path_length(subgraph)
                features['diameter'] = nx.diameter(subgraph)
                features['radius'] = nx.radius(subgraph)
        except:
            features['avg_shortest_path_length'] = 0.0
            features['diameter'] = 0.0
            features['radius'] = 0.0
        
        # 聚类系数
        try:
            features['avg_clustering'] = nx.average_clustering(nx_graph)
            features['transitivity'] = nx.transitivity(nx_graph)
        except:
            features['avg_clustering'] = 0.0
            features['transitivity'] = 0.0
        
        # 小世界特性
        try:
            if nx.is_connected(nx_graph) and len(nx_graph) > 3:
                random_graph = nx.erdos_renyi_graph(len(nx_graph), features['density'])
                if nx.is_connected(random_graph):
                    actual_clustering = features['avg_clustering']
                    random_clustering = nx.average_clustering(random_graph)
                    actual_path_length = features['avg_shortest_path_length']
                    random_path_length = nx.average_shortest_path_length(random_graph)
                    
                    if random_clustering > 0 and random_path_length > 0:
                        features['small_world_sigma'] = (actual_clustering / random_clustering) / (actual_path_length / random_path_length)
                    else:
                        features['small_world_sigma'] = 0.0
                else:
                    features['small_world_sigma'] = 0.0
            else:
                features['small_world_sigma'] = 0.0
        except:
            features['small_world_sigma'] = 0.0
        
        return features
    
    def augment_node_features(self, graph_data: Data) -> Data:
        """
        用图特征增强节点特征
        
        Args:
            graph_data: 原始图数据
            
        Returns:
            增强后的图数据
        """
        # 提取图特征
        node_features = self.extract_node_features(graph_data)
        
        # 将图特征与原始特征拼接
        original_features = graph_data.x
        augmented_features = [original_features]
        
        for feature_name, feature_values in node_features.items():
            # 确保特征维度匹配
            if feature_values.size(0) == graph_data.num_nodes:
                augmented_features.append(feature_values.unsqueeze(1))
        
        # 拼接所有特征
        graph_data.x = torch.cat(augmented_features, dim=1)
        
        print(f"✅ 节点特征增强完成: {original_features.size(1)} -> {graph_data.x.size(1)}")
        
        return graph_data


def create_multi_layer_graph(data: Dict[str, Any], graph_builder: SocialGraphBuilder) -> Dict[str, Data]:
    """
    创建多层图结构
    
    Args:
        data: MR2数据
        graph_builder: 图构建器
        
    Returns:
        多层图字典
    """
    print("🏗️  创建多层图结构...")
    
    graphs = {}
    
    try:
        # 文本相似度图
        print("\n1. 构建文本相似度图...")
        text_graph = graph_builder.build_text_similarity_graph(data, similarity_threshold=0.2)
        graphs['text_similarity'] = text_graph
    except Exception as e:
        print(f"❌ 文本相似度图构建失败: {e}")
        graphs['text_similarity'] = None
    
    try:
        # 实体共现图
        print("\n2. 构建实体共现图...")
        entity_graph = graph_builder.build_entity_cooccurrence_graph(data)
        graphs['entity_cooccurrence'] = entity_graph
    except Exception as e:
        print(f"❌ 实体共现图构建失败: {e}")
        graphs['entity_cooccurrence'] = None
    
    try:
        # 域名图
        print("\n3. 构建域名图...")
        domain_graph = graph_builder.build_domain_graph(data)
        graphs['domain'] = domain_graph
    except Exception as e:
        print(f"❌ 域名图构建失败: {e}")
        graphs['domain'] = None
    
    # 统计信息
    print(f"\n📊 多层图构建完成:")
    for graph_name, graph_data in graphs.items():
        if graph_data is not None:
            print(f"   {graph_name}: {graph_data.num_nodes} 节点, {graph_data.edge_index.size(1)} 边")
        else:
            print(f"   {graph_name}: 构建失败")
    
    return graphs


# 使用示例和测试代码
if __name__ == "__main__":
    print("🔗 测试社交图构建模块")
    
    # 创建图构建器
    try:
        builder = SocialGraphBuilder()
        
        # 加载数据
        print("\n📚 加载MR2数据...")
        train_data = builder.load_mr2_data('train')
        
        # 限制数据量以便测试
        limited_data = dict(list(train_data.items())[:50])  # 只使用前50条数据
        print(f"使用 {len(limited_data)} 条数据进行测试")
        
        # 创建多层图
        print("\n🏗️  创建多层图结构...")
        graphs = create_multi_layer_graph(limited_data, builder)
        
        # 测试特征提取
        feature_extractor = GraphFeatureExtractor()
        
        for graph_name, graph_data in graphs.items():
            if graph_data is not None:
                print(f"\n📊 分析 {graph_name} 图:")
                
                # 提取图级特征
                graph_features = feature_extractor.extract_graph_features(graph_data)
                print(f"   图级特征:")
                for feature_name, value in list(graph_features.items())[:5]:  # 只显示前5个特征
                    print(f"     {feature_name}: {value:.4f}")
                
                # 增强节点特征
                try:
                    augmented_graph = feature_extractor.augment_node_features(graph_data)
                    print(f"   节点特征维度: {graph_data.x.size(1)} -> {augmented_graph.x.size(1)}")
                except Exception as e:
                    print(f"   节点特征增强失败: {e}")
                
                # 保存图
                try:
                    filename = f"{graph_name}_graph.pt"
                    builder.save_graph(graph_data, filename)
                except Exception as e:
                    print(f"   保存图失败: {e}")
        
        # 测试图加载
        print(f"\n💾 测试图加载...")
        try:
            for graph_name in graphs.keys():
                if graphs[graph_name] is not None:
                    filename = f"{graph_name}_graph.pt"
                    loaded_graph = builder.load_graph(filename)
                    print(f"   ✅ {graph_name} 图加载成功: {loaded_graph.num_nodes} 节点")
                    break  # 只测试一个
        except Exception as e:
            print(f"   ❌ 图加载测试失败: {e}")
        
        print(f"\n✅ 社交图构建模块测试完成")
        print(f"   输出目录: {builder.output_dir}")
        
    except Exception as e:
        print(f"❌ 社交图构建测试失败: {e}")
        
        # 创建演示图数据
        print("\n🔧 创建演示图数据...")
        
        # 简单的演示图
        num_nodes = 20
        edge_index = torch.tensor([
            [i for i in range(num_nodes-1)] + [i for i in range(1, num_nodes)],
            [i for i in range(1, num_nodes)] + [i for i in range(num_nodes-1)]
        ], dtype=torch.long)
        
        node_features = torch.randn(num_nodes, 8)
        labels = torch.randint(0, 3, (num_nodes,))
        
        demo_graph = Data(
            x=node_features,
            edge_index=edge_index,
            y=labels,
            num_nodes=num_nodes
        )
        
        print(f"✅ 演示图创建成功:")
        print(f"   节点数: {demo_graph.num_nodes}")
        print(f"   边数: {demo_graph.edge_index.size(1)}")
        print(f"   特征维度: {demo_graph.x.size(1)}")
        
        # 测试特征提取
        feature_extractor = GraphFeatureExtractor()
        
        print(f"\n📊 测试特征提取...")
        try:
            graph_features = feature_extractor.extract_graph_features(demo_graph)
            print(f"   图级特征数: {len(graph_features)}")
            
            augmented_graph = feature_extractor.augment_node_features(demo_graph)
            print(f"   增强后特征维度: {augmented_graph.x.size(1)}")
            
        except Exception as e:
            print(f"   ❌ 特征提取失败: {e}")
    
    print("\n✅ 社交图构建模块测试完成")