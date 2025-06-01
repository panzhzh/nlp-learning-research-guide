#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/graph_neural_networks/demo.py

"""
图神经网络模块演示
展示基础GNN、社交图构建、多模态GNN的完整流程
专门为MR2数据集和谣言检测任务设计
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入模块
try:
    from basic_gnn_layers import BasicGNN, GNNClassifier, create_gnn_model
    from social_graph_builder import SocialGraphBuilder, GraphFeatureExtractor, create_multi_layer_graph
    from multimodal_gnn import MultimodalGNN, MultimodalGraphClassifier, create_multimodal_gnn_config
    LOCAL_IMPORT = True
    print("✅ 成功导入本地模块")
    
    # 尝试导入项目配置
    try:
        from utils.config_manager import get_output_path
        USE_PROJECT_CONFIG = True
        print("✅ 成功导入项目配置")
    except ImportError:
        USE_PROJECT_CONFIG = False
        print("⚠️  项目配置不可用，使用默认路径")
        
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    LOCAL_IMPORT = False
    USE_PROJECT_CONFIG = False
    sys.exit(1)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNDemoRunner:
    """图神经网络演示运行器"""
    
    def __init__(self):
        """初始化演示运行器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # 输出目录
        if USE_PROJECT_CONFIG:
            try:
                self.output_dir = get_output_path('graphs', 'demo_results')
            except:
                self.output_dir = Path('outputs/graphs/demo_results')
                self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path('outputs/graphs/demo_results')
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 图神经网络演示初始化")
        print(f"   设备: {self.device}")
        print(f"   输出目录: {self.output_dir}")
    
    def demo_basic_gnn(self):
        """演示基础GNN功能"""
        print(f"\n{'='*70}")
        print("📊 基础图神经网络演示")
        print(f"{'='*70}")
        
        try:
            # 创建示例图数据
            num_nodes = 100
            num_edges = 300
            input_dim = 64
            num_classes = 3
            
            print(f"🔧 创建示例图数据:")
            print(f"   节点数: {num_nodes}")
            print(f"   边数: {num_edges}")
            print(f"   特征维度: {input_dim}")
            
            # 节点特征（随机）
            x = torch.randn(num_nodes, input_dim).to(self.device)
            
            # 边索引（创建一个连通的随机图）
            edge_list = []
            
            # 首先创建一个链确保连通性
            for i in range(num_nodes - 1):
                edge_list.append([i, i + 1])
                edge_list.append([i + 1, i])  # 无向图
            
            # 添加随机边
            remaining_edges = num_edges - (num_nodes - 1) * 2
            for _ in range(remaining_edges):
                src = np.random.randint(0, num_nodes)
                dst = np.random.randint(0, num_nodes)
                if src != dst:
                    edge_list.append([src, dst])
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
            
            # 标签（随机）
            labels = torch.randint(0, num_classes, (num_nodes,)).to(self.device)
            
            print(f"✅ 图数据创建完成: {edge_index.size(1)} 条边")
            
            # 测试不同类型的GNN
            gnn_types = ['gcn', 'gat', 'graphsage', 'gin']
            gnn_results = {}
            
            for gnn_type in gnn_types:
                print(f"\n🧪 测试 {gnn_type.upper()} 模型:")
                
                start_time = time.time()
                
                # 创建模型
                model = GNNClassifier(
                    input_dim=input_dim,
                    hidden_dims=[128, 64],
                    num_classes=num_classes,
                    gnn_type=gnn_type,
                    dropout=0.5
                ).to(self.device)
                
                # 前向传播测试
                model.eval()
                with torch.no_grad():
                    # 节点级预测
                    node_logits = model(x, edge_index)
                    node_preds, node_probs = model.predict(x, edge_index)
                    
                    # 图级预测（使用batch）
                    batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
                    graph_logits = model(x, edge_index, batch)
                    graph_preds, graph_probs = model.predict(x, edge_index, batch)
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # 统计结果
                param_count = model.get_parameter_count()
                
                result = {
                    'model_type': gnn_type,
                    'parameters': param_count,
                    'inference_time': inference_time,
                    'node_output_shape': node_logits.shape,
                    'graph_output_shape': graph_logits.shape,
                    'node_predictions': node_preds[:10].cpu().numpy(),  # 前10个节点的预测
                    'graph_prediction': graph_preds.cpu().numpy()
                }
                
                gnn_results[gnn_type] = result
                
                print(f"   ✅ 参数量: {param_count:,}")
                print(f"   ✅ 推理时间: {inference_time:.4f}s")
                print(f"   ✅ 节点输出形状: {node_logits.shape}")
                print(f"   ✅ 图输出形状: {graph_logits.shape}")
            
            # 保存结果
            self.results['basic_gnn'] = gnn_results
            
            # 性能对比
            print(f"\n📊 GNN模型性能对比:")
            print(f"{'模型':<12} {'参数量':<12} {'推理时间(s)':<12}")
            print("-" * 40)
            
            for gnn_type, result in gnn_results.items():
                print(f"{gnn_type.upper():<12} {result['parameters']:<12,} {result['inference_time']:<12.4f}")
            
            print(f"✅ 基础GNN演示完成")
            return True
            
        except Exception as e:
            print(f"❌ 基础GNN演示失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def demo_social_graph_builder(self):
        """演示社交图构建功能"""
        print(f"\n{'='*70}")
        print("🔗 社交图构建演示")
        print(f"{'='*70}")
        
        try:
            # 创建图构建器
            graph_builder = SocialGraphBuilder()
            
            # 创建演示数据（模拟MR2数据格式）
            print(f"🔧 创建演示数据...")
            demo_data = self._create_demo_mr2_data()
            
            print(f"📚 演示数据统计:")
            print(f"   数据项数: {len(demo_data)}")
            
            # 构建不同类型的图
            graphs = {}
            
            print(f"\n🏗️  构建多层图结构...")
            
            # 1. 文本相似度图
            try:
                print(f"\n1. 构建文本相似度图...")
                text_graph = graph_builder.build_text_similarity_graph(demo_data, similarity_threshold=0.1)
                graphs['text_similarity'] = text_graph
                
                print(f"   ✅ 文本图: {text_graph.num_nodes} 节点, {text_graph.edge_index.size(1)} 边")
                print(f"   ✅ 特征维度: {text_graph.x.size(1)}")
                
            except Exception as e:
                print(f"   ❌ 文本图构建失败: {e}")
                graphs['text_similarity'] = None
            
            # 2. 实体共现图
            try:
                print(f"\n2. 构建实体共现图...")
                entity_graph = graph_builder.build_entity_cooccurrence_graph(demo_data)
                graphs['entity_cooccurrence'] = entity_graph
                
                print(f"   ✅ 实体图: {entity_graph.num_nodes} 节点, {entity_graph.edge_index.size(1)} 边")
                print(f"   ✅ 实体数: {len(entity_graph.entities) if hasattr(entity_graph, 'entities') else 'N/A'}")
                
            except Exception as e:
                print(f"   ❌ 实体图构建失败: {e}")
                graphs['entity_cooccurrence'] = None
            
            # 3. 域名图
            try:
                print(f"\n3. 构建域名图...")
                domain_graph = graph_builder.build_domain_graph(demo_data)
                graphs['domain'] = domain_graph
                
                print(f"   ✅ 域名图: {domain_graph.num_nodes} 节点, {domain_graph.edge_index.size(1)} 边")
                print(f"   ✅ 域名数: {len(domain_graph.domains) if hasattr(domain_graph, 'domains') else 'N/A'}")
                
            except Exception as e:
                print(f"   ❌ 域名图构建失败: {e}")
                graphs['domain'] = None
            
            # 图特征提取
            print(f"\n📊 图特征提取演示...")
            feature_extractor = GraphFeatureExtractor()
            
            for graph_name, graph_data in graphs.items():
                if graph_data is not None:
                    print(f"\n分析 {graph_name} 图:")
                    
                    try:
                        # 提取图级特征
                        graph_features = feature_extractor.extract_graph_features(graph_data)
                        
                        print(f"   图级特征:")
                        key_features = ['num_nodes', 'num_edges', 'density', 'avg_clustering', 'is_connected']
                        for feature_name in key_features:
                            if feature_name in graph_features:
                                value = graph_features[feature_name]
                                print(f"     {feature_name}: {value:.4f}" if isinstance(value, float) else f"     {feature_name}: {value}")
                        
                        # 增强节点特征
                        original_dim = graph_data.x.size(1)
                        augmented_graph = feature_extractor.augment_node_features(graph_data)
                        new_dim = augmented_graph.x.size(1)
                        
                        print(f"   节点特征增强: {original_dim} -> {new_dim}")
                        
                        # 保存图
                        filename = f"demo_{graph_name}_graph.pt"
                        graph_builder.save_graph(graph_data, filename)
                        print(f"   ✅ 图已保存: {filename}")
                        
                    except Exception as e:
                        print(f"   ❌ 特征提取失败: {e}")
            
            # 保存结果
            self.results['social_graphs'] = {
                'graphs_built': len([g for g in graphs.values() if g is not None]),
                'total_graphs': len(graphs),
                'graph_info': {
                    name: {
                        'nodes': graph.num_nodes if graph else 0,
                        'edges': graph.edge_index.size(1) if graph else 0,
                        'features': graph.x.size(1) if graph else 0
                    } for name, graph in graphs.items()
                }
            }
            
            print(f"\n✅ 社交图构建演示完成")
            print(f"   成功构建: {len([g for g in graphs.values() if g is not None])}/{len(graphs)} 个图")
            
            return graphs
            
        except Exception as e:
            print(f"❌ 社交图构建演示失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def demo_multimodal_gnn(self):
        """演示多模态GNN功能"""
        print(f"\n{'='*70}")
        print("🤖 多模态图神经网络演示")
        print(f"{'='*70}")
        
        try:
            # 测试不同的配置
            test_configs = [
                {
                    'name': 'GAT + Attention Fusion',
                    'gnn_type': 'gat',
                    'fusion_method': 'attention'
                },
                {
                    'name': 'GCN + Concatenation Fusion',
                    'gnn_type': 'gcn',
                    'fusion_method': 'concat'
                },
                {
                    'name': 'GraphSAGE + Gate Fusion',
                    'gnn_type': 'graphsage',
                    'fusion_method': 'gate'
                }
            ]
            
            multimodal_results = {}
            
            for i, config in enumerate(test_configs):
                print(f"\n🧪 测试配置 {i+1}: {config['name']}")
                print("-" * 50)
                
                try:
                    # 创建模型配置
                    model_config = create_multimodal_gnn_config(
                        gnn_type=config['gnn_type'],
                        fusion_method=config['fusion_method']
                    )
                    
                    # 创建分类器
                    classifier = MultimodalGraphClassifier(model_config, device=str(self.device))
                    
                    # 获取模型信息
                    param_info = classifier.model.get_parameter_count()
                    print(f"   📊 模型参数:")
                    print(f"     总参数: {param_info['total']:,}")
                    print(f"     可训练参数: {param_info['trainable']:,}")
                    print(f"     冻结参数: {param_info['frozen']:,}")
                    
                    # 快速训练演示
                    print(f"   🏋️ 开始演示训练...")
                    start_time = time.time()
                    
                    history = classifier.train_demo(epochs=3, learning_rate=1e-4)
                    
                    train_time = time.time() - start_time
                    
                    print(f"   ⏱️ 训练时间: {train_time:.2f}s")
                    
                    # 推理测试
                    print(f"   🔍 推理测试...")
                    classifier.test_inference()
                    
                    # 保存结果
                    result = {
                        'config_name': config['name'],
                        'gnn_type': config['gnn_type'],
                        'fusion_method': config['fusion_method'],
                        'parameters': param_info,
                        'training_time': train_time,
                        'final_train_loss': history['train_losses'][-1] if history['train_losses'] else 0,
                        'final_val_accuracy': history['val_accuracies'][-1] if history['val_accuracies'] else 0,
                        'final_val_f1': history['val_f1_scores'][-1] if history['val_f1_scores'] else 0
                    }
                    
                    multimodal_results[f"config_{i+1}"] = result
                    
                    print(f"   ✅ 配置 {i+1} 测试成功")
                    
                    # 为了节省时间，只完整测试第一个配置
                    if i == 0:
                        print(f"   ⚠️  为节省演示时间，完整测试第一个配置")
                    else:
                        print(f"   ⚠️  快速验证配置 {i+1}")
                        break
                    
                except Exception as e:
                    print(f"   ❌ 配置 {i+1} 测试失败: {e}")
                    multimodal_results[f"config_{i+1}"] = {
                        'config_name': config['name'],
                        'error': str(e)
                    }
            
            # 结果汇总
            print(f"\n📊 多模态GNN测试结果汇总:")
            print(f"{'配置':<25} {'参数量':<12} {'训练时间(s)':<12} {'验证F1':<10}")
            print("-" * 65)
            
            for config_key, result in multimodal_results.items():
                if 'error' not in result:
                    config_name = result['config_name'][:22] + "..." if len(result['config_name']) > 25 else result['config_name']
                    param_count = result['parameters']['total']
                    train_time = result['training_time']
                    val_f1 = result['final_val_f1']
                    
                    print(f"{config_name:<25} {param_count:<12,} {train_time:<12.2f} {val_f1:<10.4f}")
                else:
                    print(f"{result['config_name']:<25} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10}")
            
            # 保存结果
            self.results['multimodal_gnn'] = multimodal_results
            
            print(f"\n✅ 多模态GNN演示完成")
            return True
            
        except Exception as e:
            print(f"❌ 多模态GNN演示失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_demo_mr2_data(self):
        """创建演示MR2数据"""
        demo_data = {}
        
        # 模拟文本数据
        demo_texts = [
            "这是一个关于科技进步的真实新闻报道，包含了详细的技术细节和专家观点",
            "This is fake news about celebrity scandal without any credible sources or evidence",
            "未经证实的传言需要进一步调查验证，目前无法确定消息的真实性",
            "Breaking news: Major breakthrough in artificial intelligence announced by researchers",
            "网传某地发生重大事故，但官方尚未确认消息的准确性",
            "Scientific study reveals new insights about climate change effects",
            "谣传某知名公司即将倒闭，但公司官方已经发布辟谣声明",
            "Weather alert issued by meteorological department for severe storms",
            "社交媒体广泛流传的未证实消息引发了公众的关注和讨论",
            "Economic indicators show positive growth trends in technology sector",
            "新研究表明人工智能技术对社会产生了深远的影响",
            "Unverified claims about health benefits spread rapidly online",
            "政府发布官方声明澄清网络传言并提供了准确的信息",
            "False information about vaccine side effects causes public concern",
            "专家呼吁公众理性对待网络信息，避免盲目传播未经证实的消息"
        ]
        
        # 模拟域名数据
        demo_domains = [
            "cnn.com", "bbc.com", "reuters.com", "nytimes.com", "washingtonpost.com",
            "fakesource.net", "unreliable.org", "suspicious.info", "dubious.co",
            "twitter.com", "facebook.com", "instagram.com", "youtube.com",
            "wikipedia.org", "github.com"
        ]
        
        # 模拟实体数据
        demo_entities = [
            "artificial intelligence", "machine learning", "technology", "research",
            "climate change", "economy", "health", "vaccine", "government",
            "social media", "fake news", "misinformation", "verification"
        ]
        
        for i in range(len(demo_texts)):
            item_data = {
                'caption': demo_texts[i],
                'label': i % 3,  # 0: Non-rumor, 1: Rumor, 2: Unverified
                'direct_path': f'demo/direct/{i}',
                'inv_path': f'demo/inverse/{i}',
                'image_path': f'demo/img/{i}.jpg'
            }
            
            demo_data[str(i)] = item_data
        
        # 为了演示图构建，我们需要在内存中模拟一些标注数据
        # 这里我们直接在 demo_data 中添加模拟的实体和域名信息
        import random
        
        for item_id, item_data in demo_data.items():
            # 模拟实体
            num_entities = random.randint(2, 5)
            item_entities = random.sample(demo_entities, min(num_entities, len(demo_entities)))
            item_data['mock_entities'] = item_entities
            
            # 模拟域名
            num_domains = random.randint(1, 3)
            item_domains = random.sample(demo_domains, min(num_domains, len(demo_domains)))
            item_data['mock_domains'] = item_domains
        
        return demo_data
    
    def run_full_demo(self):
        """运行完整演示"""
        print(f"🚀 图神经网络模块完整演示")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # 1. 基础GNN演示
        print(f"\n🔄 第1部分：基础图神经网络")
        basic_success = self.demo_basic_gnn()
        
        # 2. 社交图构建演示
        print(f"\n🔄 第2部分：社交图构建")
        social_graphs = self.demo_social_graph_builder()
        
        # 3. 多模态GNN演示
        print(f"\n🔄 第3部分：多模态图神经网络")
        multimodal_success = self.demo_multimodal_gnn()
        
        total_time = time.time() - start_time
        
        # 生成演示报告
        self._generate_demo_report(basic_success, len(social_graphs) > 0, multimodal_success, total_time)
        
        return self.results
    
    def _generate_demo_report(self, basic_success, social_success, multimodal_success, total_time):
        """生成演示报告"""
        print(f"\n{'='*70}")
        print(f"📋 图神经网络模块演示报告")
        print(f"{'='*70}")
        
        print(f"🕐 总演示时间: {total_time:.2f}秒")
        print(f"\n📊 演示结果汇总:")
        
        # 基础GNN结果
        if basic_success:
            print(f"   ✅ 基础GNN: 成功")
            if 'basic_gnn' in self.results:
                gnn_count = len(self.results['basic_gnn'])
                print(f"      - 测试了 {gnn_count} 种GNN架构")
                
                fastest_model = min(self.results['basic_gnn'].items(), 
                                  key=lambda x: x[1]['inference_time'])
                print(f"      - 最快模型: {fastest_model[0].upper()} ({fastest_model[1]['inference_time']:.4f}s)")
        else:
            print(f"   ❌ 基础GNN: 失败")
        
        # 社交图构建结果
        if social_success:
            print(f"   ✅ 社交图构建: 成功")
            if 'social_graphs' in self.results:
                built_count = self.results['social_graphs']['graphs_built']
                total_count = self.results['social_graphs']['total_graphs']
                print(f"      - 成功构建: {built_count}/{total_count} 个图")
        else:
            print(f"   ❌ 社交图构建: 失败")
        
        # 多模态GNN结果
        if multimodal_success:
            print(f"   ✅ 多模态GNN: 成功")
            if 'multimodal_gnn' in self.results:
                config_count = len([r for r in self.results['multimodal_gnn'].values() 
                                  if 'error' not in r])
                print(f"      - 成功测试: {config_count} 个配置")
        else:
            print(f"   ❌ 多模态GNN: 失败")
        
        # 保存报告
        try:
            import json
            report_file = self.output_dir / 'demo_report.json'
            
            report_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_time': total_time,
                'results': {
                    'basic_gnn_success': basic_success,
                    'social_graph_success': social_success,
                    'multimodal_gnn_success': multimodal_success
                },
                'detailed_results': self.results
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 演示报告已保存: {report_file}")
            
        except Exception as e:
            print(f"\n⚠️  保存报告失败: {e}")
        
        print(f"\n🎉 图神经网络模块演示完成！")
        print(f"📁 输出目录: {self.output_dir}")


def quick_demo():
    """快速演示（5分钟版本）"""
    print("⚡ 图神经网络模块快速演示（5分钟版本）")
    
    demo_runner = GNNDemoRunner()
    
    print(f"\n🔄 运行快速演示...")
    
    # 只运行基础GNN演示
    success = demo_runner.demo_basic_gnn()
    
    if success:
        print(f"\n✅ 快速演示成功完成！")
        print(f"💡 运行完整演示: python {__file__} --full")
    else:
        print(f"\n❌ 快速演示失败")
    
    return success


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='图神经网络模块演示')
    parser.add_argument('--full', action='store_true', 
                       help='运行完整演示（包含所有模块）')
    parser.add_argument('--quick', action='store_true', 
                       help='运行快速演示（仅基础功能）')
    
    args = parser.parse_args()
    
    if args.full:
        print("🚀 运行完整演示...")
        demo_runner = GNNDemoRunner()
        results = demo_runner.run_full_demo()
    elif args.quick:
        print("⚡ 运行快速演示...")
        quick_demo()
    else:
        # 默认运行快速演示
        print("🎯 图神经网络模块演示")
        print("使用 --full 运行完整演示，或 --quick 运行快速演示")
        print("默认运行快速演示...")
        quick_demo()


if __name__ == "__main__":
    main()