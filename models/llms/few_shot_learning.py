#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/llms/few_shot_learning.py

"""
少样本学习策略模块
实现多种少样本学习方法用于谣言检测
包括示例选择、示例排序、动态示例生成等功能
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入项目模块
try:
    from data_utils.data_loaders import create_all_dataloaders
    from utils.config_manager import get_config_manager, get_output_path
    from models.llms.prompt_engineering import PromptManager
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块")
except ImportError as e:
    print(f"⚠️  导入项目模块失败: {e}")
    USE_PROJECT_MODULES = False


class ExampleSelector:
    """示例选择器基类"""
    
    def __init__(self, selection_strategy: str = "random"):
        """
        初始化示例选择器
        
        Args:
            selection_strategy: 选择策略 ("random", "similarity", "diversity", "balanced")
        """
        self.selection_strategy = selection_strategy
        self.vectorizer = None
        
    def select_examples(self, 
                       candidate_examples: List[Dict],
                       query_text: str,
                       num_examples: int = 3,
                       **kwargs) -> List[Dict]:
        """
        选择示例
        
        Args:
            candidate_examples: 候选示例列表
            query_text: 查询文本
            num_examples: 选择的示例数量
            **kwargs: 其他参数
            
        Returns:
            选择的示例列表
        """
        if self.selection_strategy == "random":
            return self._random_selection(candidate_examples, num_examples)
        elif self.selection_strategy == "similarity":
            return self._similarity_selection(candidate_examples, query_text, num_examples)
        elif self.selection_strategy == "diversity":
            return self._diversity_selection(candidate_examples, num_examples)
        elif self.selection_strategy == "balanced":
            return self._balanced_selection(candidate_examples, num_examples)
        else:
            return self._random_selection(candidate_examples, num_examples)
    
    def _random_selection(self, examples: List[Dict], num_examples: int) -> List[Dict]:
        """随机选择示例"""
        if len(examples) <= num_examples:
            return examples
        return random.sample(examples, num_examples)
    
    def _similarity_selection(self, examples: List[Dict], query_text: str, num_examples: int) -> List[Dict]:
        """基于相似度选择示例"""
        if len(examples) <= num_examples:
            return examples
        
        try:
            # 提取文本
            texts = [example.get('text', '') for example in examples]
            all_texts = texts + [query_text]
            
            # 计算TF-IDF向量
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # 计算相似度
            query_vector = tfidf_matrix[-1]
            example_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(query_vector, example_vectors).flatten()
            
            # 选择最相似的示例
            top_indices = np.argsort(similarities)[-num_examples:]
            return [examples[i] for i in top_indices]
            
        except Exception as e:
            logger.warning(f"相似度选择失败，回退到随机选择: {e}")
            return self._random_selection(examples, num_examples)
    
    def _diversity_selection(self, examples: List[Dict], num_examples: int) -> List[Dict]:
        """基于多样性选择示例"""
        if len(examples) <= num_examples:
            return examples
        
        try:
            # 提取文本
            texts = [example.get('text', '') for example in examples]
            
            # 计算TF-IDF向量
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # 贪心选择多样性示例
            selected_indices = []
            remaining_indices = list(range(len(examples)))
            
            # 随机选择第一个
            first_idx = random.choice(remaining_indices)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            
            # 依次选择与已选示例最不相似的
            for _ in range(num_examples - 1):
                if not remaining_indices:
                    break
                
                max_min_distance = -1
                best_idx = None
                
                for idx in remaining_indices:
                    # 计算与已选示例的最小距离
                    min_distance = float('inf')
                    for selected_idx in selected_indices:
                        distance = 1 - cosine_similarity(
                            tfidf_matrix[idx], tfidf_matrix[selected_idx]
                        )[0][0]
                        min_distance = min(min_distance, distance)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_idx = idx
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
            
            return [examples[i] for i in selected_indices]
            
        except Exception as e:
            logger.warning(f"多样性选择失败，回退到随机选择: {e}")
            return self._random_selection(examples, num_examples)
    
    def _balanced_selection(self, examples: List[Dict], num_examples: int) -> List[Dict]:
        """平衡选择示例（确保各类别均衡）"""
        if len(examples) <= num_examples:
            return examples
        
        # 按标签分组
        label_groups = defaultdict(list)
        for example in examples:
            label = example.get('label', 'unknown')
            label_groups[label].append(example)
        
        # 计算每个标签应选择的数量
        num_labels = len(label_groups)
        examples_per_label = max(1, num_examples // num_labels)
        remaining = num_examples % num_labels
        
        selected_examples = []
        labels = list(label_groups.keys())
        
        for i, label in enumerate(labels):
            current_examples = examples_per_label + (1 if i < remaining else 0)
            current_examples = min(current_examples, len(label_groups[label]))
            
            selected = random.sample(label_groups[label], current_examples)
            selected_examples.extend(selected)
        
        # 如果选择的示例不够，随机补充
        if len(selected_examples) < num_examples:
            remaining_examples = [ex for ex in examples if ex not in selected_examples]
            additional = random.sample(
                remaining_examples, 
                min(num_examples - len(selected_examples), len(remaining_examples))
            )
            selected_examples.extend(additional)
        
        return selected_examples[:num_examples]


class FewShotLearner:
    """少样本学习器"""
    
    def __init__(self, 
                 selection_strategy: str = "balanced",
                 prompt_manager: Optional[PromptManager] = None):
        """
        初始化少样本学习器
        
        Args:
            selection_strategy: 示例选择策略
            prompt_manager: 提示管理器
        """
        self.selection_strategy = selection_strategy
        self.example_selector = ExampleSelector(selection_strategy)
        
        # 初始化提示管理器
        if prompt_manager is None and USE_PROJECT_MODULES:
            self.prompt_manager = PromptManager()
        else:
            self.prompt_manager = prompt_manager
        
        # 示例库
        self.example_pool = []
        self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
        
        # 设置输出目录
        if USE_PROJECT_MODULES:
            config_manager = get_config_manager()
            self.output_dir = get_output_path('models', 'llms')
        else:
            self.output_dir = Path('outputs/models/llms')
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载或创建示例池
        self._initialize_example_pool()
        
        print(f"🎯 少样本学习器初始化完成")
        print(f"   选择策略: {selection_strategy}")
        print(f"   示例池大小: {len(self.example_pool)}")
    
    def _initialize_example_pool(self):
        """初始化示例池"""
        try:
            if USE_PROJECT_MODULES:
                # 从真实数据集加载示例
                self._load_examples_from_dataset()
            else:
                # 使用预定义示例
                self._load_predefined_examples()
            
            print(f"✅ 示例池加载完成: {len(self.example_pool)} 个示例")
            
        except Exception as e:
            logger.warning(f"示例池初始化失败: {e}")
            self._load_predefined_examples()
    
    def _load_examples_from_dataset(self):
        """从数据集加载示例"""
        try:
            dataloaders = create_all_dataloaders(
                batch_sizes={'train': 32, 'val': 32, 'test': 32}
            )
            
            # 从训练集提取示例
            for batch in dataloaders['train']:
                texts = batch.get('text', batch.get('caption', []))
                labels = batch.get('labels', batch.get('label', []))
                
                if hasattr(labels, 'tolist'):
                    labels = labels.tolist()
                
                for text, label in zip(texts, labels):
                    if text and len(text.strip()) > 10:  # 过滤太短的文本
                        self.example_pool.append({
                            'text': text.strip(),
                            'label': self.label_mapping.get(label, 'unknown'),
                            'label_id': label
                        })
            
            # 限制示例池大小
            if len(self.example_pool) > 1000:
                self.example_pool = random.sample(self.example_pool, 1000)
            
        except Exception as e:
            logger.error(f"从数据集加载示例失败: {e}")
            raise
    
    def _load_predefined_examples(self):
        """加载预定义示例"""
        predefined_examples = [
            {
                'text': '中国科学院发布最新研究成果，在人工智能领域取得重大突破',
                'label': 'Non-rumor',
                'label_id': 0
            },
            {
                'text': '教育部正式发布新的高考改革方案，将于明年开始实施',
                'label': 'Non-rumor', 
                'label_id': 0
            },
            {
                'text': '世界卫生组织确认新冠疫苗对变异株仍然有效',
                'label': 'Non-rumor',
                'label_id': 0
            },
            {
                'text': '网传某地明天将发生大地震，请大家做好防护准备',
                'label': 'Rumor',
                'label_id': 1
            },
            {
                'text': '谣传新冠疫苗含有控制芯片，已被多项研究证实为虚假信息',
                'label': 'Rumor',
                'label_id': 1
            },
            {
                'text': '网上流传某明星涉嫌违法犯罪，但当事人已发声明辟谣',
                'label': 'Rumor',
                'label_id': 1
            },
            {
                'text': '据不完全统计，某新产品在市场上反响良好',
                'label': 'Unverified',
                'label_id': 2
            },
            {
                'text': '有消息称某公司将进行大规模裁员，但公司尚未官方回应',
                'label': 'Unverified',
                'label_id': 2
            },
            {
                'text': '业内人士透露，某行业可能面临重大政策调整',
                'label': 'Unverified',
                'label_id': 2
            },
            # 英文示例
            {
                'text': 'NASA announces successful launch of new Mars exploration mission',
                'label': 'Non-rumor',
                'label_id': 0
            },
            {
                'text': 'Breaking: Celebrity found dead in apparent overdose, police investigating',
                'label': 'Rumor',
                'label_id': 1
            },
            {
                'text': 'Sources suggest major tech company planning significant layoffs',
                'label': 'Unverified',
                'label_id': 2
            }
        ]
        
        self.example_pool = predefined_examples
    
    def select_examples_for_query(self, 
                                 query_text: str,
                                 num_examples: int = 3,
                                 strategy: Optional[str] = None) -> List[Dict]:
        """
        为查询文本选择示例
        
        Args:
            query_text: 查询文本
            num_examples: 示例数量
            strategy: 选择策略（可选，覆盖默认策略）
            
        Returns:
            选择的示例列表
        """
        if strategy:
            selector = ExampleSelector(strategy)
        else:
            selector = self.example_selector
        
        selected = selector.select_examples(
            self.example_pool,
            query_text,
            num_examples
        )
        
        return selected
    
    def create_few_shot_prompt(self, 
                              query_text: str,
                              num_examples: int = 3,
                              selection_strategy: Optional[str] = None,
                              prompt_style: str = "formal") -> str:
        """
        创建少样本提示
        
        Args:
            query_text: 查询文本
            num_examples: 示例数量
            selection_strategy: 选择策略
            prompt_style: 提示风格
            
        Returns:
            少样本提示字符串
        """
        # 选择示例
        examples = self.select_examples_for_query(
            query_text, 
            num_examples, 
            selection_strategy
        )
        
        # 生成提示
        if self.prompt_manager:
            prompt = self.prompt_manager.create_few_shot_prompt(
                query_text, 
                examples,
                style=prompt_style
            )
        else:
            prompt = self._create_simple_few_shot_prompt(query_text, examples)
        
        return prompt
    
    def _create_simple_few_shot_prompt(self, query_text: str, examples: List[Dict]) -> str:
        """创建简单的少样本提示"""
        prompt = "以下是一些谣言检测的例子：\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"例子{i}:\n"
            prompt += f"文本: {example['text']}\n"
            prompt += f"标签: {example['label']}\n\n"
        
        prompt += f"现在请分析以下文本:\n文本: {query_text}\n标签: "
        
        return prompt
    
    def analyze_example_distribution(self) -> Dict[str, Any]:
        """分析示例池的分布情况"""
        label_counts = Counter([ex['label'] for ex in self.example_pool])
        text_lengths = [len(ex['text']) for ex in self.example_pool]
        
        analysis = {
            'total_examples': len(self.example_pool),
            'label_distribution': dict(label_counts),
            'text_length_stats': {
                'mean': np.mean(text_lengths),
                'median': np.median(text_lengths),
                'min': min(text_lengths),
                'max': max(text_lengths),
                'std': np.std(text_lengths)
            },
            'balance_ratio': min(label_counts.values()) / max(label_counts.values()) if label_counts else 0
        }
        
        return analysis
    
    def evaluate_selection_strategies(self, 
                                    test_queries: List[str],
                                    num_examples: int = 3) -> Dict[str, Any]:
        """
        评估不同选择策略的效果
        
        Args:
            test_queries: 测试查询列表
            num_examples: 示例数量
            
        Returns:
            评估结果
        """
        strategies = ["random", "similarity", "diversity", "balanced"]
        results = {}
        
        for strategy in strategies:
            print(f"🔍 评估策略: {strategy}")
            
            strategy_results = {
                'strategy': strategy,
                'examples_selected': [],
                'label_distributions': [],
                'diversity_scores': []
            }
            
            for query in test_queries:
                # 选择示例
                examples = self.select_examples_for_query(
                    query, num_examples, strategy
                )
                
                # 分析选择的示例
                labels = [ex['label'] for ex in examples]
                label_dist = Counter(labels)
                
                # 计算多样性分数（简单的标签多样性）
                diversity_score = len(set(labels)) / len(labels) if labels else 0
                
                strategy_results['examples_selected'].append(examples)
                strategy_results['label_distributions'].append(label_dist)
                strategy_results['diversity_scores'].append(diversity_score)
            
            # 计算平均多样性
            strategy_results['avg_diversity'] = np.mean(strategy_results['diversity_scores'])
            
            results[strategy] = strategy_results
        
        return results
    
    def optimize_example_selection(self, 
                                  validation_queries: List[Tuple[str, str]],
                                  num_examples: int = 3) -> str:
        """
        基于验证集优化示例选择策略
        
        Args:
            validation_queries: 验证查询列表 [(text, true_label), ...]
            num_examples: 示例数量
            
        Returns:
            最优策略名称
        """
        strategies = ["random", "similarity", "diversity", "balanced"]
        strategy_scores = {}
        
        print("🔍 优化示例选择策略...")
        
        for strategy in strategies:
            print(f"   测试策略: {strategy}")
            
            # 这里可以实现更复杂的评估逻辑
            # 目前使用简单的多样性分数作为评估指标
            diversity_scores = []
            
            for query_text, true_label in validation_queries:
                examples = self.select_examples_for_query(
                    query_text, num_examples, strategy
                )
                
                labels = [ex['label'] for ex in examples]
                diversity_score = len(set(labels)) / len(labels) if labels else 0
                diversity_scores.append(diversity_score)
            
            avg_score = np.mean(diversity_scores)
            strategy_scores[strategy] = avg_score
            
            print(f"   {strategy}: {avg_score:.4f}")
        
        # 选择最优策略
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        
        print(f"✅ 最优策略: {best_strategy} (分数: {strategy_scores[best_strategy]:.4f})")
        
        # 更新选择策略
        self.selection_strategy = best_strategy
        self.example_selector = ExampleSelector(best_strategy)
        
        return best_strategy
    
    def add_examples(self, new_examples: List[Dict]):
        """添加新示例到示例池"""
        for example in new_examples:
            if 'text' in example and 'label' in example:
                self.example_pool.append(example)
        
        print(f"✅ 添加 {len(new_examples)} 个新示例，示例池大小: {len(self.example_pool)}")
    
    def save_example_pool(self, save_path: Optional[str] = None):
        """保存示例池"""
        if save_path is None:
            save_path = self.output_dir / "example_pool.json"
        
        data = {
            'examples': self.example_pool,
            'selection_strategy': self.selection_strategy,
            'label_mapping': self.label_mapping,
            'total_examples': len(self.example_pool)
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 示例池已保存到: {save_path}")
    
    def load_example_pool(self, load_path: str):
        """加载示例池"""
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.example_pool = data.get('examples', [])
        self.selection_strategy = data.get('selection_strategy', 'balanced')
        self.label_mapping = data.get('label_mapping', self.label_mapping)
        
        # 更新选择器
        self.example_selector = ExampleSelector(self.selection_strategy)
        
        print(f"✅ 示例池已从 {load_path} 加载: {len(self.example_pool)} 个示例")


class AdaptiveFewShotLearner(FewShotLearner):
    """自适应少样本学习器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_history = []
        self.strategy_performance = defaultdict(list)
    
    def adaptive_select_examples(self, 
                                query_text: str,
                                num_examples: int = 3,
                                performance_threshold: float = 0.7) -> List[Dict]:
        """
        自适应选择示例
        
        Args:
            query_text: 查询文本
            num_examples: 示例数量
            performance_threshold: 性能阈值
            
        Returns:
            选择的示例
        """
        # 如果历史性能不佳，尝试不同策略
        if self._should_explore_strategy(performance_threshold):
            strategy = self._choose_exploration_strategy()
            print(f"🔄 探索新策略: {strategy}")
        else:
            strategy = self.selection_strategy
        
        return self.select_examples_for_query(query_text, num_examples, strategy)
    
    def _should_explore_strategy(self, threshold: float) -> bool:
        """判断是否应该探索新策略"""
        if len(self.performance_history) < 5:
            return False
        
        recent_performance = np.mean(self.performance_history[-5:])
        return recent_performance < threshold
    
    def _choose_exploration_strategy(self) -> str:
        """选择探索策略"""
        strategies = ["random", "similarity", "diversity", "balanced"]
        
        # 基于历史性能选择
        if self.strategy_performance:
            strategy_scores = {
                strategy: np.mean(scores) 
                for strategy, scores in self.strategy_performance.items()
                if scores
            }
            if strategy_scores:
                return max(strategy_scores, key=strategy_scores.get)
        
        # 随机选择
        return random.choice(strategies)
    
    def update_performance(self, strategy: str, performance: float):
        """更新性能记录"""
        self.performance_history.append(performance)
        self.strategy_performance[strategy].append(performance)
        
        # 保持历史记录在合理范围内
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        for strategy_name in self.strategy_performance:
            if len(self.strategy_performance[strategy_name]) > 50:
                self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-50:]


def demo_few_shot_learning():
    """演示少样本学习功能"""
    print("🎯 少样本学习演示")
    print("=" * 50)
    
    try:
        # 创建少样本学习器
        learner = FewShotLearner(selection_strategy="balanced")
        
        # 分析示例池
        print("📊 示例池分析:")
        analysis = learner.analyze_example_distribution()
        print(f"   总示例数: {analysis['total_examples']}")
        print(f"   标签分布: {analysis['label_distribution']}")
        print(f"   平均文本长度: {analysis['text_length_stats']['mean']:.1f}")
        print(f"   平衡比例: {analysis['balance_ratio']:.2f}")
        
        # 测试不同选择策略
        test_query = "专家学者在国际期刊发表重要研究成果"
        print(f"\n🔍 测试查询: {test_query}")
        
        strategies = ["random", "similarity", "diversity", "balanced"]
        for strategy in strategies:
            print(f"\n策略: {strategy.upper()}")
            examples = learner.select_examples_for_query(test_query, 3, strategy)
            
            for i, example in enumerate(examples, 1):
                print(f"   例子{i}: {example['text'][:50]}... ({example['label']})")
        
        # 创建少样本提示
        print(f"\n📝 生成少样本提示:")
        prompt = learner.create_few_shot_prompt(test_query, 3, "balanced")
        print(prompt[:300] + "...")
        
        # 评估选择策略
        print(f"\n📈 评估选择策略:")
        test_queries = [
            "科学家发现新的治疗方法",
            "网传某地将发生自然灾害",
            "据消息人士透露的未确认信息"
        ]
        
        evaluation = learner.evaluate_selection_strategies(test_queries, 2)
        for strategy, result in evaluation.items():
            print(f"   {strategy}: 平均多样性 {result['avg_diversity']:.3f}")
        
        # 保存示例池
        learner.save_example_pool()
        
        print(f"\n✅ 少样本学习演示完成!")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        raise


if __name__ == "__main__":
    demo_few_shot_learning()