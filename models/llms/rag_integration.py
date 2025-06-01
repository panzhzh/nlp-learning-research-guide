#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/llms/rag_integration.py

"""
RAG (检索增强生成) 集成模块
结合检索和生成，提升谣言检测的准确性和可解释性
基于Qwen3-0.6B模型实现，支持动态知识检索
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import sys
import json
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 尝试导入sentence_transformers和faiss，如果失败则使用备用方案
HAS_SENTENCE_TRANSFORMERS = True

try:
    import faiss
    HAS_FAISS = True
    print("✅ 成功导入faiss")
except ImportError:
    print("⚠️  faiss导入失败，将使用sklearn")
    HAS_FAISS = False

# 导入项目模块
try:
    from data_utils.data_loaders import create_all_dataloaders
    from utils.config_manager import get_config_manager, get_output_path
    from models.llms.open_source_llms import QwenRumorClassifier
    from models.llms.prompt_engineering import PromptManager
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块")
except ImportError as e:
    print(f"⚠️  导入项目模块失败: {e}")
    USE_PROJECT_MODULES = False


class KnowledgeBase:
    """知识库管理器"""
    
    def __init__(self, knowledge_sources: Optional[List[str]] = None):
        """
        初始化知识库
        
        Args:
            knowledge_sources: 知识来源列表
        """
        self.knowledge_sources = knowledge_sources or ['dataset', 'predefined']
        self.documents = []
        self.embeddings = None
        self.index = None
        self.vectorizer = None
        
        # 初始化嵌入模型
        self.embedding_model = None
        print("⚠️  使用TF-IDF进行文档检索（避免依赖冲突）")
        
        # 构建知识库
        self._build_knowledge_base()
        
    def _build_knowledge_base(self):
        """构建知识库"""
        print("🔄 构建知识库...")
        
        # 添加预定义的谣言检测知识
        self._add_predefined_knowledge()
        
        # 从数据集添加知识
        if 'dataset' in self.knowledge_sources and USE_PROJECT_MODULES:
            self._add_dataset_knowledge()
        
        # 构建索引
        self._build_index()
        
        print(f"✅ 知识库构建完成: {len(self.documents)} 个文档")
    
    def _add_predefined_knowledge(self):
        """添加预定义的谣言检测知识"""
        predefined_docs = [
            {
                'content': '权威机构发布的官方信息通常具有高可信度，如政府部门、科研机构、知名媒体的正式声明。',
                'type': 'guideline',
                'category': 'credibility',
                'label': 'Non-rumor'
            },
            {
                'content': '包含"据不完全统计"、"有消息称"、"网传"等模糊表述的信息需要谨慎对待，可能缺乏事实依据。',
                'type': 'guideline', 
                'category': 'language_pattern',
                'label': 'Unverified'
            },
            {
                'content': '明显夸大事实、使用极端词汇、缺乏具体时间地点的信息往往是谣言的特征。',
                'type': 'guideline',
                'category': 'rumor_pattern',
                'label': 'Rumor'
            },
            {
                'content': '科学研究需要同行评议和多次验证，单一研究结果不足以得出绝对结论。',
                'type': 'guideline',
                'category': 'scientific_method',
                'label': 'Non-rumor'
            },
            {
                'content': '社交媒体上流传的未经证实的消息，特别是涉及健康、安全等敏感话题的，需要官方确认。',
                'type': 'guideline',
                'category': 'social_media',
                'label': 'Unverified'
            },
            {
                'content': '谣言往往利用人们的恐惧心理，使用"紧急"、"危险"、"立即"等词汇制造紧迫感。',
                'type': 'guideline',
                'category': 'psychological_manipulation',
                'label': 'Rumor'
            },
            {
                'content': '可以通过查证官方网站、权威媒体报道、专家意见等多个渠道来验证信息真实性。',
                'type': 'verification_method',
                'category': 'fact_checking',
                'label': 'Non-rumor'
            },
            {
                'content': '医学健康信息应该来源于正规医疗机构、医学期刊或执业医师，避免传播未经验证的偏方。',
                'type': 'domain_specific',
                'category': 'health',
                'label': 'Non-rumor'
            },
            {
                'content': '自然灾害预警信息应以气象局、地震局等官方机构发布为准，非官方预测不可信。',
                'type': 'domain_specific',
                'category': 'disaster',
                'label': 'Rumor'
            },
            {
                'content': '经济金融信息应关注发布机构的权威性，避免被虚假投资信息误导。',
                'type': 'domain_specific',
                'category': 'finance',
                'label': 'Unverified'
            }
        ]
        
        self.documents.extend(predefined_docs)
        print(f"📚 添加预定义知识: {len(predefined_docs)} 个文档")
    
    def _add_dataset_knowledge(self):
        """从训练集添加知识 - 只使用训练集构建知识库"""
        try:
            # 只加载训练集数据用于构建知识库
            print("📊 从训练集构建知识库...")
            dataloaders = create_all_dataloaders(
                batch_sizes={'train': 32, 'val': 32, 'test': 32}
            )
            
            # 只从训练集提取样本作为知识库
            train_loader = dataloaders['train']
            sample_count = 0
            
            for batch in train_loader:
                # 使用caption字段（这是实际的文本内容）
                captions = batch.get('caption', batch.get('text', []))
                labels = batch.get('labels', batch.get('label', []))
                
                if hasattr(labels, 'tolist'):
                    labels = labels.tolist()
                
                for caption, label in zip(captions, labels):
                    if caption and len(caption.strip()) > 10:  # 过滤太短的文本
                        label_map = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
                        doc = {
                            'content': caption.strip(),
                            'type': 'train_example',
                            'category': 'dataset_sample',
                            'label': label_map.get(label, 'Unknown'),
                            'source': 'train_set'
                        }
                        self.documents.append(doc)
                        sample_count += 1
                        
                        # 限制数量避免知识库过大，但保持足够的样本
                        if sample_count >= 200:
                            break
                
                if sample_count >= 200:
                    break
            
            print(f"📊 从训练集添加知识: {sample_count} 个样本")
            
            # 显示标签分布
            train_labels = [doc['label'] for doc in self.documents if doc.get('source') == 'train_set']
            from collections import Counter
            label_dist = Counter(train_labels)
            print(f"📊 训练集知识库标签分布: {dict(label_dist)}")
            
        except Exception as e:
            logger.warning(f"从训练集添加知识失败: {e}")
            print(f"⚠️  从训练集构建知识库失败: {e}")
            print("    将继续使用预定义知识")
    
    def _build_index(self):
        """构建文档索引"""
        if not self.documents:
            logger.warning("没有文档可以索引")
            return
        
        # 提取文档内容
        doc_contents = [doc['content'] for doc in self.documents]
        
        # 使用TF-IDF
        print("🔄 计算TF-IDF向量...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None,  # 不使用英文停用词，因为有中文内容
            min_df=1,         # 降低最小文档频率
            max_df=0.95,      # 设置最大文档频率
            token_pattern=r'(?u)\b\w+\b',  # 支持中文字符
            lowercase=True,
            analyzer='word'
        )
        self.embeddings = self.vectorizer.fit_transform(doc_contents)
        print(f"✅ TF-IDF索引构建完成，特征维度: {self.embeddings.shape}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            相关文档列表
        """
        if not self.documents:
            print("⚠️  知识库为空，无法检索")
            return []
        
        try:
            # TF-IDF检索
            if self.vectorizer is None:
                print("⚠️  TF-IDF向量化器未初始化")
                return []
            
            try:
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.embeddings).flatten()
                
                # 调试信息
                print(f"🔍 调试信息: 查询='{query[:30]}...', 相似度范围=[{similarities.min():.4f}, {similarities.max():.4f}]")
                print(f"🔍 查询向量非零元素: {query_vector.nnz}, 文档矩阵形状: {self.embeddings.shape}")
                
                # 如果所有相似度都是0，尝试查看词汇表
                if similarities.max() == 0.0:
                    query_terms = self.vectorizer.get_feature_names_out()
                    query_tokens = query.split()
                    print(f"🔍 查询词汇: {query_tokens[:5]}")
                    print(f"🔍 TF-IDF词汇表大小: {len(query_terms)}")
                    
                    # 检查文档内容
                    if len(self.documents) > 0:
                        print(f"🔍 第一个文档: {self.documents[0]['content'][:50]}...")
                
                # 获取top_k个最相似的文档
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                results = []
                for i, idx in enumerate(top_indices):
                    if idx < len(self.documents):  # 移除相似度>0的限制
                        doc = self.documents[idx].copy()
                        doc['score'] = float(similarities[idx])
                        doc['rank'] = i + 1
                        results.append(doc)
                
                print(f"🔍 找到 {len(results)} 个文档，最高相似度: {similarities[top_indices[0]] if len(top_indices) > 0 else 0:.4f}")
                
                return results
                
            except Exception as e:
                print(f"⚠️  TF-IDF检索失败: {e}")
                # 返回前几个文档作为备用
                results = []
                for i, doc in enumerate(self.documents[:min(top_k, 3)]):
                    doc_copy = doc.copy()
                    doc_copy['score'] = 0.1  # 给一个较低的默认分数
                    doc_copy['rank'] = i + 1
                    results.append(doc_copy)
                return results
                
        except Exception as e:
            logger.error(f"检索失败: {e}")
            print(f"⚠️  检索过程出错: {e}")
            # 返回前几个文档作为备用
            results = []
            for i, doc in enumerate(self.documents[:min(top_k, 3)]):
                doc_copy = doc.copy()
                doc_copy['score'] = 0.1  # 给一个较低的默认分数
                doc_copy['rank'] = i + 1
                results.append(doc_copy)
            return results


class RAGRumorDetector:
    """基于RAG的谣言检测器"""
    
    def __init__(self, 
                 llm_model: Optional[QwenRumorClassifier] = None,
                 knowledge_base: Optional[KnowledgeBase] = None):
        """
        初始化RAG谣言检测器
        
        Args:
            llm_model: LLM模型实例
            knowledge_base: 知识库实例
        """
        # 初始化LLM模型
        if llm_model is None:
            try:
                from models.llms.open_source_llms import create_qwen_classifier
                self.llm_model = create_qwen_classifier(use_lora=True, load_in_4bit=False)
                print("✅ 成功创建Qwen分类器")
            except Exception as e:
                print(f"⚠️  LLM模型初始化失败: {e}")
                self.llm_model = None
        else:
            self.llm_model = llm_model
        
        # 初始化知识库
        if knowledge_base is None:
            self.knowledge_base = KnowledgeBase()
        else:
            self.knowledge_base = knowledge_base
        
        # 初始化提示管理器
        if USE_PROJECT_MODULES:
            self.prompt_manager = PromptManager()
        else:
            self.prompt_manager = None
        
        # 设置输出目录
        if USE_PROJECT_MODULES:
            config_manager = get_config_manager()
            self.output_dir = get_output_path('models', 'llms')
        else:
            self.output_dir = Path('outputs/models/llms')
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🤖 RAG谣言检测器初始化完成")
    
    def create_rag_prompt(self, query_text: str, retrieved_docs: List[Dict]) -> str:
        """
        创建RAG提示
        
        Args:
            query_text: 查询文本
            retrieved_docs: 检索到的文档
            
        Returns:
            RAG提示字符串
        """
        # 构建上下文信息
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:3], 1):  # 最多使用前3个文档
            context_parts.append(
                f"参考{i}：{doc['content']} "
                f"(类型: {doc.get('type', '未知')}, "
                f"标签: {doc.get('label', '未知')}, "
                f"相关度: {doc.get('score', 0):.3f})"
            )
        
        context = "\n".join(context_parts)
        
        # 创建RAG提示
        prompt = f"""基于以下相关信息和你的知识，请分析这段文本是否为谣言。

相关参考信息：
{context}

待分析文本：{query_text}

请根据参考信息和谣言检测知识，从以下三个类别中选择：
1. Non-rumor (非谣言): 内容真实可信，有可靠依据
2. Rumor (谣言): 内容虚假或误导，缺乏事实支撑  
3. Unverified (未验证): 无法确定真伪，需要进一步核实

请说明你的分析理由，并给出最终分类。

分析："""

        return prompt
    
    def retrieve_and_generate(self, query_text: str, 
                            retrieve_top_k: int = 5,
                            use_context: bool = True) -> Dict[str, Any]:
        """
        执行检索增强生成
        
        Args:
            query_text: 查询文本
            retrieve_top_k: 检索文档数量
            use_context: 是否使用检索上下文
            
        Returns:
            分析结果字典
        """
        try:
            # 1. 检索相关文档
            retrieved_docs = self.knowledge_base.retrieve(query_text, retrieve_top_k)
            
            # 2. 生成增强提示
            if use_context and retrieved_docs:
                prompt = self.create_rag_prompt(query_text, retrieved_docs)
                generation_type = "rag_enhanced"
            else:
                # 不使用上下文的标准提示
                if self.prompt_manager:
                    prompt = self.prompt_manager.create_classification_prompt(query_text)
                else:
                    prompt = f"请分析以下文本是否为谣言：\n\n{query_text}\n\n分类："
                generation_type = "standard"
            
            # 3. LLM生成
            if self.llm_model:
                response = self.llm_model.generate_response(
                    prompt, 
                    max_new_tokens=200,
                    temperature=0.3
                )
                
                # 解析响应
                predicted_label = self._parse_rag_response(response)
                confidence = self._calculate_rag_confidence(response, retrieved_docs)
            else:
                response = "模型未加载，无法生成响应"
                predicted_label = 0
                confidence = 0.0
            
            # 4. 构建结果
            result = {
                'query_text': query_text,
                'retrieved_docs': retrieved_docs,
                'retrieved_count': len(retrieved_docs),
                'prompt': prompt,
                'raw_response': response,
                'predicted_label': predicted_label,
                'predicted_class': {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}.get(predicted_label, 'Unknown'),
                'confidence': confidence,
                'generation_type': generation_type,
                'context_used': use_context and len(retrieved_docs) > 0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"RAG处理失败: {e}")
            return {
                'query_text': query_text,
                'error': str(e),
                'predicted_label': 0,
                'predicted_class': 'Non-rumor',
                'confidence': 0.0
            }
    
    def _parse_rag_response(self, response: str) -> int:
        """解析RAG响应"""
        response_lower = response.lower()
        
        # 检查明确的分类标志
        if 'rumor' in response_lower and 'non-rumor' not in response_lower:
            return 1  # Rumor
        elif 'non-rumor' in response_lower:
            return 0  # Non-rumor
        elif 'unverified' in response_lower:
            return 2  # Unverified
        
        # 检查中文标志
        if '谣言' in response_lower and '非谣言' not in response_lower:
            return 1
        elif '非谣言' in response_lower or '真实' in response_lower:
            return 0
        elif '未验证' in response_lower or '不确定' in response_lower:
            return 2
        
        # 检查关键词
        rumor_keywords = ['虚假', '误导', '不实', '错误']
        non_rumor_keywords = ['可信', '真实', '正确', '官方']
        unverified_keywords = ['需要', '核实', '确认', '证实']
        
        for keyword in rumor_keywords:
            if keyword in response_lower:
                return 1
        
        for keyword in non_rumor_keywords:
            if keyword in response_lower:
                return 0
        
        for keyword in unverified_keywords:
            if keyword in response_lower:
                return 2
        
        # 默认返回Non-rumor
        return 0
    
    def _calculate_rag_confidence(self, response: str, retrieved_docs: List[Dict]) -> float:
        """计算RAG置信度"""
        base_confidence = 0.6
        
        # 如果使用了检索上下文
        if retrieved_docs:
            # 检索质量加分
            avg_score = np.mean([doc.get('score', 0) for doc in retrieved_docs])
            base_confidence += avg_score * 0.2
            
            # 一致性检查
            response_lower = response.lower()
            consistent_docs = 0
            
            for doc in retrieved_docs[:3]:  # 检查前3个文档
                doc_label = doc.get('label', '').lower()
                if (('non-rumor' in doc_label and 'non-rumor' in response_lower) or
                    ('rumor' in doc_label and 'rumor' in response_lower and 'non-rumor' not in response_lower) or
                    ('unverified' in doc_label and 'unverified' in response_lower)):
                    consistent_docs += 1
            
            if len(retrieved_docs) > 0:
                consistency_ratio = consistent_docs / min(3, len(retrieved_docs))
                base_confidence += consistency_ratio * 0.15
        
        # 响应质量评估
        if len(response) > 50:  # 详细的分析
            base_confidence += 0.05
        
        if any(keyword in response.lower() for keyword in ['因为', '由于', '根据', 'because', 'since']):
            base_confidence += 0.05  # 包含解释
        
        return min(base_confidence, 1.0)
    
    def batch_analyze(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        批量分析文本
        
        Args:
            texts: 文本列表
            **kwargs: 其他参数
            
        Returns:
            分析结果列表
        """
        results = []
        
        print(f"🔄 开始批量RAG分析 {len(texts)} 个文本...")
        
        for i, text in enumerate(texts):
            print(f"  处理 {i+1}/{len(texts)}: {text[:50]}...")
            result = self.retrieve_and_generate(text, **kwargs)
            results.append(result)
        
        return results
    
    def evaluate_rag_performance(self, test_data: Optional[List[Tuple[str, int]]] = None) -> Dict[str, Any]:
        """
        评估RAG性能
        
        Args:
            test_data: 测试数据 [(text, true_label), ...]
            
        Returns:
            评估结果
        """
        if test_data is None:
            # 使用默认测试数据
            test_data = [
                ("科学家在实验室发现新的治疗方法，已通过同行评议发表", 0),  # Non-rumor
                ("网传某地明天发生大地震，请大家提前撤离", 1),  # Rumor
                ("据消息人士透露，某公司可能进行重组", 2),  # Unverified
                ("权威医学期刊发表研究显示新药物疗效显著", 0),  # Non-rumor
                ("朋友圈流传的养生偏方能治愈所有疾病", 1)   # Rumor
            ]
        
        print(f"📊 开始RAG性能评估，测试样本: {len(test_data)}")
        
        # 测试标准模式
        print("🔍 测试标准模式...")
        standard_results = []
        for text, true_label in test_data:
            result = self.retrieve_and_generate(text, use_context=False)
            standard_results.append({
                'text': text,
                'true_label': true_label,
                'predicted_label': result['predicted_label'],
                'confidence': result['confidence'],
                'correct': result['predicted_label'] == true_label
            })
        
        # 测试RAG模式
        print("🔍 测试RAG增强模式...")
        rag_results = []
        for text, true_label in test_data:
            result = self.retrieve_and_generate(text, use_context=True)
            rag_results.append({
                'text': text,
                'true_label': true_label,
                'predicted_label': result['predicted_label'],
                'confidence': result['confidence'],
                'retrieved_count': result['retrieved_count'],
                'context_used': result['context_used'],
                'correct': result['predicted_label'] == true_label
            })
        
        # 计算指标
        standard_accuracy = np.mean([r['correct'] for r in standard_results])
        rag_accuracy = np.mean([r['correct'] for r in rag_results])
        
        standard_confidence = np.mean([r['confidence'] for r in standard_results])
        rag_confidence = np.mean([r['confidence'] for r in rag_results])
        
        evaluation = {
            'test_samples': len(test_data),
            'standard_mode': {
                'accuracy': standard_accuracy,
                'avg_confidence': standard_confidence,
                'results': standard_results
            },
            'rag_mode': {
                'accuracy': rag_accuracy,
                'avg_confidence': rag_confidence,
                'avg_retrieved_docs': np.mean([r['retrieved_count'] for r in rag_results]),
                'context_usage_rate': np.mean([r['context_used'] for r in rag_results]),
                'results': rag_results
            },
            'improvement': {
                'accuracy_gain': rag_accuracy - standard_accuracy,
                'confidence_gain': rag_confidence - standard_confidence
            }
        }
        
        print(f"✅ RAG评估完成:")
        print(f"   标准模式准确率: {standard_accuracy:.4f}")
        print(f"   RAG模式准确率: {rag_accuracy:.4f}")
        print(f"   准确率提升: {evaluation['improvement']['accuracy_gain']:+.4f}")
        
        return evaluation
    
    def save_knowledge_base(self, save_path: Optional[str] = None):
        """保存知识库"""
        if save_path is None:
            save_path = self.output_dir / "rag_knowledge_base.json"
        
        # 保存文档数据（不包含嵌入向量）
        knowledge_data = {
            'documents': self.knowledge_base.documents,
            'knowledge_sources': self.knowledge_base.knowledge_sources,
            'total_documents': len(self.knowledge_base.documents),
            'embedding_model': 'all-MiniLM-L6-v2' if self.knowledge_base.embedding_model else 'TF-IDF'
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 知识库已保存到: {save_path}")


def demo_rag_integration():
    """演示RAG集成功能"""
    print("🤖 RAG集成功能演示")
    print("=" * 60)
    
    try:
        # 创建RAG检测器
        print("🔄 初始化RAG检测器...")
        rag_detector = RAGRumorDetector()
        
        # 测试单个文本分析
        test_texts = [
            "中国科学院发布最新研究成果，在量子计算领域取得重大突破",
            "网传某市明天将发生8级大地震，请大家做好撤离准备",
            "据业内人士透露，某互联网公司可能进行大规模裁员",
            "世界卫生组织确认新冠疫苗对变异株具有良好保护效果",
            "朋友圈热传的偏方能够完全治愈糖尿病，无需药物治疗"
        ]
        
        print(f"\n🔍 单文本RAG分析测试:")
        for i, text in enumerate(test_texts[:3], 1):  # 测试前3个
            print(f"\n--- 测试 {i} ---")
            print(f"文本: {text}")
            
            # RAG分析
            result = rag_detector.retrieve_and_generate(text)
            print(f"预测: {result['predicted_class']} (置信度: {result['confidence']:.3f})")
            print(f"检索到 {result['retrieved_count']} 个相关文档")
            print(f"使用上下文: {result['context_used']}")
            
            # 显示检索到的文档
            if result.get('retrieved_docs'):
                print("相关参考:")
                for j, doc in enumerate(result['retrieved_docs'][:2], 1):
                    print(f"  {j}. {doc['content'][:80]}... (相关度: {doc.get('score', 0):.3f})")
        
        # 性能评估
        print(f"\n📊 RAG性能评估:")
        evaluation = rag_detector.evaluate_rag_performance()
        
        print(f"✅ 评估结果:")
        print(f"   测试样本数: {evaluation['test_samples']}")
        print(f"   标准模式准确率: {evaluation['standard_mode']['accuracy']:.4f}")
        print(f"   RAG模式准确率: {evaluation['rag_mode']['accuracy']:.4f}")
        print(f"   准确率提升: {evaluation['improvement']['accuracy_gain']:+.4f}")
        print(f"   平均检索文档数: {evaluation['rag_mode']['avg_retrieved_docs']:.1f}")
        
        # 保存知识库
        print(f"\n💾 保存知识库...")
        rag_detector.save_knowledge_base()
        
        # 对比分析：标准模式 vs RAG模式
        print(f"\n🔬 对比分析示例:")
        comparison_text = "专家称某地区可能发生地质灾害，建议居民注意防范"
        
        # 标准模式
        standard_result = rag_detector.retrieve_and_generate(comparison_text, use_context=False)
        print(f"标准模式:")
        print(f"  预测: {standard_result['predicted_class']} (置信度: {standard_result['confidence']:.3f})")
        
        # RAG模式
        rag_result = rag_detector.retrieve_and_generate(comparison_text, use_context=True)
        print(f"RAG模式:")
        print(f"  预测: {rag_result['predicted_class']} (置信度: {rag_result['confidence']:.3f})")
        print(f"  检索文档: {rag_result['retrieved_count']} 个")
        
        # 知识库统计
        print(f"\n📚 知识库统计:")
        kb = rag_detector.knowledge_base
        doc_types = {}
        doc_labels = {}
        
        for doc in kb.documents:
            doc_type = doc.get('type', 'unknown')
            doc_label = doc.get('label', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            doc_labels[doc_label] = doc_labels.get(doc_label, 0) + 1
        
        print(f"  总文档数: {len(kb.documents)}")
        print(f"  文档类型分布: {doc_types}")
        print(f"  标签分布: {doc_labels}")
        
        print(f"\n✅ RAG集成功能演示完成!")
        
    except Exception as e:
        print(f"❌ RAG演示失败: {e}")
        import traceback
        traceback.print_exc()


def create_rag_detector(use_existing_llm: bool = False) -> RAGRumorDetector:
    """
    创建RAG检测器的便捷函数
    
    Args:
        use_existing_llm: 是否使用已存在的LLM模型
        
    Returns:
        RAG检测器实例
    """
    print("🚀 创建RAG谣言检测器...")
    
    llm_model = None
    if use_existing_llm:
        try:
            from models.llms.open_source_llms import create_qwen_classifier
            llm_model = create_qwen_classifier(use_lora=True, load_in_4bit=False)
        except Exception as e:
            print(f"⚠️  LLM模型创建失败: {e}")
    
    detector = RAGRumorDetector(llm_model=llm_model)
    return detector


class AdvancedRAGFeatures:
    """高级RAG功能"""
    
    def __init__(self, rag_detector: RAGRumorDetector):
        """
        初始化高级RAG功能
        
        Args:
            rag_detector: RAG检测器实例
        """
        self.rag_detector = rag_detector
        self.query_history = []
        self.feedback_data = []
    
    def multi_query_rag(self, query_text: str, query_variations: int = 3) -> Dict[str, Any]:
        """
        多查询RAG：生成多个查询变体来获取更全面的检索结果
        
        Args:
            query_text: 原始查询文本
            query_variations: 查询变体数量
            
        Returns:
            综合分析结果
        """
        print(f"🔄 执行多查询RAG分析...")
        
        # 生成查询变体
        query_variants = self._generate_query_variants(query_text, query_variations)
        
        # 对每个变体进行检索
        all_results = []
        all_retrieved_docs = []
        
        for i, variant in enumerate(query_variants):
            print(f"  处理查询变体 {i+1}: {variant[:50]}...")
            result = self.rag_detector.retrieve_and_generate(variant)
            all_results.append(result)
            all_retrieved_docs.extend(result.get('retrieved_docs', []))
        
        # 去重和重新排序文档
        unique_docs = self._deduplicate_documents(all_retrieved_docs)
        
        # 综合分析
        final_prediction = self._ensemble_predictions([r['predicted_label'] for r in all_results])
        avg_confidence = np.mean([r['confidence'] for r in all_results])
        
        return {
            'original_query': query_text,
            'query_variants': query_variants,
            'individual_results': all_results,
            'unique_retrieved_docs': unique_docs,
            'ensemble_prediction': final_prediction,
            'ensemble_class': {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}.get(final_prediction, 'Unknown'),
            'average_confidence': avg_confidence,
            'total_unique_docs': len(unique_docs)
        }
    
    def _generate_query_variants(self, query_text: str, num_variants: int) -> List[str]:
        """生成查询变体"""
        variants = [query_text]  # 原始查询
        
        # 基于关键词的变体
        keywords = query_text.split()[:5]  # 取前5个词
        if len(keywords) > 2:
            variants.append(" ".join(keywords[:3]))  # 前3个关键词
            variants.append(" ".join(keywords[-3:]))  # 后3个关键词
        
        # 基于问题类型的变体
        if '网传' in query_text or 'rumor' in query_text.lower():
            variants.append(f"这个信息是否可信：{query_text}")
        
        if '专家' in query_text or 'expert' in query_text.lower():
            variants.append(f"权威性分析：{query_text}")
        
        # 返回所需数量的变体
        return variants[:num_variants]
    
    def _deduplicate_documents(self, docs: List[Dict]) -> List[Dict]:
        """去重文档"""
        seen_contents = set()
        unique_docs = []
        
        for doc in docs:
            content = doc['content']
            if content not in seen_contents:
                seen_contents.add(content)
                unique_docs.append(doc)
        
        # 按相关度排序
        unique_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
        return unique_docs[:10]  # 返回前10个最相关的
    
    def _ensemble_predictions(self, predictions: List[int]) -> int:
        """集成预测结果"""
        if not predictions:
            return 0
        
        # 投票法
        from collections import Counter
        vote_counts = Counter(predictions)
        return vote_counts.most_common(1)[0][0]
    
    def iterative_rag(self, query_text: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        迭代式RAG：基于初始结果进行迭代优化
        
        Args:
            query_text: 查询文本
            max_iterations: 最大迭代次数
            
        Returns:
            迭代分析结果
        """
        print(f"🔄 执行迭代式RAG分析...")
        
        iteration_results = []
        current_query = query_text
        
        for i in range(max_iterations):
            print(f"  迭代 {i+1}/{max_iterations}")
            
            # 当前迭代的RAG分析
            result = self.rag_detector.retrieve_and_generate(current_query)
            iteration_results.append({
                'iteration': i + 1,
                'query': current_query,
                'result': result
            })
            
            # 如果置信度足够高，停止迭代
            if result['confidence'] > 0.8:
                print(f"  高置信度达成，停止迭代")
                break
            
            # 基于当前结果优化下一次查询
            if i < max_iterations - 1:
                current_query = self._refine_query_from_result(query_text, result)
        
        # 选择最佳结果
        best_result = max(iteration_results, key=lambda x: x['result']['confidence'])
        
        return {
            'original_query': query_text,
            'iterations': iteration_results,
            'best_iteration': best_result['iteration'],
            'best_result': best_result['result'],
            'confidence_progression': [ir['result']['confidence'] for ir in iteration_results]
        }
    
    def _refine_query_from_result(self, original_query: str, result: Dict) -> str:
        """基于结果优化查询"""
        # 如果检索到相关文档，提取关键概念
        if result.get('retrieved_docs'):
            doc_contents = [doc['content'] for doc in result['retrieved_docs'][:2]]
            combined_content = " ".join(doc_contents)
            
            # 简单的关键词提取（实际应用中可以使用更复杂的NLP技术）
            important_words = []
            for word in combined_content.split():
                if len(word) > 3 and word not in ['这个', '那个', '可以', '应该']:
                    important_words.append(word)
            
            if important_words:
                refined_query = f"{original_query} {' '.join(important_words[:3])}"
                return refined_query
        
        return original_query
    
    def add_user_feedback(self, query: str, predicted_label: int, true_label: int, 
                         feedback_type: str = "correction"):
        """
        添加用户反馈用于改进RAG系统
        
        Args:
            query: 查询文本
            predicted_label: 预测标签
            true_label: 真实标签
            feedback_type: 反馈类型
        """
        feedback = {
            'query': query,
            'predicted_label': predicted_label,
            'true_label': true_label,
            'feedback_type': feedback_type,
            'timestamp': np.datetime64('now').astype(str),
            'is_correct': predicted_label == true_label
        }
        
        self.feedback_data.append(feedback)
        print(f"📝 添加用户反馈: {'正确' if feedback['is_correct'] else '错误'}")
    
    def analyze_feedback(self) -> Dict[str, Any]:
        """分析用户反馈"""
        if not self.feedback_data:
            return {'message': '暂无反馈数据'}
        
        total_feedback = len(self.feedback_data)
        correct_predictions = sum(1 for f in self.feedback_data if f['is_correct'])
        accuracy = correct_predictions / total_feedback
        
        # 按标签分析
        label_analysis = {}
        for label in [0, 1, 2]:
            label_feedback = [f for f in self.feedback_data if f['true_label'] == label]
            if label_feedback:
                label_correct = sum(1 for f in label_feedback if f['is_correct'])
                label_analysis[label] = {
                    'total': len(label_feedback),
                    'correct': label_correct,
                    'accuracy': label_correct / len(label_feedback)
                }
        
        return {
            'total_feedback': total_feedback,
            'overall_accuracy': accuracy,
            'label_analysis': label_analysis,
            'recent_feedback': self.feedback_data[-5:] if len(self.feedback_data) >= 5 else self.feedback_data
        }


def demo_advanced_rag_features():
    """演示高级RAG功能"""
    print("🔬 高级RAG功能演示")
    print("=" * 60)
    
    try:
        # 创建RAG检测器
        rag_detector = create_rag_detector(use_existing_llm=False)
        advanced_rag = AdvancedRAGFeatures(rag_detector)
        
        # 测试多查询RAG
        print("\n🔍 多查询RAG测试:")
        test_query = "网传某地发生重大地质灾害，专家建议撤离"
        multi_result = advanced_rag.multi_query_rag(test_query, query_variations=3)
        
        print(f"原始查询: {multi_result['original_query']}")
        print(f"查询变体: {multi_result['query_variants']}")
        print(f"集成预测: {multi_result['ensemble_class']} (置信度: {multi_result['average_confidence']:.3f})")
        print(f"检索到唯一文档: {multi_result['total_unique_docs']} 个")
        
        # 测试迭代RAG
        print("\n🔄 迭代RAG测试:")
        iterative_result = advanced_rag.iterative_rag(test_query, max_iterations=2)
        
        print(f"迭代次数: {len(iterative_result['iterations'])}")
        print(f"最佳迭代: 第{iterative_result['best_iteration']}次")
        print(f"置信度变化: {iterative_result['confidence_progression']}")
        print(f"最终预测: {iterative_result['best_result']['predicted_class']}")
        
        # 模拟用户反馈
        print("\n📝 用户反馈测试:")
        test_cases = [
            ("官方发布的权威声明", 0, 0),  # 正确预测
            ("网上流传的未证实消息", 1, 2),  # 错误预测
            ("专家学者的研究成果", 0, 0)   # 正确预测
        ]
        
        for query, predicted, true in test_cases:
            advanced_rag.add_user_feedback(query, predicted, true)
        
        feedback_analysis = advanced_rag.analyze_feedback()
        print(f"反馈分析:")
        print(f"  总反馈数: {feedback_analysis['total_feedback']}")
        print(f"  整体准确率: {feedback_analysis['overall_accuracy']:.3f}")
        
        print(f"\n✅ 高级RAG功能演示完成!")
        
    except Exception as e:
        print(f"❌ 高级RAG演示失败: {e}")
        import traceback
        traceback.print_exc()


# 主执行代码
if __name__ == "__main__":
    print("🚀 RAG集成模块测试")
    print("=" * 60)
    
    try:
        # 基础RAG功能演示
        demo_rag_integration()
        
        print("\n" + "=" * 60)
        
        # 高级RAG功能演示
        demo_advanced_rag_features()
        
        print("\n✅ RAG集成模块测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)