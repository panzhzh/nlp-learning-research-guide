#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/traditional/ml_classifiers.py

"""
传统机器学习分类器模块 - 修复版本
修复数据加载器调用问题
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 快速路径设置
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入项目模块
try:
    from datasets.data_loaders import create_all_dataloaders
    from utils.config_manager import get_config_manager, get_output_path
    from preprocessing.text_processing import TextProcessor
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块")
except ImportError as e:
    print(f"⚠️  导入项目模块失败: {e}")
    USE_PROJECT_MODULES = False

import logging
logger = logging.getLogger(__name__)


class MLClassifierTrainer:
    """
    传统机器学习分类器训练器
    支持多种算法和自动特征提取
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化训练器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.models = {}
        self.results = {}
        self.vectorizers = {}
        
        # 初始化文本处理器
        if USE_PROJECT_MODULES:
            self.text_processor = TextProcessor(language='mixed')
        else:
            self.text_processor = None
        
        # 设置输出目录
        if USE_PROJECT_MODULES:
            config_manager = get_config_manager()
            self.output_dir = get_output_path('models', 'traditional')
        else:
            self.output_dir = Path('outputs/models/traditional')
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 标签映射
        self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
        
        print(f"🤖 传统ML分类器训练器初始化完成")
        print(f"   数据目录: {self.data_dir}")
        print(f"   输出目录: {self.output_dir}")
    
    def load_data(self) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        加载MR2数据集 - 修复版本
        
        Returns:
            数据字典 {split: (texts, labels)}
        """
        print("📚 加载MR2数据集...")
        
        if USE_PROJECT_MODULES:
            try:
                # 修复：使用正确的函数调用方式
                dataloaders = create_all_dataloaders(
                    batch_sizes={'train': 32, 'val': 32, 'test': 32}
                )
                
                data = {}
                for split, dataloader in dataloaders.items():
                    texts = []
                    labels = []
                    
                    for batch in dataloader:
                        # 提取文本和标签
                        if 'text' in batch:
                            texts.extend(batch['text'])
                        elif 'caption' in batch:  # MR2数据集使用caption字段
                            texts.extend(batch['caption'])
                        
                        if 'labels' in batch:
                            labels.extend(batch['labels'].tolist())
                        elif 'label' in batch:
                            labels.extend(batch['label'])
                    
                    data[split] = (texts, labels)
                    print(f"✅ 加载 {split}: {len(texts)} 样本")
                
                return data
                
            except Exception as e:
                print(f"❌ 使用项目数据加载器失败: {e}")
                return self._create_demo_data()
        else:
            return self._create_demo_data()
    
    def _create_demo_data(self) -> Dict[str, Tuple[List[str], List[int]]]:
        """创建演示数据"""
        print("🔧 创建演示数据...")
        
        demo_texts = [
            "这是一个关于科技进步的真实新闻报道",
            "This is a fake news about celebrity scandal",
            "未经证实的传言需要进一步调查验证",
            "Breaking: Major breakthrough in AI technology announced",
            "网传某地发生重大事故，官方尚未确认",
            "Scientists discover new species in deep ocean",
            "谣传某公司倒闭，实际情况有待核实",
            "Weather alert: Severe storm approaching coastal areas",
            "社交媒体流传的未证实消息引发关注",
            "Economic indicators show positive growth trends"
        ]
        
        demo_labels = [0, 1, 2, 0, 2, 0, 1, 0, 2, 0]  # 对应 Non-rumor, Rumor, Unverified
        
        # 创建更多样本以便训练
        extended_texts = demo_texts * 5  # 重复以创建更多样本
        extended_labels = demo_labels * 5
        
        # 按比例分割数据
        total_size = len(extended_texts)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        
        return {
            'train': (extended_texts[:train_size], extended_labels[:train_size]),
            'val': (extended_texts[train_size:train_size+val_size], 
                   extended_labels[train_size:train_size+val_size]),
            'test': (extended_texts[train_size+val_size:], 
                    extended_labels[train_size+val_size:])
        }
    
    def create_feature_extractors(self):
        """创建特征提取器"""
        print("🔧 设置特征提取器...")
        
        # TF-IDF向量化器
        self.vectorizers['tfidf'] = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None,  # 保留停用词，因为是多语言
            lowercase=True,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # 词袋模型向量化器
        self.vectorizers['count'] = CountVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words=None,
            lowercase=True,
            min_df=2,
            max_df=0.95
        )
        
        print("✅ 特征提取器设置完成")
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        预处理文本
        
        Args:
            texts: 原始文本列表
            
        Returns:
            处理后的文本列表
        """
        if self.text_processor:
            # 使用项目的文本处理器
            processed_texts = []
            for text in texts:
                # 清洗文本
                cleaned_text = self.text_processor.clean_text(text)
                # 分词并重新组合
                tokens = self.text_processor.tokenize(cleaned_text)
                processed_text = ' '.join(tokens) if tokens else cleaned_text
                processed_texts.append(processed_text)
            return processed_texts
        else:
            # 简单的文本清理
            import re
            processed_texts = []
            for text in texts:
                # 基本清理
                text = re.sub(r'http\S+', '', text)  # 移除URL
                text = re.sub(r'@\w+', '', text)    # 移除@提及
                text = re.sub(r'#\w+', '', text)    # 移除#标签
                text = re.sub(r'\s+', ' ', text)    # 标准化空白
                text = text.strip()
                processed_texts.append(text)
            return processed_texts
    
    def create_models(self):
        """创建机器学习模型"""
        print("🤖 创建机器学习模型...")
        
        # SVM分类器
        self.models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # 随机森林分类器
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        # 朴素贝叶斯分类器
        self.models['naive_bayes'] = MultinomialNB(
            alpha=1.0,
            fit_prior=True
        )
        
        # 逻辑回归分类器
        self.models['logistic_regression'] = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            multi_class='ovr'
        )
        
        print(f"✅ 创建了 {len(self.models)} 个模型")
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, 
                          y_train: np.ndarray, X_val: np.ndarray, 
                          y_val: np.ndarray) -> Dict[str, Any]:
        """
        训练单个模型
        
        Args:
            model_name: 模型名称
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            训练结果字典
        """
        print(f"🏋️ 训练 {model_name} 模型...")
        
        model = self.models[model_name]
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # 计算指标
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        # 详细分类报告
        val_report = classification_report(y_val, y_val_pred, 
                                         target_names=list(self.label_mapping.values()),
                                         output_dict=True)
        
        result = {
            'model_name': model_name,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'classification_report': val_report,
            'confusion_matrix': confusion_matrix(y_val, y_val_pred).tolist()
        }
        
        print(f"✅ {model_name} 训练完成:")
        print(f"   训练准确率: {train_accuracy:.4f}")
        print(f"   验证准确率: {val_accuracy:.4f}")
        print(f"   验证F1分数: {val_f1:.4f}")
        
        return result
    
    def hyperparameter_tuning(self, model_name: str, X_train: np.ndarray, 
                             y_train: np.ndarray) -> Dict[str, Any]:
        """
        超参数调优
        
        Args:
            model_name: 模型名称
            X_train: 训练特征
            y_train: 训练标签
            
        Returns:
            最佳参数字典
        """
        print(f"🔍 进行 {model_name} 超参数调优...")
        
        # 定义参数网格
        param_grids = {
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        }
        
        if model_name not in param_grids:
            print(f"⚠️  {model_name} 不支持超参数调优")
            return {}
        
        # 执行网格搜索
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ {model_name} 最佳参数: {grid_search.best_params_}")
        print(f"   最佳分数: {grid_search.best_score_:.4f}")
        
        # 更新模型为最佳参数
        self.models[model_name] = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def train_all_models(self, use_hyperparameter_tuning: bool = False):
        """
        训练所有模型
        
        Args:
            use_hyperparameter_tuning: 是否使用超参数调优
        """
        print("🚀 开始训练所有模型...")
        
        # 加载数据
        data = self.load_data()
        
        # 创建特征提取器和模型
        self.create_feature_extractors()
        self.create_models()
        
        # 预处理文本
        train_texts = self.preprocess_texts(data['train'][0])
        val_texts = self.preprocess_texts(data['val'][0])
        test_texts = self.preprocess_texts(data['test'][0])
        
        train_labels = np.array(data['train'][1])
        val_labels = np.array(data['val'][1])
        test_labels = np.array(data['test'][1])
        
        # 使用TF-IDF特征提取
        print("🔧 提取TF-IDF特征...")
        X_train_tfidf = self.vectorizers['tfidf'].fit_transform(train_texts)
        X_val_tfidf = self.vectorizers['tfidf'].transform(val_texts)
        X_test_tfidf = self.vectorizers['tfidf'].transform(test_texts)
        
        print(f"✅ 特征提取完成，特征维度: {X_train_tfidf.shape[1]}")
        
        # 训练每个模型
        for model_name in self.models.keys():
            print(f"\n{'='*50}")
            print(f"训练模型: {model_name.upper()}")
            print(f"{'='*50}")
            
            # 超参数调优（可选）
            if use_hyperparameter_tuning:
                tuning_results = self.hyperparameter_tuning(
                    model_name, X_train_tfidf, train_labels
                )
            else:
                tuning_results = {}
            
            # 训练模型
            train_result = self.train_single_model(
                model_name, X_train_tfidf, train_labels,
                X_val_tfidf, val_labels
            )
            
            # 测试集评估
            model = self.models[model_name]
            test_pred = model.predict(X_test_tfidf)
            test_accuracy = accuracy_score(test_labels, test_pred)
            test_f1 = f1_score(test_labels, test_pred, average='macro')
            
            # 保存结果
            self.results[model_name] = {
                **train_result,
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'hyperparameter_tuning': tuning_results,
                'feature_dim': X_train_tfidf.shape[1]
            }
            
            print(f"🎯 {model_name} 最终结果:")
            print(f"   测试准确率: {test_accuracy:.4f}")
            print(f"   测试F1分数: {test_f1:.4f}")
        
        # 保存模型和结果
        self.save_models_and_results()
        
        # 显示最终对比
        self.print_model_comparison()
    
    def save_models_and_results(self):
        """保存训练好的模型和结果"""
        print("\n💾 保存模型和结果...")
        
        # 保存每个模型
        for model_name, model in self.models.items():
            model_file = self.output_dir / f'{model_name}_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ 保存模型: {model_file}")
        
        # 保存特征提取器
        for vec_name, vectorizer in self.vectorizers.items():
            vec_file = self.output_dir / f'{vec_name}_vectorizer.pkl'
            with open(vec_file, 'wb') as f:
                pickle.dump(vectorizer, f)
            print(f"✅ 保存特征提取器: {vec_file}")
        
        # 保存结果
        results_file = self.output_dir / 'training_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"✅ 保存训练结果: {results_file}")
        
        # 保存模型比较
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train_Accuracy': result['train_accuracy'],
                'Val_Accuracy': result['val_accuracy'],
                'Test_Accuracy': result['test_accuracy'],
                'Train_F1': result['train_f1'],
                'Val_F1': result['val_f1'],
                'Test_F1': result['test_f1']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = self.output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        print(f"✅ 保存模型比较: {comparison_file}")
    
    def print_model_comparison(self):
        """打印模型比较结果"""
        print(f"\n📊 {'='*60}")
        print("模型性能对比")
        print(f"{'='*60}")
        
        # 创建比较表格
        headers = ['模型', '训练准确率', '验证准确率', '测试准确率', '测试F1']
        
        print(f"{'模型':<15} {'训练准确率':<10} {'验证准确率':<10} {'测试准确率':<10} {'测试F1':<10}")
        print("-" * 60)
        
        # 按测试F1分数排序
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['test_f1'], 
                              reverse=True)
        
        for model_name, result in sorted_results:
            print(f"{model_name:<15} "
                  f"{result['train_accuracy']:<10.4f} "
                  f"{result['val_accuracy']:<10.4f} "
                  f"{result['test_accuracy']:<10.4f} "
                  f"{result['test_f1']:<10.4f}")
        
        # 找出最佳模型
        best_model = sorted_results[0]
        print(f"\n🏆 最佳模型: {best_model[0]}")
        print(f"   测试F1分数: {best_model[1]['test_f1']:.4f}")
        print(f"   测试准确率: {best_model[1]['test_accuracy']:.4f}")
    
    def load_trained_model(self, model_name: str, vectorizer_name: str = 'tfidf'):
        """
        加载训练好的模型
        
        Args:
            model_name: 模型名称
            vectorizer_name: 特征提取器名称
            
        Returns:
            (model, vectorizer) 元组
        """
        model_file = self.output_dir / f'{model_name}_model.pkl'
        vec_file = self.output_dir / f'{vectorizer_name}_vectorizer.pkl'
        
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        if not vec_file.exists():
            raise FileNotFoundError(f"特征提取器文件不存在: {vec_file}")
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        with open(vec_file, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    
    def predict_single_text(self, text: str, model_name: str = 'best') -> Dict[str, Any]:
        """
        对单个文本进行预测
        
        Args:
            text: 待预测文本
            model_name: 模型名称，'best'表示使用最佳模型
            
        Returns:
            预测结果字典
        """
        # 如果指定使用最佳模型，找出最佳模型
        if model_name == 'best':
            if not self.results:
                raise ValueError("没有训练结果，请先训练模型")
            
            best_model_name = max(self.results.keys(), 
                                key=lambda x: self.results[x]['test_f1'])
            model_name = best_model_name
        
        # 加载模型和特征提取器
        model, vectorizer = self.load_trained_model(model_name)
        
        # 预处理文本
        processed_text = self.preprocess_texts([text])[0]
        
        # 特征提取
        features = vectorizer.transform([processed_text])
        
        # 预测
        prediction = model.predict(features)[0]
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
        else:
            probabilities = None
        
        result = {
            'original_text': text,
            'processed_text': processed_text,
            'prediction': int(prediction),
            'prediction_label': self.label_mapping.get(prediction, 'Unknown'),
            'model_used': model_name
        }
        
        if probabilities is not None:
            result['probabilities'] = {
                self.label_mapping.get(i, f'Class_{i}'): float(prob)
                for i, prob in enumerate(probabilities)
            }
        
        return result


def main():
    """主函数，演示训练流程"""
    print("🚀 传统机器学习分类器训练演示")
    
    # 创建训练器
    trainer = MLClassifierTrainer(data_dir="data")
    
    # 训练所有模型
    trainer.train_all_models(use_hyperparameter_tuning=False)  # 设置为True启用超参数调优
    
    # 演示预测
    print("\n🔮 演示预测功能:")
    test_texts = [
        "这是一个关于新技术突破的真实新闻",
        "网传某地发生重大事故，官方尚未确认",
        "This might be fake news about celebrities"
    ]
    
    for text in test_texts:
        try:
            result = trainer.predict_single_text(text)
            print(f"\n文本: {text}")
            print(f"预测: {result['prediction_label']} (置信度: {max(result.get('probabilities', {0: 0}).values()):.3f})")
        except Exception as e:
            print(f"预测失败: {e}")
    
    print("\n✅ 训练演示完成!")
    print(f"📁 模型和结果已保存到: {trainer.output_dir}")


if __name__ == "__main__":
    main()