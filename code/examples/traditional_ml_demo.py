#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# examples/traditional_ml_demo.py

"""
传统机器学习模型训练演示 - 修复版本
专门针对传统ML模型的训练，使用真实MR2数据
修复了模块导入和数据加载问题
"""

import sys
import os
import json
import time
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pickle

warnings.filterwarnings('ignore')

# 添加项目路径
current_file = Path(__file__).resolve()
code_root = current_file.parent.parent
sys.path.append(str(code_root))

print("🤖 MR2传统机器学习模型训练演示")
print("="*50)

def check_real_data():
    """检查真实数据是否存在"""
    print("\n📁 检查真实数据文件...")
    
    data_dir = code_root / "data"
    required_files = [
        "dataset_items_train.json",
        "dataset_items_val.json", 
        "dataset_items_test.json"
    ]
    
    existing_files = []
    
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            file_size = file_path.stat().st_size
            existing_files.append((file_name, file_size))
            print(f"   ✅ {file_name} ({file_size/1024:.1f} KB)")
        else:
            print(f"   ❌ {file_name} - 文件不存在")
            return False
    
    print(f"\n✅ 所有数据文件就绪! 共 {len(existing_files)} 个文件")
    return True

def load_real_mr2_data():
    """直接加载真实MR2数据"""
    print("\n📚 加载真实MR2数据...")
    
    data_dir = code_root / "data"
    
    # 直接读取JSON文件
    datasets = {}
    total_samples = 0
    
    for split in ['train', 'val', 'test']:
        file_path = data_dir / f'dataset_items_{split}.json'
        
        print(f"📂 读取 {split} 数据: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 提取文本和标签
        texts = []
        labels = []
        
        for item_id, item_data in raw_data.items():
            if 'caption' in item_data and 'label' in item_data:
                texts.append(item_data['caption'])
                labels.append(item_data['label'])
        
        datasets[split] = (texts, labels)
        total_samples += len(texts)
        
        print(f"   ✅ {split}: {len(texts)} 样本")
        
        # 显示标签分布
        from collections import Counter
        label_dist = Counter(labels)
        print(f"   📊 标签分布: {dict(label_dist)}")
    
    print(f"\n📊 总样本数: {total_samples}")
    return datasets

def preprocess_text_simple(text):
    """简单文本预处理"""
    if not isinstance(text, str):
        return ""
    
    # 基本清理
    import re
    text = re.sub(r'http\S+', '', text)  # 移除URL
    text = re.sub(r'@\w+', '', text)     # 移除@提及
    text = re.sub(r'#\w+', '', text)     # 移除#标签
    text = re.sub(r'\s+', ' ', text)     # 标准化空白
    text = text.strip().lower()
    
    return text

def train_traditional_models(datasets):
    """训练传统机器学习模型"""
    print("\n🤖 开始训练传统机器学习模型...")
    
    # 合并数据
    train_texts, train_labels = datasets['train']
    val_texts, val_labels = datasets['val']
    test_texts, test_labels = datasets['test']
    
    # 预处理文本
    print("🔧 预处理文本数据...")
    train_texts = [preprocess_text_simple(text) for text in train_texts]
    val_texts = [preprocess_text_simple(text) for text in val_texts]
    test_texts = [preprocess_text_simple(text) for text in test_texts]
    
    # 特征提取
    print("🔧 提取TF-IDF特征...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words=None  # 保留停用词，因为是多语言
    )
    
    # 训练特征提取器
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"✅ 特征维度: {X_train.shape[1]}")
    print(f"✅ 训练集: {X_train.shape[0]} 样本")
    print(f"✅ 验证集: {X_val.shape[0]} 样本")
    print(f"✅ 测试集: {X_test.shape[0]} 样本")
    
    # 转换标签
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)
    y_test = np.array(test_labels)
    
    # 定义模型
    models = {
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    }
    
    results = {}
    label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
    
    # 训练每个模型
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"训练模型: {model_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 训练
        print(f"🏋️ 开始训练 {model_name}...")
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # 预测
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # 计算指标
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        train_f1 = f1_score(y_train, train_pred, average='macro')
        val_f1 = f1_score(y_val, val_pred, average='macro')
        test_f1 = f1_score(y_test, test_pred, average='macro')
        
        # 详细报告
        test_report = classification_report(
            y_test, test_pred,
            target_names=list(label_mapping.values()),
            output_dict=True
        )
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, test_pred)
        
        results[model_name] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'training_time': training_time,
            'classification_report': test_report,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"✅ {model_name} 训练完成 (耗时: {training_time:.2f}秒)")
        print(f"   训练准确率: {train_acc:.4f}")
        print(f"   验证准确率: {val_acc:.4f}")
        print(f"   测试准确率: {test_acc:.4f}")
        print(f"   测试F1分数: {test_f1:.4f}")
    
    return results, vectorizer, models

def save_results(results, vectorizer, models):
    """保存训练结果"""
    print("\n💾 保存训练结果...")
    
    # 创建输出目录
    output_dir = code_root / 'outputs' / 'traditional_ml_demo'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    for model_name, model in models.items():
        model_file = output_dir / f'{model_name.lower().replace(" ", "_")}_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ 保存模型: {model_file}")
    
    # 保存特征提取器
    vectorizer_file = output_dir / 'tfidf_vectorizer.pkl'
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✅ 保存特征提取器: {vectorizer_file}")
    
    # 保存结果JSON
    results_file = output_dir / 'training_results.json'
    
    # 转换numpy类型为可序列化类型
    serializable_results = {}
    for model_name, result in results.items():
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            else:
                serializable_result[key] = value
        serializable_results[model_name] = serializable_result
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 保存结果: {results_file}")
    
    # 保存CSV比较
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Train_Accuracy': result['train_accuracy'],
            'Val_Accuracy': result['val_accuracy'],
            'Test_Accuracy': result['test_accuracy'],
            'Train_F1': result['train_f1'],
            'Val_F1': result['val_f1'],
            'Test_F1': result['test_f1'],
            'Training_Time': result['training_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = output_dir / 'model_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"✅ 保存比较: {comparison_file}")
    
    return output_dir

def display_final_results(results):
    """显示最终结果"""
    print(f"\n📊 {'='*80}")
    print("传统机器学习模型性能对比")
    print(f"{'='*80}")
    
    print(f"{'模型':<18} {'训练准确率':<10} {'验证准确率':<10} {'测试准确率':<10} {'测试F1':<10} {'训练时间':<10}")
    print("-" * 80)
    
    # 按测试F1分数排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_f1'], reverse=True)
    
    for model_name, result in sorted_results:
        print(f"{model_name:<18} "
              f"{result['train_accuracy']:<10.4f} "
              f"{result['val_accuracy']:<10.4f} "
              f"{result['test_accuracy']:<10.4f} "
              f"{result['test_f1']:<10.4f} "
              f"{result['training_time']:<10.2f}")
    
    # 最佳模型
    best_model_name, best_result = sorted_results[0]
    print(f"\n🏆 最佳模型: {best_model_name}")
    print(f"   测试准确率: {best_result['test_accuracy']:.4f}")
    print(f"   测试F1分数: {best_result['test_f1']:.4f}")
    print(f"   训练时间: {best_result['training_time']:.2f}秒")
    
    # 显示详细分类报告
    print(f"\n📋 {best_model_name} 详细分类报告:")
    report = best_result['classification_report']
    
    label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
    
    print(f"{'类别':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<8}")
    print("-" * 50)
    
    for i, label_name in label_mapping.items():
        if str(i) in report:
            metrics = report[str(i)]
            print(f"{label_name:<12} "
                  f"{metrics['precision']:<8.3f} "
                  f"{metrics['recall']:<8.3f} "
                  f"{metrics['f1-score']:<8.3f} "
                  f"{int(metrics['support']):<8}")
    
    # 总体指标
    if 'macro avg' in report:
        macro_avg = report['macro avg']
        print("-" * 50)
        print(f"{'Macro Avg':<12} "
              f"{macro_avg['precision']:<8.3f} "
              f"{macro_avg['recall']:<8.3f} "
              f"{macro_avg['f1-score']:<8.3f} "
              f"{int(macro_avg['support']):<8}")

def analyze_results(results):
    """分析结果并提供建议"""
    print(f"\n🔍 结果分析:")
    
    # 获取所有F1分数
    f1_scores = [result['test_f1'] for result in results.values()]
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"   平均测试F1分数: {avg_f1:.4f} (±{std_f1:.4f})")
    
    if avg_f1 > 0.8:
        print("   🎉 整体性能优秀!")
    elif avg_f1 > 0.7:
        print("   👍 整体性能良好!")
    elif avg_f1 > 0.6:
        print("   💡 性能中等，有提升空间")
    else:
        print("   ⚠️  性能较低，需要优化")
    
    print(f"\n💡 优化建议:")
    if avg_f1 < 0.8:
        print("   1. 尝试更复杂的特征工程 (N-gram, Word2Vec)")
        print("   2. 使用超参数调优")
        print("   3. 尝试集成方法")
        print("   4. 增加数据预处理步骤")
    
    print("   5. 考虑使用神经网络模型")
    print("   6. 添加多模态特征 (图像+文本)")

def main():
    """主函数"""
    print("欢迎使用传统机器学习模型训练演示!")
    print("本演示将使用真实MR2数据训练传统ML模型\n")
    
    # 1. 检查数据
    if not check_real_data():
        print("❌ 请确保真实数据文件存在于 data/ 目录")
        return
    
    # 2. 加载真实数据
    datasets = load_real_mr2_data()
    
    # 3. 训练模型
    start_time = time.time()
    results, vectorizer, models = train_traditional_models(datasets)
    total_time = time.time() - start_time
    
    # 4. 保存结果
    output_dir = save_results(results, vectorizer, models)
    
    # 5. 显示结果
    display_final_results(results)
    
    # 6. 分析结果
    analyze_results(results)
    
    # 7. 总结
    print(f"\n🎉 === 传统ML模型训练完成 ===")
    print(f"✅ 总训练时间: {total_time:.2f}秒")
    print(f"✅ 共训练 {len(models)} 个模型")
    print(f"📁 结果已保存到: {output_dir}")
    
    print(f"\n🚀 下一步建议:")
    print("   1. 查看生成的CSV文件分析详细结果")
    print("   2. 尝试调整超参数提升性能")
    print("   3. 运行神经网络模型进行对比")
    print("   4. 实验多模态融合方法")

if __name__ == "__main__":
    main()