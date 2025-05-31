#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/traditional/demo.py

"""
传统机器学习演示 - 简化版
直接运行即可体验传统ML模型训练
"""

from ml_classifiers import MLClassifierTrainer

def main():
    """简单演示传统机器学习模型训练"""
    print("🤖 传统机器学习模型演示")
    print("="*50)
    
    # 创建训练器
    trainer = MLClassifierTrainer(data_dir="../../data")
    
    # 训练所有模型
    trainer.train_all_models(use_hyperparameter_tuning=False)
    
    # 演示预测
    print("\n🔮 预测演示:")
    test_texts = [
        "这是一个关于新技术的真实新闻",
        "网传某地发生事故，官方未确认", 
        "This might be fake news"
    ]
    
    for text in test_texts:
        try:
            result = trainer.predict_single_text(text)
            print(f"文本: {text[:30]}...")
            print(f"预测: {result['prediction_label']}")
        except:
            print(f"预测失败: {text[:30]}...")
    
    print("\n✅ 演示完成!")

if __name__ == "__main__":
    main()