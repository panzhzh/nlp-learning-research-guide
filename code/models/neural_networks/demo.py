#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/neural_networks/demo.py

"""
神经网络模型演示 - 简化版
直接运行即可体验神经网络模型训练
"""

from text_models import NeuralTextClassifier

def main():
    """简单演示神经网络模型训练"""
    print("🧠 神经网络模型演示")
    print("="*50)
    
    # 创建训练器
    trainer = NeuralTextClassifier(data_dir="../../data")
    
    # 训练所有模型（快速版本）
    trainer.train_all_models(
        epochs=5,           # 减少训练轮数
        batch_size=16,      # 小批次
        learning_rate=0.001
    )
    
    print("\n✅ 神经网络演示完成!")

if __name__ == "__main__":
    main()