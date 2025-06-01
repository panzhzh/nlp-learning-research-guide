#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/explainable_ai/anchors_explainer.py

"""
使用Anchors算法进行模型解释
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# 注意：需要安装 anchor-exp 包
# pip install anchor-exp
try:
    from anchor import anchor_tabular
    HAS_ANCHOR = True
except ImportError:
    print("⚠️  anchor-exp 未安装，AnchorsExplainer 将不可用。请运行: pip install anchor-exp")
    HAS_ANCHOR = False

from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# 路径设置 (与之前类似)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
logger = logging.getLogger(__name__)

class AnchorsExplainer:
    """
    使用Anchors进行表格数据模型解释
    """
    def __init__(self, 
                 training_data: np.ndarray, 
                 feature_names: List[str],
                 class_names: List[str],
                 categorical_names: Optional[Dict[int, List[str]]] = None):
        """
        初始化Anchors解释器

        Args:
            training_data: 用于训练解释器的背景数据 (numpy array)
            feature_names: 特征名称列表
            class_names: 类别名称列表
            categorical_names: 字典，键是分类特征的索引，值是该特征的可能取值列表
        """
        if not HAS_ANCHOR:
            raise ImportError("anchor-exp 包未安装，无法使用AnchorsExplainer。")
            
        self.feature_names = feature_names
        self.class_names = class_names
        
        self.explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=self.class_names,
            feature_names=self.feature_names,
            train_data=training_data,
            categorical_names=categorical_names if categorical_names else {}
        )
        print("✅ Anchors解释器初始化完成")

    def explain_instance(self, 
                         instance: np.ndarray, 
                         model_predict_fn, 
                         threshold: float = 0.95,
                         **kwargs) -> Any: # anchor.explanation.AnchorExplanation
        """
        解释单个样本的预测

        Args:
            instance: 单个样本 (1D numpy array)
            model_predict_fn: 模型预测函数，输入numpy array，输出预测类别索引
            threshold: 锚点规则应达到的精度阈值
            **kwargs: 传递给 explainer.explain_instance 的其他参数

        Returns:
            AnchorExplanation对象或错误信息
        """
        if not hasattr(model_predict_fn, '__call__'):
            raise ValueError("model_predict_fn 必须是一个可调用对象 (函数或方法)")

        try:
            # Anchors 需要一个返回整数类别标签的预测函数
            explanation = self.explainer.explain_instance(
                data_row=instance,
                predict_fn=model_predict_fn,
                threshold=threshold,
                **kwargs
            )
            return explanation
        except Exception as e:
            logger.error(f"Anchors解释失败: {e}")
            return str(e)

# --- 演示代码 ---
def demo_anchors_explainer():
    if not HAS_ANCHOR:
        print("跳过Anchors演示，因为 anchor-exp 未安装。")
        return

    print("\n🚀 Anchors 可解释性演示")
    print("="*50)

    # 1. 准备数据和模型 (与LIME/SHAP演示类似)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=5, n_informative=2, n_redundant=0, random_state=42)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    class_names = ['class_0', 'class_1']
    
    # 假设 feature_2 是分类特征，取值为 ['A', 'B', 'C']
    # 为了演示，我们将 feature_2 的值映射为 0, 1, 2
    categorical_feature_index = 2
    X[:, categorical_feature_index] = np.random.randint(0, 3, X.shape[0])
    categorical_names = {categorical_feature_index: ['ValueA', 'ValueB', 'ValueC']}


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print(f"模型训练完成。测试准确率: {model.score(X_test, y_test):.4f}")

    # 2. 初始化Anchors解释器
    try:
        anchors_explainer = AnchorsExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            categorical_names=categorical_names
        )
    except ImportError: # 如果在主脚本中导入失败但在这里成功，再次捕获
        print("无法初始化Anchors解释器，因为 anchor-exp 导入失败。")
        return
    except Exception as e:
        print(f"初始化Anchors解释器失败: {e}")
        return

    # 3. 解释一个实例
    instance_to_explain = X_test[0]
    true_label = class_names[y_test[0]]
    
    # Anchors 需要一个返回预测类别索引的函数
    def model_predict_labels(data_array):
        return model.predict(data_array)

    print(f"\n解释样本 (真实标签: {true_label}): {instance_to_explain}")
    
    anchor_explanation = anchors_explainer.explain_instance(
        instance_to_explain,
        model_predict_labels, # 注意传递的是返回类别标签的函数
        threshold=0.90 # 可以调整锚点精度要求
    )

    if isinstance(anchor_explanation, str): # 检查是否返回了错误信息
        print(f"获取锚点解释失败: {anchor_explanation}")
    else:
        print("\n锚点解释:")
        print(f"  预测类别: {class_names[model.predict(instance_to_explain.reshape(1, -1))[0]]}")
        print('  规则 (Anchor): %s' % (' AND '.join(anchor_explanation.names())))
        print('  精度 (Precision): %.2f' % anchor_explanation.precision())
        print('  覆盖率 (Coverage): %.2f' % anchor_explanation.coverage())
        
        # 还可以查看锚点覆盖的样本
        # print('  示例匹配此锚点且预测相同的样本:')
        # for exp in anchor_explanation.examples(only_same_prediction=True):
        #     print(exp)

    print("\n✅ Anchors 演示完成。")

if __name__ == "__main__":
    demo_anchors_explainer()