#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/explainable_ai/explainers.py

"""
可解释性AI模块
包含LIME和SHAP等模型解释方法的实现
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import lime
import lime.lime_tabular
import shap
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 快速路径设置 (根据实际项目结构调整)
current_file = Path(__file__).resolve()
# 假设 explainable_ai 文件夹在 models 文件夹下，models 在项目根目录
project_root = current_file.parent.parent.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root (from explainers.py): {project_root}")

# 导入项目模块 (根据需要调整)
try:
    # 假设你需要从传统模型中加载一个已训练的模型或数据
    from models.traditional.ml_classifiers import MLClassifierTrainer 
    # 假设你需要数据加载器
    from data_utils.data_loaders import create_all_dataloaders 
    # 假设你需要配置文件管理器
    from utils.config_manager import get_config_manager, get_output_path
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块 (explainers.py)")
except ImportError as e:
    print(f"⚠️ 导入项目模块失败 (explainers.py): {e}")
    USE_PROJECT_MODULES = False

import logging
logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    模型解释器基类
    """
    def __init__(self, model: Any, feature_names: List[str]):
        """
        初始化解释器

        Args:
            model: 已训练的机器学习模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names

    def explain_instance(self, instance: np.ndarray, **kwargs) -> Any:
        """
        解释单个样本的预测

        Args:
            instance: 单个样本的特征值 (1D numpy array)
            **kwargs: 解释方法特定的参数

        Returns:
            解释结果
        """
        raise NotImplementedError("子类必须实现此方法")

    def explain_model(self, data: np.ndarray, **kwargs) -> Any:
        """
        解释整个模型的行为

        Args:
            data: 用于解释的数据集 (2D numpy array)
            **kwargs: 解释方法特定的参数

        Returns:
            模型级别的解释结果
        """
        raise NotImplementedError("子类必须实现此方法")


class LimeExplainer(ModelExplainer):
    """
    使用LIME进行模型解释
    """
    def __init__(self, model: Any, training_data: np.ndarray, feature_names: List[str], 
                 class_names: List[str], mode: str = "classification"):
        """
        初始化LIME解释器

        Args:
            model: 已训练的分类或回归模型
            training_data: 用于LIME背景分布的训练数据 (numpy array)
            feature_names: 特征名称列表
            class_names: 类别名称列表 (仅分类任务)
            mode: 'classification' 或 'regression'
        """
        super().__init__(model, feature_names)
        self.training_data = training_data
        self.class_names = class_names
        self.mode = mode
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.mode,
            discretize_continuous=True
        )
        print("✅ LIME解释器初始化完成")

    def explain_instance(self, instance: np.ndarray, num_features: int = 5, **kwargs) -> lime.explanation.Explanation:
        """
        解释单个样本的预测

        Args:
            instance: 单个样本 (1D numpy array)
            num_features: 要显示的特征数量
            **kwargs: 传递给LIME explainer.explain_instance的额外参数

        Returns:
            LIME的Explanation对象
        """
        if self.mode == "classification":
            # 分类任务需要 predict_proba 方法
            if not hasattr(self.model, 'predict_proba'):
                raise AttributeError("分类模型必须有 predict_proba 方法才能使用LIME。")
            explanation = self.explainer.explain_instance(
                data_row=instance,
                predict_fn=self.model.predict_proba,
                num_features=num_features,
                **kwargs
            )
        else: # regression
            if not hasattr(self.model, 'predict'):
                raise AttributeError("回归模型必须有 predict 方法才能使用LIME。")
            explanation = self.explainer.explain_instance(
                data_row=instance,
                predict_fn=self.model.predict,
                num_features=num_features,
                **kwargs
            )
        return explanation

    def explain_model(self, data: np.ndarray, **kwargs) -> Any:
        """
        LIME通常用于解释局部预测，全局解释可以通过聚合局部解释实现，
        但LIME本身不直接提供一个标准的“全局模型解释”对象。
        这里可以返回多个样本的解释。
        """
        print("⚠️  LIME主要用于局部解释。此方法将返回多个实例的解释。")
        explanations = []
        for i in range(min(5, data.shape[0])): # 解释前5个样本作为示例
            explanations.append(self.explain_instance(data[i], **kwargs))
        return explanations


class ShapExplainer(ModelExplainer):
    """
    使用SHAP进行模型解释
    """
    def __init__(self, model: Any, data: Optional[pd.DataFrame] = None, feature_names: Optional[List[str]] = None):
        """
        初始化SHAP解释器

        Args:
            model: 已训练的模型
            data: 用于SHAP背景分布的数据 (Pandas DataFrame 或 numpy array)。
                  对于某些SHAP解释器类型（如TreeExplainer的某些情况）可能不需要。
            feature_names: 特征名称，如果data是numpy array则需要提供。
        """
        if feature_names is None and data is not None and isinstance(data, np.ndarray):
            raise ValueError("当data是numpy array时，必须提供feature_names。")
        
        _feature_names = feature_names
        if data is not None and isinstance(data, pd.DataFrame):
            _feature_names = data.columns.tolist()
        
        super().__init__(model, _feature_names)
        self.data = data
        
        # 根据模型类型选择合适的SHAP解释器
        # TreeExplainer: 适用于树模型 (RandomForest, XGBoost, LightGBM, CatBoost)
        # KernelExplainer: 模型无关，但速度较慢
        # DeepExplainer: 适用于深度学习模型 (TensorFlow, Keras, PyTorch)
        # LinearExplainer: 适用于线性模型
        
        # 尝试为树模型使用TreeExplainer
        if hasattr(model, 'predict') and (isinstance(model, (RandomForestClassifier, LogisticRegression)) or "xgboost" in str(type(model)).lower() or "lightgbm" in str(type(model)).lower()):
             # 对于非深度学习的sklearn模型，通常SHAP会尝试封装
            print("INFO: 尝试使用 shap.Explainer 自动选择解释器...")
            self.explainer = shap.Explainer(self.model, self.data)
        elif "torch" in str(type(model)).lower() and hasattr(model, 'forward'):
            # 假设是PyTorch模型，需要 DeepExplainer 或 GradientExplainer
            # 注意: PyTorch模型的SHAP解释通常更复杂，可能需要特定的包装或输入格式
            print(f"INFO: 检测到PyTorch模型。你可能需要使用 shap.DeepExplainer 或 shap.GradientExplainer，并确保输入格式正确。")
            # 示例: self.explainer = shap.DeepExplainer(self.model, background_data_tensor)
            # 这里我们使用 KernelExplainer 作为通用回退，但它可能很慢
            if self.data is None:
                raise ValueError("对于 KernelExplainer，必须提供背景数据。")
            print("WARN: 回退到 shap.KernelExplainer，对于复杂模型可能会很慢。")
            self.explainer = shap.KernelExplainer(self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict, self.data)
        else:
            # 对于其他未知类型的模型，KernelExplainer是一个更通用的选择，但可能较慢
            if self.data is None:
                raise ValueError("对于 KernelExplainer，必须提供背景数据。")
            print("INFO: 未检测到特定模型类型，尝试使用 shap.KernelExplainer。")
            # KernelExplainer 需要一个返回概率（分类）或预测值（回归）的函数
            predict_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict
            self.explainer = shap.KernelExplainer(predict_fn, self.data)
            
        print("✅ SHAP解释器初始化完成")

    def explain_instance(self, instance: Union[np.ndarray, pd.DataFrame], **kwargs) -> shap.Explanation:
        """
        解释单个样本的预测

        Args:
            instance: 单个样本 (1D numpy array 或 Pandas Series/DataFrame)
            **kwargs: 传递给SHAP explainer的额外参数

        Returns:
            SHAP的Explanation对象或SHAP值数组
        """
        # TreeExplainer可以直接处理numpy array，KernelExplainer也是
        # 如果instance是1D numpy array，且explainer期望DataFrame，需要转换
        if isinstance(instance, np.ndarray) and instance.ndim == 1 and self.feature_names:
            if isinstance(self.data, pd.DataFrame) or (hasattr(self.explainer, 'expected_data_format') and self.explainer.expected_data_format == 'dataframe'):
                 instance_df = pd.DataFrame([instance], columns=self.feature_names)
                 shap_values_instance = self.explainer(instance_df, **kwargs)
                 return shap_values_instance
        
        # 对于其他情况或如果explainer能直接处理instance类型
        shap_values_instance = self.explainer(instance, **kwargs)
        return shap_values_instance

    def explain_model(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> shap.Explanation:
        """
        计算整个数据集的SHAP值

        Args:
            data: 要解释的数据集 (numpy array 或 Pandas DataFrame)
             **kwargs: 传递给SHAP explainer的额外参数

        Returns:
            SHAP的Explanation对象或SHAP值数组
        """
        shap_values = self.explainer(data, **kwargs)
        return shap_values

def load_sample_data(n_samples=1000, n_features=10, n_classes=2) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """加载或生成一个简单的样本数据集用于演示"""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features//2, 
                               n_classes=n_classes, random_state=42)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    class_names = [f"class_{i}" for i in range(n_classes)]
    return X, y, feature_names, class_names

def demo_explainable_ai():
    """
    演示可解释性AI模块的使用
    """
    print("🚀 可解释性AI模块演示")
    print("="*50)

    # 1. 加载数据和训练一个简单模型
    print("\n[阶段1: 加载数据和训练模型]")
    X, y, feature_names, class_names = load_sample_data(n_samples=200) # 使用较小数据集以加速SHAP
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练一个随机森林模型
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"随机森林模型训练完成。测试集准确率: {accuracy:.4f}")

    # 2. LIME解释
    print("\n[阶段2: 使用LIME进行解释]")
    try:
        lime_explainer = LimeExplainer(
            model=model,
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification"
        )
        
        # 解释测试集中的一个样本
        instance_to_explain_lime = X_test[0]
        lime_explanation = lime_explainer.explain_instance(instance_to_explain_lime, num_features=5)
        
        print(f"\nLIME解释样本0 (真实标签: {class_names[y_test[0]]}):")
        # LIME的解释可以直接打印或保存为HTML
        # lime_explanation.show_in_notebook(show_table=True, show_all=False) # 在Jupyter Notebook中显示
        # lime_explanation.save_to_file('lime_report.html') # 保存为HTML
        print("LIME解释特征权重 (对于预测概率最高的类别):")
        for feature, weight in lime_explanation.as_list():
            print(f"  特征: {feature}, 权重: {weight:.4f}")

    except Exception as e:
        print(f"❌ LIME解释失败: {e}")
        import traceback
        traceback.print_exc()

    # 3. SHAP解释
    print("\n[阶段3: 使用SHAP进行解释]")
    try:
        # 对于SHAP，通常传递DataFrame更容易处理特征名称
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        shap_explainer = ShapExplainer(
            model=model,
            data=X_train_df # 背景数据
        )
        
        # 解释测试集中的一个样本
        instance_to_explain_shap = X_test_df.iloc[[0]] # SHAP通常期望DataFrame
        shap_explanation_instance = shap_explainer.explain_instance(instance_to_explain_shap)
        
        print(f"\nSHAP解释样本0 (真实标签: {class_names[y_test[0]]}):")
        print(f"  基础值 (Expected Value): {shap_explanation_instance.base_values[0]}") # 对于多输出模型，可能是数组
        print(f"  SHAP值 (对于预测概率最高的类别):")
        # shap_explanation_instance.values 是一个数组，对于二分类，通常是 [shap_values_for_class_0, shap_values_for_class_1]
        # 我们这里假设关注class 1的SHAP值 (如果模型输出概率的话)
        # 对于随机森林，shap.Explainer(model, data) 返回的是针对每个类别的SHAP值
        # shap_values_instance.values的形状可能是 (num_instances, num_features, num_classes) 或 (num_instances, num_features)
        
        shap_values_for_instance = shap_explanation_instance.values
        if shap_values_for_instance.ndim == 3: # (instances, features, classes)
            # 通常关注正类（如类别1）的SHAP值
            # 如果模型是二分类，并且输出两个类别的概率，shap_values_for_instance.values[0, :, 1] 是样本对类别1的SHAP值
            # 如果模型直接输出类别1的概率（或logit），则是 shap_values_for_instance.values[0, :]
            # 对于 RandomForestClassifier，shap.Explainer 返回的是针对每个类输出的 SHAP 值
            # 这里我们打印类别1的SHAP值
            class_index_to_explain = 1 
            if shap_values_for_instance.shape[2] > class_index_to_explain:
                for feature_idx, feature_name in enumerate(feature_names):
                    print(f"  特征: {feature_name}, SHAP值 (对类别 {class_names[class_index_to_explain]}): {shap_values_for_instance[0, feature_idx, class_index_to_explain]:.4f}")
            else:
                print("SHAP值输出的类别维度不足。")

        elif shap_values_for_instance.ndim == 2: # (instances, features) - 可能针对单一输出或特定类别
             for feature_idx, feature_name in enumerate(feature_names):
                print(f"  特征: {feature_name}, SHAP值: {shap_values_for_instance[0, feature_idx]:.4f}")
        else:
            print("SHAP值数组的维度不符合预期。")


        # SHAP图 (如果环境支持matplotlib)
        try:
            print("\n尝试生成SHAP摘要图 (可能需要matplotlib)...")
            # 计算整个测试集的SHAP值
            shap_values_test = shap_explainer.explain_model(X_test_df)
            # shap.summary_plot(shap_values_test.values[:,:,1], X_test_df, show=False) # 对于多分类输出
            # 如果是单一输出或只想解释一个类别的shap值
            # plt.title("SHAP Summary Plot for Class 1")
            # plt.savefig("shap_summary_plot.png")
            # plt.close()
            # print("SHAP摘要图已尝试保存为 shap_summary_plot.png (如果matplotlib可用且配置正确)")
            
            # SHAP力图 (force plot) for the first instance
            # shap.force_plot(shap_explainer.explainer.expected_value[1], shap_values_test.values[0,:,1], X_test_df.iloc[0,:], matplotlib=True, show=False)
            # plt.title("SHAP Force Plot for Instance 0, Class 1")
            # plt.savefig("shap_force_plot_instance0.png")
            # plt.close()
            # print("SHAP力图已尝试保存为 shap_force_plot_instance0.png")
            print("由于环境限制，SHAP绘图部分已注释掉。在本地运行时可以取消注释以查看图形。")

        except Exception as plot_e:
            print(f"❌ 生成SHAP图失败: {plot_e}")

    except Exception as e:
        print(f"❌ SHAP解释失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n[阶段4: 演示完成]")
    print("✅ 可解释性AI模块演示完成。")

if __name__ == "__main__":
    demo_explainable_ai()