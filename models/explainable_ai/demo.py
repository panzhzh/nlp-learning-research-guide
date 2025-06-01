#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/explainable_ai/demo.py

"""
演示可解释性AI模块 (LIME, SHAP) 的使用，并包含可视化输出。
SHAP Force Plot 中的特征值将格式化为两位小数。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 快速路径设置 (根据实际项目结构调整)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root (from demo.py): {project_root}")
sys.path.insert(0, str(current_file.parent))

try:
    from explainers import LimeExplainer, ShapExplainer
    print("✅ 成功导入项目模块 (explainers from demo.py)")
except ImportError as e:
    print(f"⚠️ 导入项目模块失败 (demo.py): {e}")
    try:
        from models.explainable_ai.explainers import LimeExplainer, ShapExplainer
        print("✅ 成功通过绝对路径导入项目模块 (explainers from demo.py)")
    except ImportError as e_abs:
        print(f"⚠️ 通过绝对路径导入项目模块也失败 (demo.py): {e_abs}")
        sys.exit(1)

import logging
logger = logging.getLogger(__name__)

# <<< --- 前面的 generate_data 和 train_model 函数保持不变 --- >>>
def generate_data(n_samples: int = 250, n_features: int = 15, n_classes: int = 2, random_state: int = 42) \
        -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    生成用于分类任务的样本数据。
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(1, n_features // 4),
        n_clusters_per_class=1,
        n_classes=n_classes,
        random_state=random_state
    )
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    class_names = [f"class_{j}" for j in range(n_classes)]
    print(f"数据已生成: X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_names, class_names

def train_model(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> RandomForestClassifier:
    """
    训练一个简单的随机森林分类器。
    """
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, min_samples_leaf=5)
    model.fit(X_train, y_train)
    print("模型训练完成: RandomForestClassifier")
    return model

def run_explanation_demo():
    """
    执行LIME和SHAP解释的完整演示，包含可视化输出。
    """
    print("\n🚀 开始可解释性AI演示 (demo.py) 🚀")
    print("=" * 60)

    # 创建用于保存可视化结果的目录
    output_visualization_dir = project_root / "outputs" / "explainable_ai_visuals"
    output_visualization_dir.mkdir(parents=True, exist_ok=True)
    print(f"可视化结果将保存到: {output_visualization_dir}")

    # 1. 数据准备
    print("\n[阶段1: 生成数据]")
    X, y, feature_names, class_names = generate_data(n_samples=300, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    print(f"训练集: {X_train_df.shape}, 测试集: {X_test_df.shape}")

    # 2. 模型训练
    print("\n[阶段2: 训练模型]")
    model = train_model(X_train_df, y_train)
    y_pred_test = model.predict(X_test_df)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"模型在测试集上的准确率: {test_accuracy:.4f}")

    sample_idx = 0
    instance_to_explain_np = X_test[sample_idx]
    instance_to_explain_df_row = X_test_df.iloc[[sample_idx]]

    true_label = class_names[y_test[sample_idx]]
    predicted_proba = model.predict_proba(instance_to_explain_df_row)[0]
    predicted_class_idx = np.argmax(predicted_proba)
    predicted_label = class_names[predicted_class_idx]

    print(f"\n选择测试集样本 #{sample_idx} 进行解释:")
    print(f"  特征值 (原始): {instance_to_explain_np}") # 打印原始值以对比
    print(f"  特征值 (显示用, 四舍五入到两位小数): {instance_to_explain_np.round(2)}")
    print(f"  真实标签: {true_label}")
    print(f"  模型预测标签: {predicted_label} (概率: {predicted_proba[predicted_class_idx]:.4f})")
    print(f"  模型预测概率: {dict(zip(class_names, predicted_proba.round(4)))}")

    # 3. 使用LIME进行解释
    print("\n[阶段3: LIME解释]")
    try:
        lime_explainer = LimeExplainer(
            model=model,
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification"
        )

        lime_explanation = lime_explainer.explain_instance(
            instance_to_explain_np,
            num_features=5,
            top_labels=1
        )

        print(f"\nLIME 对样本 #{sample_idx} (预测: {predicted_label}) 的解释:")
        print("  特征贡献 (LIME):")
        for feature, weight in lime_explanation.as_list():
            print(f"    - {feature}: {weight:.4f}")

        # --- LIME 可视化: 保存为 HTML ---
        lime_html_path = output_visualization_dir / f"lime_report_sample_{sample_idx}.html"
        lime_explanation.save_to_file(lime_html_path)
        print(f"  ✅ LIME解释报告已保存到: {lime_html_path}")
        print(f"     您可以直接在浏览器中打开此HTML文件查看可视化LIME解释。")

    except Exception as e:
        print(f"❌ LIME解释过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    # 4. 使用SHAP进行解释
    print("\n[阶段4: SHAP解释与可视化]")
    try:
        shap_explainer_instance = ShapExplainer(
            model=model,
            data=X_train_df
        )

        shap_values_one_instance = shap_explainer_instance.explain_instance(instance_to_explain_df_row)

        print(f"\nSHAP 对样本 #{sample_idx} (预测: {predicted_label}) 的文本解释:")
        if hasattr(shap_values_one_instance, 'base_values') and shap_values_one_instance.base_values is not None:
            base_val_text = ""
            if shap_values_one_instance.base_values.ndim > 0 and len(shap_values_one_instance.base_values) > predicted_class_idx:
                 base_val_text = f"{shap_values_one_instance.base_values[predicted_class_idx]:.4f} (类别 {predicted_label})"
            elif shap_values_one_instance.base_values.ndim == 0 :
                 base_val_text = f"{shap_values_one_instance.base_values:.4f}"
            else:
                 base_val_text = f"{shap_values_one_instance.base_values} (所有类别)"
            print(f"  SHAP基础值: {base_val_text}")

        print("  特征贡献 (SHAP values):")
        current_shap_values = shap_values_one_instance.values
        shap_values_for_predicted_class_plot = None # 用于绘图

        if current_shap_values.ndim == 3 and current_shap_values.shape[0] == 1:
            shap_vals_for_predicted_class = current_shap_values[0, :, predicted_class_idx]
            shap_values_for_predicted_class_plot = shap_vals_for_predicted_class
            for i, feature_name in enumerate(feature_names):
                print(f"    - {feature_name}: {shap_vals_for_predicted_class[i]:.4f} (对类别 {predicted_label})")
        elif current_shap_values.ndim == 2 and current_shap_values.shape[0] == 1:
            shap_vals_for_instance = current_shap_values[0, :]
            shap_values_for_predicted_class_plot = shap_vals_for_instance
            for i, feature_name in enumerate(feature_names):
                print(f"    - {feature_name}: {shap_vals_for_instance[i]:.4f}")
        else:
            print(f"    SHAP值格式未知或不匹配单个实例解释: shape={current_shap_values.shape}")

        # --- SHAP 可视化 ---
        print("\n  生成SHAP可视化图:")
        try:
            import matplotlib.pyplot as plt
            import shap
            # shap.initjs() # 在脚本中运行时，此行主要用于Jupyter环境但可能导致IPython依赖问题，已注释

            # 1. 单个实例的力图 (Force Plot)
            if shap_values_for_predicted_class_plot is not None:
                explainer_base_value = shap_explainer_instance.explainer.expected_value
                if isinstance(explainer_base_value, (list, np.ndarray)) and len(explainer_base_value) > predicted_class_idx:
                    base_value_for_plot = explainer_base_value[predicted_class_idx]
                else:
                    base_value_for_plot = explainer_base_value

                # ---- 修改开始: 格式化用于显示的特征值 ----
                instance_for_plot_display = instance_to_explain_df_row.copy()
                for col in instance_for_plot_display.columns:
                    if pd.api.types.is_numeric_dtype(instance_for_plot_display[col]):
                        instance_for_plot_display[col] = instance_for_plot_display[col].round(2)
                # ---- 修改结束 ----

                plt.figure()
                shap.force_plot(
                    base_value_for_plot,
                    shap_values_for_predicted_class_plot,
                    instance_for_plot_display, # <--- 使用格式化后的DataFrame进行显示
                    matplotlib=True,
                    show=False
                )
                force_plot_path = output_visualization_dir / f"shap_force_plot_sample_{sample_idx}_class_{predicted_label}.png"
                plt.savefig(force_plot_path, bbox_inches='tight')
                plt.close()
                print(f"  ✅ SHAP力图已保存到: {force_plot_path}")
            else:
                print("  ⚠️ 无法生成SHAP力图，因SHAP值格式不适用于单实例绘图。")

            # 2. 所有测试样本的SHAP值 (用于摘要图)
            print("  计算整个测试集的SHAP值 (可能需要一些时间)...")
            shap_values_test_set = shap_explainer_instance.explain_model(X_test_df)

            shap_values_for_summary = None
            if shap_values_test_set.values.ndim == 3:
                class_to_summarize_idx = 1
                if shap_values_test_set.values.shape[2] > class_to_summarize_idx:
                    shap_values_for_summary = shap_values_test_set.values[:, :, class_to_summarize_idx]
                    summary_plot_title = f"SHAP Summary Plot (Class {class_names[class_to_summarize_idx]})"
                else:
                    print(f"  ⚠️ 无法为类别索引 {class_to_summarize_idx} 生成摘要图，SHAP值类别维度不足。")
            elif shap_values_test_set.values.ndim == 2:
                shap_values_for_summary = shap_values_test_set.values
                summary_plot_title = "SHAP Summary Plot"

            if shap_values_for_summary is not None:
                # ---- 修改开始: 准备用于摘要图显示的特征值 (如果需要，但摘要图通常用颜色表示原始值范围) ----
                # 对于摘要图，特征值通常通过颜色编码显示其原始范围，而不是直接显示数值。
                # 但如果确实需要在X轴标签上显示截断的值，需要传递格式化后的X_test_df。
                # 这里我们保持X_test_df不变，因为摘要图的目的是展示分布。
                # X_test_df_display_for_summary = X_test_df.copy()
                # for col in X_test_df_display_for_summary.columns:
                #    if pd.api.types.is_numeric_dtype(X_test_df_display_for_summary[col]):
                #        X_test_df_display_for_summary[col] = X_test_df_display_for_summary[col].round(2)
                # ---- 修改结束 ----

                plt.figure()
                shap.summary_plot(
                    shap_values_for_summary,
                    X_test_df, # 传递原始的 X_test_df，颜色会基于原始值
                    show=False,
                    plot_size=(10, 8)
                )
                plt.title(summary_plot_title, fontsize=14)
                summary_plot_path = output_visualization_dir / "shap_summary_plot.png"
                plt.savefig(summary_plot_path, bbox_inches='tight')
                plt.close()
                print(f"  ✅ SHAP摘要图已保存到: {summary_plot_path}")
                print(f"     摘要图显示了每个特征对模型输出的整体影响。点越分散，说明该特征对不同样本的影响差异越大。颜色代表特征值的高低。")

        except ImportError:
            print("  ⚠️ matplotlib或shap未完全安装或配置正确，无法生成SHAP图。请运行: pip install matplotlib shap")
        except Exception as plot_e:
            print(f"  ❌ 生成SHAP图时发生错误: {plot_e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"❌ SHAP解释过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("🎉 可解释性AI演示完成! 🎉")

if __name__ == "__main__":
    run_explanation_demo()