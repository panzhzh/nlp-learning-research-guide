#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/llms/demo.py

"""
LLM模块简单演示
调用各个子模块的演示功能
"""

import sys
from pathlib import Path

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """主演示函数"""
    print("🤖 大语言模型(LLMs)模块演示")
    print("=" * 60)
    print("本演示将依次展示各个子模块的功能")
    print("=" * 60)
    
    # 1. 演示开源LLM模块
    print("\n" + "🔥" * 20)
    print("1. 开源LLM模块演示 (Qwen3-0.6B)")
    print("🔥" * 20)
    
    try:
        from models.llms.open_source_llms import demo_qwen_classification
        demo_qwen_classification()
        print("✅ 开源LLM模块演示完成")
    except Exception as e:
        print(f"❌ 开源LLM模块演示失败: {e}")
    
    # 2. 演示提示工程模块
    print("\n" + "📝" * 20)
    print("2. 提示工程模块演示")
    print("📝" * 20)
    
    try:
        from models.llms.prompt_engineering import demo_prompt_engineering
        demo_prompt_engineering()
        print("✅ 提示工程模块演示完成")
    except Exception as e:
        print(f"❌ 提示工程模块演示失败: {e}")
    
    # 3. 演示少样本学习模块
    print("\n" + "🎯" * 20)
    print("3. 少样本学习模块演示")
    print("🎯" * 20)
    
    try:
        from models.llms.few_shot_learning import demo_few_shot_learning
        demo_few_shot_learning()
        print("✅ 少样本学习模块演示完成")
    except Exception as e:
        print(f"❌ 少样本学习模块演示失败: {e}")
    
    # 演示总结
    print("\n" + "🎉" * 20)
    print("LLMs模块演示总结")
    print("🎉" * 20)
    print("✅ 完成了以下模块的演示:")
    print("   1. open_source_llms.py - Qwen3-0.6B谣言检测")
    print("   2. prompt_engineering.py - 多种提示工程技术")
    print("   3. few_shot_learning.py - 少样本学习策略")
    print("\n📚 学习要点:")
    print("   - 如何使用开源大语言模型")
    print("   - 如何设计有效的提示模板")
    print("   - 如何实现少样本学习")
    print("   - 如何进行谣言检测任务")
    
    print(f"\n🎯 如需单独运行某个模块:")
    print(f"   python models/llms/open_source_llms.py")
    print(f"   python models/llms/prompt_engineering.py")
    print(f"   python models/llms/few_shot_learning.py")


if __name__ == "__main__":
    main()