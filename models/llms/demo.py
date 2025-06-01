#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/llms/demo.py

"""
LLM模块完整演示
调用各个子模块的演示功能，包括新增的RAG功能
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
    print("🤖 大语言模型(LLMs)模块完整演示")
    print("=" * 60)
    print("本演示将依次展示各个子模块的功能，包括新增的RAG集成")
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
    
    # 4. 演示RAG集成模块 (新增)
    print("\n" + "🔍" * 20)
    print("4. RAG集成模块演示")
    print("🔍" * 20)
    
    try:
        from models.llms.rag_integration import demo_rag_integration
        demo_rag_integration()
        print("✅ RAG集成模块演示完成")
    except Exception as e:
        print(f"❌ RAG集成模块演示失败: {e}")
    
    # 5. 演示高级RAG功能 (新增)
    print("\n" + "🔬" * 20)
    print("5. 高级RAG功能演示")
    print("🔬" * 20)
    
    try:
        from models.llms.rag_integration import demo_advanced_rag_features
        demo_advanced_rag_features()
        print("✅ 高级RAG功能演示完成")
    except Exception as e:
        print(f"❌ 高级RAG功能演示失败: {e}")
    
    # 演示总结
    print("\n" + "🎉" * 20)
    print("LLMs模块演示总结")
    print("🎉" * 20)
    print("✅ 完成了以下模块的演示:")
    print("   1. open_source_llms.py - Qwen3-0.6B谣言检测")
    print("   2. prompt_engineering.py - 多种提示工程技术")
    print("   3. few_shot_learning.py - 少样本学习策略")
    print("   4. rag_integration.py - RAG检索增强生成 (新增)")
    print("   5. advanced_rag_features.py - 高级RAG功能 (新增)")
    
    print("\n📚 学习要点:")
    print("   - 如何使用开源大语言模型")
    print("   - 如何设计有效的提示模板")
    print("   - 如何实现少样本学习")
    print("   - 如何构建RAG系统提升检测准确性")
    print("   - 如何使用高级RAG技术(多查询、迭代等)")
    print("   - 如何进行谣言检测任务")
    
    print(f"\n🎯 如需单独运行某个模块:")
    print(f"   python models/llms/open_source_llms.py")
    print(f"   python models/llms/prompt_engineering.py")
    print(f"   python models/llms/few_shot_learning.py")
    print(f"   python models/llms/rag_integration.py")
    print(f"   python models/llms/test_rag.py  # RAG功能测试")
    
    # 综合演示：RAG vs 标准方法对比
    print("\n" + "⚖️" * 20)
    print("6. RAG vs 标准方法对比演示")
    print("⚖️" * 20)
    
    try:
        demo_rag_vs_standard_comparison()
        print("✅ 对比演示完成")
    except Exception as e:
        print(f"❌ 对比演示失败: {e}")


def demo_rag_vs_standard_comparison():
    """演示RAG方法与标准方法的对比"""
    print("🔄 执行RAG vs 标准方法对比...")
    
    try:
        from models.llms.rag_integration import create_rag_detector
        
        # 创建RAG检测器
        rag_detector = create_rag_detector(use_existing_llm=False)
        
        # 测试案例
        test_cases = [
            {
                'text': '中国科学院发布最新研究成果，在人工智能领域取得重大突破',
                'expected': 'Non-rumor',
                'description': '权威机构发布'
            },
            {
                'text': '网传某市明天将发生大地震，请大家做好撤离准备',
                'expected': 'Rumor', 
                'description': '地震谣言'
            },
            {
                'text': '据不完全统计，新产品在市场上反响良好',
                'expected': 'Unverified',
                'description': '模糊信息源'
            }
        ]
        
        print(f"\n📊 对比结果:")
        print(f"{'案例':<30} {'标准方法':<15} {'RAG方法':<15} {'期望结果':<15}")
        print("-" * 80)
        
        for i, case in enumerate(test_cases, 1):
            text = case['text']
            expected = case['expected']
            
            # 标准方法
            standard_result = rag_detector.retrieve_and_generate(text, use_context=False)
            standard_pred = standard_result['predicted_class']
            
            # RAG方法
            rag_result = rag_detector.retrieve_and_generate(text, use_context=True)
            rag_pred = rag_result['predicted_class']
            
            # 显示结果
            description = case['description']
            print(f"{description:<30} {standard_pred:<15} {rag_pred:<15} {expected:<15}")
            
            # 详细信息
            print(f"  标准置信度: {standard_result['confidence']:.3f}")
            print(f"  RAG置信度: {rag_result['confidence']:.3f}")
            print(f"  检索文档数: {rag_result['retrieved_count']}")
            print()
        
        # 性能评估
        print("🔬 性能评估:")
        evaluation = rag_detector.evaluate_rag_performance()
        
        print(f"  标准方法准确率: {evaluation['standard_mode']['accuracy']:.4f}")
        print(f"  RAG方法准确率: {evaluation['rag_mode']['accuracy']:.4f}")
        print(f"  性能提升: {evaluation['improvement']['accuracy_gain']:+.4f}")
        print(f"  平均检索文档数: {evaluation['rag_mode']['avg_retrieved_docs']:.1f}")
        
    except Exception as e:
        print(f"对比演示出错: {e}")
        import traceback
        traceback.print_exc()


def demo_rag_workflow():
    """演示完整的RAG工作流程"""
    print("\n🔄 RAG工作流程演示")
    print("-" * 40)
    
    try:
        from models.llms.rag_integration import RAGRumorDetector, KnowledgeBase
        
        # Step 1: 构建知识库
        print("步骤1: 构建知识库")
        kb = KnowledgeBase()
        print(f"  知识库包含 {len(kb.documents)} 个文档")
        
        # Step 2: 创建RAG检测器
        print("步骤2: 创建RAG检测器")
        rag_detector = RAGRumorDetector(knowledge_base=kb)
        print("  RAG检测器初始化完成")
        
        # Step 3: 文档检索演示
        print("步骤3: 文档检索演示")
        query = "权威机构发布信息"
        retrieved_docs = kb.retrieve(query, top_k=3)
        print(f"  检索查询: {query}")
        print(f"  检索到 {len(retrieved_docs)} 个相关文档:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"    {i}. {doc['content'][:60]}... (相关度: {doc.get('score', 0):.3f})")
        
        # Step 4: 提示生成演示
        print("步骤4: 提示生成演示")
        test_text = "官方媒体报道重要新闻"
        prompt = rag_detector.create_rag_prompt(test_text, retrieved_docs)
        print(f"  测试文本: {test_text}")
        print(f"  生成的RAG提示长度: {len(prompt)} 字符")
        print(f"  提示预览: {prompt[:150]}...")
        
        # Step 5: 完整分析演示
        print("步骤5: 完整分析演示")
        result = rag_detector.retrieve_and_generate(test_text)
        print(f"  分析结果: {result['predicted_class']}")
        print(f"  置信度: {result['confidence']:.3f}")
        print(f"  使用上下文: {result['context_used']}")
        
        print("✅ RAG工作流程演示完成")
        
    except Exception as e:
        print(f"❌ RAG工作流程演示失败: {e}")


if __name__ == "__main__":
    main()