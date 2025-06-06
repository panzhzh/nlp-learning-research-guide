#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/llms/__init__.py

"""
大语言模型模块

包含开源LLM、提示工程、少样本学习和RAG集成等功能
"""

from .open_source_llms import QwenRumorClassifier
from .prompt_engineering import RumorPromptTemplate, PromptManager
from .few_shot_learning import FewShotLearner
from .rag_integration import RAGRumorDetector, KnowledgeBase, AdvancedRAGFeatures

__all__ = [
    'QwenRumorClassifier',
    'RumorPromptTemplate', 
    'PromptManager',
    'FewShotLearner',
    'RAGRumorDetector',
    'KnowledgeBase',
    'AdvancedRAGFeatures'
]