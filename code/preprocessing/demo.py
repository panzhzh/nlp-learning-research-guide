#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# preprocessing/demo.py

"""
预处理模块演示 - 简化版
直接运行即可体验文本和图像预处理
"""

from text_processing import TextProcessor
from image_processing import ImageProcessor

def main():
    """简单演示预处理功能"""
    print("🔧 预处理模块演示")
    print("="*50)
    
    # 文本处理演示
    print("\n📝 文本处理演示:")
    processor = TextProcessor(language='mixed')
    
    test_texts = [
        "这是一个测试文本 This is a test!",
        "包含URL的文本 https://example.com 和@username",
        "混合语言文本 with English words 中文字符"
    ]
    
    for text in test_texts:
        print(f"\n原文: {text}")
        cleaned = processor.clean_text(text)
        tokens = processor.tokenize(text)
        print(f"清洗: {cleaned}")
        print(f"分词: {tokens[:5]}...")  # 只显示前5个
    
    # 图像处理演示
    print("\n🖼️  图像处理演示:")
    img_processor = ImageProcessor(target_size=(224, 224))
    
    # 处理数据集（只处理train，演示用）
    try:
        results = img_processor.process_mr2_dataset(splits=['train'])
        if results:
            print("图像处理完成!")
    except Exception as e:
        print(f"图像处理演示跳过: {e}")
    
    print("\n✅ 预处理演示完成!")

if __name__ == "__main__":
    main()