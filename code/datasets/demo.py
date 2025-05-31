#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# datasets/demo.py

"""
数据集分析演示 - 简化版
直接运行即可分析MR2数据集
"""

from mr2_analysis import MR2DatasetAnalyzer

def main():
    """简单演示数据集分析"""
    print("📊 MR2数据集分析演示")
    print("="*50)
    
    # 创建分析器
    analyzer = MR2DatasetAnalyzer(data_dir='../data')
    
    # 运行完整分析
    analyzer.run_complete_analysis()
    
    print("\n✅ 数据分析演示完成!")

if __name__ == "__main__":
    main()