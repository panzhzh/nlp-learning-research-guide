#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# scripts/setup_project.py

"""
项目初始化脚本
创建完整的目录结构并生成必要的__init__.py文件
"""

import os
import json

def create_project_structure():
    """创建完整的项目目录结构"""
    
    # 定义目录结构 - 所有内容都在code目录下
    directories = [
        # 根目录
        'code',
        
        # 配置文件目录
        'code/config',
        
        # 数据目录
        'code/data/raw/train',
        'code/data/raw/val', 
        'code/data/raw/test',
        'code/data/processed',
        
        # 预处理模块
        'code/preprocessing',
        
        # 模型模块
        'code/models',
        'code/models/traditional',
        'code/models/neural_networks',
        'code/models/pretrained',
        'code/models/multimodal',
        'code/models/graph_neural_networks',
        'code/models/llms',
        
        # 嵌入模块
        'code/embeddings',
        
        # RAG系统
        'code/rag',
        
        # 训练框架
        'code/training',
        
        # 评估模块
        'code/evaluation',
        
        # 工具模块
        'code/utils',
        
        # 数据集处理
        'code/datasets',
        
        # 示例
        'code/examples',
        'code/examples/tutorials',
        
        # 测试
        'code/tests',
        
        # 脚本
        'code/scripts'
    ]
    
    # 创建目录
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
        
        # 为Python包创建__init__.py文件
        if not directory.startswith('code/data') and not directory.startswith('code/examples/tutorials') and directory != 'code':
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                # 统一格式的__init__.py头部
                # 计算相对于code目录的路径
                relative_path = directory.replace('code/', '')
                header = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# {relative_path}/__init__.py

"""
{relative_path.replace("/", ".")} 模块
"""
'''
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(header)
                print(f"创建文件: {init_file}")

def create_requirements():
    """创建requirements.txt文件"""
    requirements = [
        # 基础包
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        
        # 深度学习框架
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "transformers>=4.20.0",
        "tokenizers>=0.12.0",
        
        # 图神经网络
        "torch-geometric>=2.1.0",
        "networkx>=2.8.0",
        
        # 中文处理
        "jieba>=0.42.1",
        "pypinyin>=0.47.0",
        
        # 英文处理  
        "nltk>=3.7.0",
        "spacy>=3.4.0",
        
        # 图像处理
        "opencv-python>=4.6.0",
        "Pillow>=9.2.0",
        
        # 多模态
        "clip-by-openai>=1.0.0",
        
        # 数据处理
        "tqdm>=4.64.0",
        "datasets>=2.4.0",
        
        # 配置管理
        "pyyaml>=6.0.0",
        "hydra-core>=1.2.0",
        
        # 实验跟踪
        "wandb>=0.13.0",
        "tensorboard>=2.10.0",
        
        # 其他工具
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "emoji>=2.0.0"
    ]
    
    with open('code/requirements.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(requirements))
    print("创建文件: code/requirements.txt")

def create_environment_yml():
    """创建conda环境配置文件"""
    env_config = {
        'name': 'nlp-toolkit',
        'channels': ['pytorch', 'conda-forge', 'defaults'],
        'dependencies': [
            'python=3.9',
            'pytorch>=1.12.0',
            'torchvision>=0.13.0', 
            'cudatoolkit=11.6',
            'pip',
            {
                'pip': [
                    'transformers>=4.20.0',
                    'torch-geometric>=2.1.0',
                    'jieba>=0.42.1',
                    'nltk>=3.7.0',
                    'opencv-python>=4.6.0',
                    'wandb>=0.13.0'
                ]
            }
        ]
    }
    
    try:
        import yaml
        with open('code/environment.yml', 'w', encoding='utf-8') as f:
            yaml.dump(env_config, f, default_flow_style=False, allow_unicode=True)
        print("创建文件: code/environment.yml")
    except ImportError:
        print("警告: 需要安装pyyaml才能创建environment.yml")

if __name__ == "__main__":
    print("🚀 开始创建NLP技术实现代码库项目结构...")
    
    create_project_structure()
    create_requirements()
    create_environment_yml()
    
    print("\n✅ 项目结构创建完成！")
    print("\n下一步:")
    print("1. 运行: conda env create -f environment.yml")
    print("2. 运行: conda activate nlp-toolkit") 
    print("3. 运行: pip install -r requirements.txt")