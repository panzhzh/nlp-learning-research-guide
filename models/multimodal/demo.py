#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/multimodal/demo.py

"""
多模态模型演示 - 简洁版本
"""

import sys
from pathlib import Path

# 快速路径设置
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from models.multimodal.vision_language_models import MultiModalTrainer
    from utils.config_manager import check_data_requirements
    print("✅ 导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def main():
    print("🎭 多模态模型演示")
    print("="*50)
    
    try:
        # 检查数据
        check_data_requirements()
        print("✅ 数据检查通过")
        
        # 创建训练器并训练
        trainer = MultiModalTrainer()
        trainer.train_all_models(
            epochs=2,           # 快速演示
            batch_size=4,       # 小批次
            learning_rate=5e-5  # 适中学习率
        )
        
        print("\n✅ 演示完成!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请确保数据集已下载到data目录")

if __name__ == "__main__":
    main()