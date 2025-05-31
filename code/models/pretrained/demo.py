#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/pretrained/demo.py

"""
预训练模型演示 - 严格版本
必须使用真实数据集，找不到数据就报错
"""

import sys
from pathlib import Path

# 添加项目路径
current_file = Path(__file__).resolve()
code_root = current_file.parent.parent.parent
sys.path.append(str(code_root))

try:
    from encoder_models import PretrainedModelTrainer
    from utils.config_manager import get_data_dir, check_data_requirements
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保从正确的项目目录运行")
    sys.exit(1)

def main():
    """严格的预训练模型训练演示"""
    print("🤗 预训练模型演示 (严格模式)")
    print("="*50)
    
    try:
        # 检查数据要求
        print("🔍 检查数据要求...")
        check_data_requirements()
        print("✅ 数据要求检查通过")
        
        # 获取数据目录
        data_dir = get_data_dir()
        print(f"📂 使用数据目录: {data_dir}")
        
        # 创建训练器
        trainer = PretrainedModelTrainer(data_dir=str(data_dir))
        
        # 快速训练演示（使用较少的epoch和较小的模型）
        print("\n🚀 开始快速训练演示...")
        trainer.train_all_models(
            model_keys=['bert-base-uncased'],  # 只训练BERT
            epochs=2,                          # 快速演示，只训练2轮
            batch_size=8,                      # 小批次，适应各种硬件
            learning_rate=2e-5,               # 标准学习率
            max_length=128                     # 较短序列，加快训练
        )
        
        # 演示预测功能
        print("\n🔮 预测演示:")
        test_texts = [
            "这是一个关于新技术发展的真实新闻报道",
            "网传某地发生重大事故，但官方尚未确认消息",
            "This might be fake news about celebrities"
        ]
        
        # 如果有训练好的模型，演示预测
        if trainer.models and trainer.tokenizers:
            model_key = list(trainer.models.keys())[0]
            print(f"使用模型: {model_key}")
            
            try:
                model = trainer.models[model_key]
                tokenizer = trainer.tokenizers[model_key]
                model.eval()
                
                import torch
                
                for text in test_texts:
                    # 简单预测演示
                    encoding = tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=128,
                        return_tensors='pt'
                    )
                    
                    with torch.no_grad():
                        logits = model(
                            encoding['input_ids'].to(trainer.device),
                            encoding['attention_mask'].to(trainer.device)
                        )
                        predicted = torch.argmax(logits, dim=1).item()
                        prediction_label = trainer.label_mapping.get(predicted, 'Unknown')
                        
                        # 计算置信度
                        probabilities = torch.softmax(logits, dim=1)
                        confidence = probabilities.max().item()
                    
                    print(f"文本: {text[:30]}...")
                    print(f"预测: {prediction_label} (置信度: {confidence:.3f})")
                    print()
            except Exception as e:
                print(f"预测演示失败: {e}")
        else:
            print("没有可用的训练模型进行预测演示")
        
        print("\n📊 预训练模型优势:")
        print("1. 强大的语义理解能力")
        print("2. 多语言支持")
        print("3. 迁移学习效果显著")
        print("4. 工业界广泛应用")
        
        print("\n✅ 预训练模型演示完成!")
        
    except FileNotFoundError as e:
        print(f"❌ 数据文件错误: {e}")
        print("\n💡 解决方案:")
        print("1. 确保MR2数据集已下载并解压")
        print("2. 检查数据目录路径是否正确")
        print("3. 验证所有数据文件都存在")
        print("下载链接: https://pan.baidu.com/s1sfUwsaeV2nfl54OkrfrKVw?pwd=jxhc")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        print("请检查数据集和环境配置")
        sys.exit(1)

if __name__ == "__main__":
    main()