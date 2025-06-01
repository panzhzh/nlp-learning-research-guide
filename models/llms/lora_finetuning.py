#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/llms/lora_finetuning.py

"""
基于 Qwen3-0.6B 的 LoRA 微调模块
专门用于谣言检测任务的参数高效微调
支持多种微调策略和评估功能
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    PeftModel, prepare_model_for_kbit_training
)
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import json
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import sys
import logging
from datetime import datetime
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(project_root)
# 导入项目模块
try:
    from data_utils.data_loaders import create_all_dataloaders
    from utils.config_manager import get_config_manager, get_output_path
    from models.llms.prompt_engineering import PromptManager
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块")
except ImportError as e:
    print(f"⚠️  导入项目模块失败: {e}")
    USE_PROJECT_MODULES = False


class RumorDetectionDataset(Dataset):
    """谣言检测数据集类"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
        
        # 创建提示模板
        self.prompt_template = "请判断以下文本是否为谣言。\n\n文本: {text}\n\n请从以下选项中选择:\n- Non-rumor: 非谣言\n- Rumor: 谣言\n- Unverified: 未验证\n\n答案: {label}"
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        label_text = self.label_mapping[label]
        
        # 创建完整的提示
        prompt = self.prompt_template.format(text=text, label=label_text)
        
        # 分词
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten().clone()
        }


class QwenLoRATrainer:
    """Qwen LoRA 微调训练器"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-0.6B",
                 max_length: int = 512,
                 device: str = "auto"):
        """
        初始化 LoRA 训练器
        
        Args:
            model_name: 模型名称
            max_length: 最大序列长度
            device: 计算设备
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  使用设备: {self.device}")
        print(f"🤖 模型: {model_name}")
        
        # 初始化组件
        self.tokenizer = None
        self.base_model = None
        self.peft_model = None
        self.lora_config = None
        
        # 设置输出目录
        if USE_PROJECT_MODULES:
            config_manager = get_config_manager()
            self.output_dir = get_output_path('models', 'llms') / 'lora_checkpoints'
        else:
            self.output_dir = Path('outputs/models/llms/lora_checkpoints')
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 输出目录: {self.output_dir}")
        
        # 加载模型和分词器
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """加载模型和分词器"""
        try:
            print("📥 加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                pad_token='<|extra_0|>',
                eos_token='<|im_end|>',
                use_fast=False
            )
            
            # 确保有pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("📥 加载基础模型...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None
            )
            
            # 如果不是自动设备映射，手动移动到设备
            if self.device.type != 'cuda':
                self.base_model = self.base_model.to(self.device)
            
            print(f"✅ 模型加载成功")
            print(f"   参数量: {sum(p.numel() for p in self.base_model.parameters()):,}")
            print(f"   词汇表大小: {len(self.tokenizer)}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def setup_lora_config(self,
                         r: int = 16,
                         lora_alpha: int = 32,
                         lora_dropout: float = 0.1,
                         target_modules: Optional[List[str]] = None,
                         bias: str = "none",
                         task_type: str = "CAUSAL_LM"):
        """
        设置 LoRA 配置
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: 目标模块
            bias: 偏置处理方式
            task_type: 任务类型
        """
        if target_modules is None:
            # Qwen3 模型的注意力模块
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        self.lora_config = LoraConfig(
            task_type=getattr(TaskType, task_type),
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            inference_mode=False
        )
        
        print(f"🔧 LoRA 配置:")
        print(f"   Rank (r): {r}")
        print(f"   Alpha: {lora_alpha}")
        print(f"   Dropout: {lora_dropout}")
        print(f"   目标模块: {target_modules}")
        
        return self.lora_config
    
    def create_peft_model(self):
        """创建 PEFT 模型"""
        if self.lora_config is None:
            raise ValueError("请先调用 setup_lora_config() 设置 LoRA 配置")
        
        try:
            print("🔧 创建 PEFT 模型...")
            
            # 准备模型进行量化训练（如果需要）
            if hasattr(self.base_model, 'gradient_checkpointing_enable'):
                self.base_model.gradient_checkpointing_enable()
            
            # 创建 PEFT 模型
            self.peft_model = get_peft_model(self.base_model, self.lora_config)
            
            # 统计参数
            total_params = sum(p.numel() for p in self.peft_model.parameters())
            trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
            
            print(f"✅ PEFT 模型创建成功")
            print(f"   总参数: {total_params:,}")
            print(f"   可训练参数: {trainable_params:,}")
            print(f"   可训练比例: {100 * trainable_params / total_params:.2f}%")
            
            return self.peft_model
            
        except Exception as e:
            print(f"❌ PEFT 模型创建失败: {e}")
            raise
    
    def prepare_datasets(self):
        """准备训练数据集"""
        try:
            print("📊 准备训练数据集...")
            
            if USE_PROJECT_MODULES:
                # 直接读取JSON文件，使用全量训练数据
                from utils.config_manager import get_data_dir
                data_dir = get_data_dir()
                
                datasets = {}
                
                # 读取训练集和测试集文件
                for split in ['train', 'test']:
                    file_path = data_dir / f'dataset_items_{split}.json'
                    
                    if file_path.exists():
                        print(f"📄 读取 {split} 数据文件: {file_path}")
                        
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        texts = []
                        labels = []
                        
                        # 提取caption和label字段
                        for item_id, item_data in data.items():
                            caption = item_data.get('caption', '')
                            label = item_data.get('label', 0)
                            
                            if caption and caption.strip():  # 确保caption不为空
                                texts.append(caption.strip())
                                labels.append(int(label))
                        
                        # 创建数据集
                        datasets[split] = RumorDetectionDataset(
                            texts=texts,
                            labels=labels,
                            tokenizer=self.tokenizer,
                            max_length=self.max_length
                        )
                        
                        print(f"   {split}: {len(datasets[split])} 样本")
                        
                        # 打印标签分布
                        label_counts = {}
                        for label in labels:
                            label_counts[label] = label_counts.get(label, 0) + 1
                        print(f"   标签分布: {label_counts}")
                    
                    else:
                        print(f"⚠️  数据文件不存在: {file_path}")
                
                # 如果没有验证集，从训练集中分割一部分作为验证集
                if 'train' in datasets and 'val' not in datasets:
                    train_dataset = datasets['train']
                    train_size = len(train_dataset)
                    val_size = min(100, train_size // 10)  # 验证集大小为训练集的10%或100个样本
                    
                    # 简单分割
                    val_texts = train_dataset.texts[:val_size]
                    val_labels = train_dataset.labels[:val_size]
                    
                    datasets['val'] = RumorDetectionDataset(
                        texts=val_texts,
                        labels=val_labels,
                        tokenizer=self.tokenizer,
                        max_length=self.max_length
                    )
                    
                    # 更新训练集（移除验证集部分）
                    train_texts = train_dataset.texts[val_size:]
                    train_labels = train_dataset.labels[val_size:]
                    
                    datasets['train'] = RumorDetectionDataset(
                        texts=train_texts,
                        labels=train_labels,
                        tokenizer=self.tokenizer,
                        max_length=self.max_length
                    )
                    
                    print(f"   从训练集分割验证集: {len(datasets['val'])} 样本")
                    print(f"   更新后训练集: {len(datasets['train'])} 样本")
            
            else:
                # 使用演示数据
                demo_data = {
                    'train': {
                        'texts': [
                            "科学家在实验室发现了新的治疗方法，经过严格的临床试验证实有效",
                            "网传某地发生重大地震，但官方气象局尚未发布相关信息", 
                            "谣传新冠疫苗含有微芯片，这一说法已被多项科学研究证明为虚假信息",
                            "教育部正式发布新的高考改革方案，将于明年开始实施",
                            "据不完全统计，新产品在市场上反响良好",
                            "世界卫生组织确认新冠疫苗对变异株仍然有效"
                        ],
                        'labels': [0, 2, 1, 0, 2, 0]
                    },
                    'val': {
                        'texts': [
                            "中国科学院发布最新研究成果，在人工智能领域取得重大突破",
                            "网上流传某明星涉嫌违法犯罪，但当事人已发声明辟谣"
                        ],
                        'labels': [0, 1]
                    },
                    'test': {
                        'texts': [
                            "政府部门发布官方声明，澄清网络传言",
                            "业内人士透露，某行业可能面临重大政策调整"
                        ],
                        'labels': [0, 2]
                    }
                }
                
                datasets = {}
                for split in ['train', 'val', 'test']:
                    datasets[split] = RumorDetectionDataset(
                        texts=demo_data[split]['texts'],
                        labels=demo_data[split]['labels'],
                        tokenizer=self.tokenizer,
                        max_length=self.max_length
                    )
                    print(f"   {split}: {len(datasets[split])} 样本")
            
            print("✅ 数据集准备完成")
            return datasets
            
        except Exception as e:
            print(f"❌ 数据集准备失败: {e}")
            raise
    
    def create_training_arguments(self,
                                output_dir: Optional[str] = None,
                                num_train_epochs: int = 3,
                                per_device_train_batch_size: int = 4,
                                per_device_eval_batch_size: int = 8,
                                learning_rate: float = 2e-4,
                                warmup_steps: int = 100,
                                logging_steps: int = 10,
                                save_steps: int = 500,
                                eval_steps: int = 500,
                                save_total_limit: int = 2,
                                load_best_model_at_end: bool = True,
                                metric_for_best_model: str = "eval_loss",
                                greater_is_better: bool = False):
        """
        创建训练参数
        
        Args:
            output_dir: 输出目录
            num_train_epochs: 训练轮数
            per_device_train_batch_size: 训练批次大小
            per_device_eval_batch_size: 评估批次大小
            learning_rate: 学习率
            warmup_steps: 预热步数
            logging_steps: 日志记录步数
            save_steps: 保存步数
            eval_steps: 评估步数
            save_total_limit: 最大保存检查点数
            load_best_model_at_end: 是否在结束时加载最佳模型
            metric_for_best_model: 最佳模型评估指标
            greater_is_better: 指标是否越大越好
            
        Returns:
            TrainingArguments 对象
        """
        if output_dir is None:
            output_dir = self.output_dir / f"qwen_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",  # 修复: 使用 eval_strategy 而不是 evaluation_strategy
            save_strategy="steps",
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            remove_unused_columns=False,
            report_to=None,  # 禁用 wandb 等外部报告
            dataloader_pin_memory=False,
            fp16=True if self.device.type == 'cuda' else False,
            gradient_checkpointing=True
        )
        
        print(f"🔧 训练参数配置:")
        print(f"   输出目录: {output_dir}")
        print(f"   训练轮数: {num_train_epochs}")
        print(f"   训练批次大小: {per_device_train_batch_size}")
        print(f"   学习率: {learning_rate}")
        print(f"   预热步数: {warmup_steps}")
        
        return training_args
    
    def train(self,
              train_dataset,
              eval_dataset,
              training_args,
              compute_metrics_fn=None):
        """
        开始训练
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            training_args: 训练参数
            compute_metrics_fn: 指标计算函数
            
        Returns:
            训练结果
        """
        if self.peft_model is None:
            raise ValueError("请先调用 create_peft_model() 创建 PEFT 模型")
        
        try:
            print("🚀 开始 LoRA 微调训练...")
            
            # 创建数据整理器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # 创建训练器
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn
            )
            
            # 开始训练
            train_result = trainer.train()
            
            # 保存模型
            trainer.save_model()
            
            print("✅ LoRA 微调训练完成")
            print(f"   训练损失: {train_result.training_loss:.4f}")
            print(f"   训练步数: {train_result.global_step}")
            
            return train_result
            
        except Exception as e:
            print(f"❌ LoRA 微调训练失败: {e}")
            raise
    
    def evaluate_model(self, test_dataset, model_path: Optional[str] = None):
        """
        评估微调后的模型
        
        Args:
            test_dataset: 测试数据集
            model_path: 模型路径（可选）
            
        Returns:
            评估结果
        """
        try:
            print("📊 评估微调后的模型...")
            
            # 如果指定了模型路径，加载模型
            if model_path:
                model = PeftModel.from_pretrained(self.base_model, model_path)
            else:
                model = self.peft_model
            
            if model is None:
                raise ValueError("没有可用的模型进行评估")
            
            model.eval()
            
            predictions = []
            true_labels = []
            
            # 逐个样本进行预测（避免批处理的复杂性）
            print(f"开始评估 {len(test_dataset)} 个测试样本...")
            
            for i in tqdm(range(len(test_dataset)), desc="评估中"):
                try:
                    # 获取单个样本
                    sample = test_dataset[i]
                    input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                    attention_mask = sample['attention_mask'].unsqueeze(0).to(self.device)
                    
                    # 构建原始提示（不包含答案）
                    text = test_dataset.texts[i]
                    prompt = f"请判断以下文本是否为谣言。\n\n文本: {text}\n\n请从以下选项中选择:\n- Non-rumor: 非谣言\n- Rumor: 谣言\n- Unverified: 未验证\n\n答案: "
                    
                    # 重新编码提示（不包含答案）
                    prompt_encoding = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=self.max_length,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    # 生成预测
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=prompt_encoding['input_ids'],
                            attention_mask=prompt_encoding['attention_mask'],
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            temperature=0.1
                        )
                    
                    # 解码并解析预测结果
                    generated = outputs[0][prompt_encoding['input_ids'].shape[1]:]
                    generated_text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                    
                    # 解析标签
                    predicted_label = self._parse_generated_label(generated_text)
                    predictions.append(predicted_label)
                    
                    # 获取真实标签
                    true_labels.append(test_dataset.labels[i])
                    
                except Exception as e:
                    print(f"⚠️  评估第 {i} 个样本时出错: {e}")
                    # 使用默认预测
                    predictions.append(0)
                    true_labels.append(test_dataset.labels[i])
            
            # 计算评估指标
            accuracy = accuracy_score(true_labels, predictions)
            f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
            f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
            
            # 分类报告
            class_names = ['Non-rumor', 'Rumor', 'Unverified']
            report = classification_report(
                true_labels, predictions,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            
            evaluation_result = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'classification_report': report,
                'num_samples': len(true_labels),
                'predictions': predictions,
                'true_labels': true_labels
            }
            
            print(f"✅ 模型评估完成:")
            print(f"   准确率: {accuracy:.4f}")
            print(f"   F1分数(macro): {f1_macro:.4f}")
            print(f"   F1分数(weighted): {f1_weighted:.4f}")
            
            # 打印一些预测样例
            print(f"\n📋 预测样例:")
            for i in range(min(3, len(true_labels))):
                print(f"   样例 {i+1}:")
                print(f"     文本: {test_dataset.texts[i][:50]}...")
                print(f"     真实: {class_names[true_labels[i]]}")
                print(f"     预测: {class_names[predictions[i]]}")
            
            return evaluation_result
            
        except Exception as e:
            print(f"❌ 模型评估失败: {e}")
            raise
    
    def _parse_generated_label(self, generated_text: str) -> int:
        """解析生成的文本中的标签"""
        text_lower = generated_text.lower().strip()
        
        if 'rumor' in text_lower and 'non-rumor' not in text_lower:
            return 1  # Rumor
        elif 'non-rumor' in text_lower:
            return 0  # Non-rumor
        elif 'unverified' in text_lower:
            return 2  # Unverified
        else:
            return 0  # 默认为 Non-rumor
    
    def save_lora_model(self, save_path: Optional[str] = None):
        """
        保存 LoRA 模型
        
        Args:
            save_path: 保存路径
        """
        if self.peft_model is None:
            raise ValueError("没有可保存的 PEFT 模型")
        
        if save_path is None:
            save_path = self.output_dir / f"qwen_lora_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 保存 LoRA 模型
            self.peft_model.save_pretrained(save_path)
            
            # 保存分词器
            self.tokenizer.save_pretrained(save_path)
            
            # 保存配置
            config = {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'lora_config': {
                    'r': self.lora_config.r,
                    'lora_alpha': self.lora_config.lora_alpha,
                    'lora_dropout': self.lora_config.lora_dropout,
                    'target_modules': list(self.lora_config.target_modules) if hasattr(self.lora_config.target_modules, '__iter__') else self.lora_config.target_modules,
                    'bias': self.lora_config.bias,
                    'task_type': str(self.lora_config.task_type)
                } if self.lora_config else None,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(save_path / 'training_config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"✅ LoRA 模型已保存到: {save_path}")
            
        except Exception as e:
            print(f"❌ 保存 LoRA 模型失败: {e}")
            raise
    
    def load_lora_model(self, model_path: str):
        """
        加载 LoRA 模型
        
        Args:
            model_path: 模型路径
        """
        try:
            print(f"📥 加载 LoRA 模型: {model_path}")
            
            # 加载 PEFT 模型
            self.peft_model = PeftModel.from_pretrained(self.base_model, model_path)
            
            # 加载配置
            config_path = Path(model_path) / 'training_config.json'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"   配置: {config}")
            
            print("✅ LoRA 模型加载成功")
            
        except Exception as e:
            print(f"❌ 加载 LoRA 模型失败: {e}")
            raise


def create_compute_metrics_fn(tokenizer):
    """创建指标计算函数"""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # 这里可以添加更复杂的指标计算
        # 目前返回简单的损失
        return {"eval_loss": predictions.mean()}
    
    return compute_metrics


def demo_lora_finetuning():
    """演示 LoRA 微调功能"""
    print("🚀 Qwen3-0.6B LoRA 微调演示")
    print("=" * 60)
    
    try:
        # 1. 创建训练器
        print("1. 创建 LoRA 训练器...")
        trainer = QwenLoRATrainer(
            model_name="Qwen/Qwen3-0.6B",
            max_length=512
        )
        
        # 2. 设置 LoRA 配置
        print("\n2. 设置 LoRA 配置...")
        trainer.setup_lora_config(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        
        # 3. 创建 PEFT 模型
        print("\n3. 创建 PEFT 模型...")
        trainer.create_peft_model()
        
        # 4. 准备数据集
        print("\n4. 准备数据集...")
        datasets = trainer.prepare_datasets()
        
        # 5. 创建训练参数
        print("\n5. 创建训练参数...")
        training_args = trainer.create_training_arguments(
            num_train_epochs=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            learning_rate=2e-4,
            warmup_steps=50,
            logging_steps=5,
            save_steps=100,
            eval_steps=100
        )
        
        # 6. 开始训练
        print("\n6. 开始 LoRA 微调训练...")
        compute_metrics = create_compute_metrics_fn(trainer.tokenizer)
        
        train_result = trainer.train(
            train_dataset=datasets['train'],
            eval_dataset=datasets['val'],
            training_args=training_args,
            compute_metrics_fn=compute_metrics
        )
        
        # 7. 评估模型
        print("\n7. 评估微调后的模型...")
        eval_result = trainer.evaluate_model(datasets['test'])
        
        # 8. 保存模型
        print("\n8. 保存 LoRA 模型...")
        trainer.save_lora_model()
        
        print(f"\n✅ LoRA 微调演示完成!")
        print(f"   最终准确率: {eval_result['accuracy']:.4f}")
        print(f"   最终 F1 分数: {eval_result['f1_macro']:.4f}")
        
    except Exception as e:
        print(f"❌ LoRA 微调演示失败: {e}")
        raise


def quick_lora_training():
    """快速 LoRA 训练函数"""
    print("⚡ 快速 LoRA 训练")
    print("=" * 40)
    
    try:
        # 创建训练器
        trainer = QwenLoRATrainer()
        
        # 简化配置
        trainer.setup_lora_config(r=8, lora_alpha=16)
        trainer.create_peft_model()
        
        # 准备数据
        datasets = trainer.prepare_datasets()
        
        # 快速训练参数
        training_args = trainer.create_training_arguments(
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            learning_rate=1e-4,
            warmup_steps=10,
            logging_steps=2,
            save_steps=50,
            eval_steps=50
        )
        
        # 训练
        trainer.train(
            train_dataset=datasets['train'],
            eval_dataset=datasets['val'],
            training_args=training_args
        )
        
        # 快速评估
        result = trainer.evaluate_model(datasets['test'])
        print(f"快速训练结果 - 准确率: {result['accuracy']:.4f}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ 快速训练失败: {e}")
        return None


class LoRAInferenceEngine:
    """LoRA 推理引擎"""
    
    def __init__(self, base_model_name: str = "Qwen/Qwen3-0.6B", lora_model_path: str = None):
        """
        初始化推理引擎
        
        Args:
            base_model_name: 基础模型名称
            lora_model_path: LoRA 模型路径
        """
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            print(f"📥 加载推理模型...")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                use_fast=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None
            )
            
            # 如果有 LoRA 模型路径，加载 LoRA 权重
            if self.lora_model_path and Path(self.lora_model_path).exists():
                print(f"📥 加载 LoRA 权重: {self.lora_model_path}")
                self.model = PeftModel.from_pretrained(base_model, self.lora_model_path)
            else:
                print("⚠️  未指定 LoRA 模型路径，使用基础模型")
                self.model = base_model
            
            self.model.eval()
            
            if self.device.type != 'cuda':
                self.model = self.model.to(self.device)
            
            print("✅ 推理模型加载成功")
            
        except Exception as e:
            print(f"❌ 推理模型加载失败: {e}")
            raise
    
    def predict(self, text: str, max_new_tokens: int = 20) -> Dict[str, Any]:
        """
        对单个文本进行推理预测
        
        Args:
            text: 输入文本
            max_new_tokens: 最大生成token数
            
        Returns:
            预测结果字典
        """
        try:
            # 创建提示
            prompt = f"请判断以下文本是否为谣言。\n\n文本: {text}\n\n请从以下选项中选择:\n- Non-rumor: 非谣言\n- Rumor: 谣言\n- Unverified: 未验证\n\n答案: "
            
            # 分词
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成预测
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=0.1
                )
            
            # 解码结果
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            # 解析标签
            predicted_label = self._parse_prediction(generated_text)
            label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
            
            return {
                'text': text,
                'predicted_label': predicted_label,
                'predicted_class': label_mapping[predicted_label],
                'raw_output': generated_text,
                'confidence': self._calculate_confidence(generated_text)
            }
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return {
                'text': text,
                'predicted_label': 0,
                'predicted_class': 'Non-rumor',
                'raw_output': '',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _parse_prediction(self, generated_text: str) -> int:
        """解析预测结果"""
        text_lower = generated_text.lower()
        
        if 'rumor' in text_lower and 'non-rumor' not in text_lower:
            return 1  # Rumor
        elif 'non-rumor' in text_lower:
            return 0  # Non-rumor
        elif 'unverified' in text_lower:
            return 2  # Unverified
        else:
            return 0  # 默认
    
    def _calculate_confidence(self, generated_text: str) -> float:
        """计算置信度（简单启发式）"""
        if any(keyword in generated_text.lower() for keyword in ['non-rumor', 'rumor', 'unverified']):
            return 0.8
        else:
            return 0.5
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量预测"""
        results = []
        for text in tqdm(texts, desc="批量预测"):
            result = self.predict(text)
            results.append(result)
        return results


def demo_lora_inference():
    """演示 LoRA 推理功能"""
    print("🔍 LoRA 推理演示")
    print("=" * 40)
    
    try:
        # 创建推理引擎（使用基础模型，因为可能没有训练好的 LoRA 模型）
        engine = LoRAInferenceEngine()
        
        # 测试文本
        test_texts = [
            "科学家发现新的治疗方法，经过严格临床试验验证",
            "网传某地发生重大事故，但官方尚未确认",
            "谣传某产品含有有害成分，已被科学研究证实为虚假信息"
        ]
        
        print("📝 单个预测测试:")
        for i, text in enumerate(test_texts, 1):
            result = engine.predict(text)
            print(f"\n文本 {i}: {text}")
            print(f"预测: {result['predicted_class']} (置信度: {result['confidence']:.2f})")
            print(f"原始输出: {result['raw_output']}")
        
        print(f"\n📊 批量预测测试:")
        batch_results = engine.batch_predict(test_texts)
        for i, result in enumerate(batch_results, 1):
            print(f"文本 {i}: {result['predicted_class']}")
        
        print("✅ LoRA 推理演示完成")
        
    except Exception as e:
        print(f"❌ LoRA 推理演示失败: {e}")


def advanced_lora_training():
    """高级 LoRA 训练功能"""
    print("🎯 高级 LoRA 训练")
    print("=" * 40)
    
    try:
        # 创建训练器
        trainer = QwenLoRATrainer()
        
        # 高级 LoRA 配置
        trainer.setup_lora_config(
            r=32,  # 更大的 rank
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head"  # 包含更多层
            ]
        )
        
        trainer.create_peft_model()
        datasets = trainer.prepare_datasets()
        
        # 高级训练参数
        training_args = trainer.create_training_arguments(
            num_train_epochs=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            learning_rate=1e-4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=200,
            eval_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        # 训练
        train_result = trainer.train(
            train_dataset=datasets['train'],
            eval_dataset=datasets['val'],
            training_args=training_args
        )
        
        # 详细评估
        eval_result = trainer.evaluate_model(datasets['test'])
        
        # 保存模型
        save_path = trainer.output_dir / "advanced_lora_model"
        trainer.save_lora_model(save_path)
        
        print(f"✅ 高级 LoRA 训练完成")
        print(f"   训练损失: {train_result.training_loss:.4f}")
        print(f"   测试准确率: {eval_result['accuracy']:.4f}")
        print(f"   模型保存至: {save_path}")
        
        return trainer, eval_result
        
    except Exception as e:
        print(f"❌ 高级 LoRA 训练失败: {e}")
        return None, None


# 主执行代码
if __name__ == "__main__":
    print("🚀 Qwen3-0.6B LoRA 微调系统")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description="LoRA 微调系统")
    parser.add_argument("--mode", type=str, default="demo", 
                       choices=["demo", "quick", "advanced", "inference"],
                       help="运行模式")
    parser.add_argument("--model_path", type=str, default=None,
                       help="LoRA 模型路径（推理模式使用）")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "demo":
            demo_lora_finetuning()
        elif args.mode == "quick":
            quick_lora_training()
        elif args.mode == "advanced":
            advanced_lora_training()
        elif args.mode == "inference":
            demo_lora_inference()
        
        print(f"\n🎉 {args.mode.upper()} 模式执行完成!")
        
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()