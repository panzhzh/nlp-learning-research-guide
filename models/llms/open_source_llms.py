#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/llms/open_source_llms.py

"""
开源大语言模型实现
使用 Qwen3-0.6B 进行谣言检测任务
支持多种推理方式和参数高效微调
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    AutoConfig, BitsAndBytesConfig,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
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

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入项目模块
try:
    from datasets.data_loaders import create_all_dataloaders
    from utils.config_manager import get_config_manager, get_output_path
    from models.llms.prompt_engineering import RumorPromptTemplate, PromptManager
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块")
except ImportError as e:
    print(f"⚠️  导入项目模块失败: {e}")
    USE_PROJECT_MODULES = False


class QwenRumorClassifier:
    """基于Qwen3-0.6B的谣言检测分类器"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-0.6B",
                 device: str = "auto",
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False,
                 use_lora: bool = True,
                 max_length: int = 512):
        """
        初始化Qwen谣言分类器
        
        Args:
            model_name: 模型名称
            device: 计算设备
            load_in_8bit: 是否使用8bit量化
            load_in_4bit: 是否使用4bit量化
            use_lora: 是否使用LoRA微调
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_lora = use_lora
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"🖥️  使用设备: {self.device}")
        print(f"🤖 加载模型: {model_name}")
        
        # 设置量化配置
        self.quantization_config = None
        if load_in_4bit:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("🔧 启用4bit量化")
        elif load_in_8bit:
            self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("🔧 启用8bit量化")
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.prompt_manager = None
        
        # 标签映射
        self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # 设置输出目录
        if USE_PROJECT_MODULES:
            config_manager = get_config_manager()
            self.output_dir = get_output_path('models', 'llms')
        else:
            self.output_dir = Path('outputs/models/llms')
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 输出目录: {self.output_dir}")
        
        # 加载模型和分词器
        self._load_model_and_tokenizer()
        
        # 初始化提示管理器
        self._init_prompt_manager()
    
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
            
            print("📥 加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                quantization_config=self.quantization_config,
                device_map="auto" if self.device.type == 'cuda' else None
            )
            
            # 如果不是自动设备映射，手动移动到设备
            if self.quantization_config is None and self.device.type != 'cuda':
                self.model = self.model.to(self.device)
            
            # 设置模型为评估模式
            self.model.eval()
            
            print(f"✅ 模型加载成功")
            print(f"   参数量: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   词汇表大小: {len(self.tokenizer)}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def _init_prompt_manager(self):
        """初始化提示管理器"""
        try:
            if USE_PROJECT_MODULES:
                self.prompt_manager = PromptManager()
            else:
                # 创建简单的提示管理器
                self.prompt_manager = type('PromptManager', (), {
                    'create_classification_prompt': self._create_simple_prompt,
                    'create_few_shot_prompt': self._create_simple_few_shot_prompt
                })()
            
            print("✅ 提示管理器初始化完成")
            
        except Exception as e:
            print(f"⚠️  提示管理器初始化失败: {e}")
            self.prompt_manager = None
    
    def _create_simple_prompt(self, text: str, task_type: str = "classification") -> str:
        """创建简单的提示模板"""
        return f"""请分析以下文本是否为谣言。

文本内容: {text}

请从以下选项中选择一个答案：
- Non-rumor: 非谣言，内容真实可信
- Rumor: 谣言，内容虚假或误导
- Unverified: 未验证，无法确定真伪

答案: """
    
    def _create_simple_few_shot_prompt(self, text: str, examples: List[Dict] = None) -> str:
        """创建简单的少样本提示"""
        prompt = "以下是一些谣言检测的例子：\n\n"
        
        if examples:
            for i, example in enumerate(examples[:3], 1):  # 最多3个例子
                prompt += f"例子{i}:\n"
                prompt += f"文本: {example.get('text', '')}\n"
                prompt += f"标签: {example.get('label', '')}\n\n"
        else:
            # 默认例子
            prompt += """例子1:
文本: 科学家发现新的治疗方法，临床试验显示显著效果
标签: Non-rumor

例子2:
文本: 网传某地发生重大事故，但官方尚未确认
标签: Unverified

例子3:
文本: 谣传疫苗含有有害物质，已被科学研究证实为虚假信息
标签: Rumor

"""
        
        prompt += f"现在请分析以下文本:\n文本: {text}\n标签: "
        return prompt
    
    def setup_lora(self, 
                   r: int = 16,
                   lora_alpha: int = 32,
                   lora_dropout: float = 0.1,
                   target_modules: List[str] = None) -> None:
        """设置LoRA参数高效微调"""
        if not self.use_lora:
            print("⚠️  LoRA未启用")
            return
        
        if target_modules is None:
            # Qwen模型的注意力模块
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        try:
            print("🔧 设置LoRA配置...")
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            
            self.peft_model = get_peft_model(self.model, lora_config)
            
            # 统计可训练参数
            trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.peft_model.parameters())
            
            print(f"✅ LoRA设置完成")
            print(f"   可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            print(f"   总参数: {total_params:,}")
            
        except Exception as e:
            print(f"❌ LoRA设置失败: {e}")
            self.peft_model = None
    
    def generate_response(self, 
                         prompt: str, 
                         max_new_tokens: int = 50,
                         temperature: float = 0.1,
                         do_sample: bool = True,
                         top_p: float = 0.9) -> str:
        """生成模型响应"""
        try:
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 选择使用的模型
            model_to_use = self.peft_model if self.peft_model is not None else self.model
            
            # 生成响应
            with torch.no_grad():
                outputs = model_to_use.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码响应
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"生成响应失败: {e}")
            return ""
    
    def classify_text(self, text: str, use_few_shot: bool = False, examples: List[Dict] = None) -> Dict[str, Any]:
        """对单个文本进行分类"""
        try:
            # 创建提示
            if use_few_shot and self.prompt_manager:
                prompt = self.prompt_manager.create_few_shot_prompt(text, examples)
            elif self.prompt_manager:
                prompt = self.prompt_manager.create_classification_prompt(text)
            else:
                if use_few_shot:
                    prompt = self._create_simple_few_shot_prompt(text, examples)
                else:
                    prompt = self._create_simple_prompt(text)
            
            # 生成响应
            response = self.generate_response(prompt, max_new_tokens=20)
            
            # 解析响应
            predicted_label = self._parse_response(response)
            confidence = self._calculate_confidence(response, predicted_label)
            
            return {
                'text': text,
                'predicted_label': predicted_label,
                'predicted_class': self.label_mapping.get(predicted_label, 'Unknown'),
                'confidence': confidence,
                'raw_response': response,
                'prompt_used': 'few_shot' if use_few_shot else 'standard'
            }
            
        except Exception as e:
            logger.error(f"文本分类失败: {e}")
            return {
                'text': text,
                'predicted_label': 0,  # 默认为Non-rumor
                'predicted_class': 'Non-rumor',
                'confidence': 0.0,
                'raw_response': '',
                'error': str(e)
            }
    
    def _parse_response(self, response: str) -> int:
        """解析模型响应，提取预测标签"""
        response_lower = response.lower().strip()
        
        # 直接匹配标签
        if 'rumor' in response_lower and 'non-rumor' not in response_lower:
            return 1  # Rumor
        elif 'non-rumor' in response_lower:
            return 0  # Non-rumor
        elif 'unverified' in response_lower:
            return 2  # Unverified
        
        # 匹配中文
        if '谣言' in response_lower and '非谣言' not in response_lower:
            return 1
        elif '非谣言' in response_lower or '真实' in response_lower:
            return 0
        elif '未验证' in response_lower or '不确定' in response_lower:
            return 2
        
        # 默认返回Non-rumor
        return 0
    
    def _calculate_confidence(self, response: str, predicted_label: int) -> float:
        """计算预测置信度（简单启发式方法）"""
        response_lower = response.lower()
        predicted_class = self.label_mapping[predicted_label].lower()
        
        # 如果响应中包含预测的类别，置信度较高
        if predicted_class.replace('-', '').replace('_', '') in response_lower.replace('-', '').replace('_', ''):
            return 0.8
        else:
            return 0.5
    
    def batch_classify(self, texts: List[str], use_few_shot: bool = False, 
                      examples: List[Dict] = None, batch_size: int = 8) -> List[Dict[str, Any]]:
        """批量文本分类"""
        results = []
        
        print(f"🔄 开始批量分类 {len(texts)} 个文本...")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="批量分类"):
            batch_texts = texts[i:i+batch_size]
            
            for text in batch_texts:
                result = self.classify_text(text, use_few_shot, examples)
                results.append(result)
        
        return results
    
    def evaluate_on_dataset(self, use_few_shot: bool = False) -> Dict[str, Any]:
        """在数据集上评估模型性能"""
        print("📊 开始评估模型性能...")
        
        try:
            # 加载数据
            if USE_PROJECT_MODULES:
                dataloaders = create_all_dataloaders(
                    batch_sizes={'train': 32, 'val': 32, 'test': 32}
                )
                
                # 提取测试数据
                test_texts = []
                test_labels = []
                
                for batch in dataloaders['test']:
                    if 'text' in batch:
                        test_texts.extend(batch['text'])
                    elif 'caption' in batch:
                        test_texts.extend(batch['caption'])
                    
                    if 'labels' in batch:
                        test_labels.extend(batch['labels'].tolist())
                    elif 'label' in batch:
                        test_labels.extend(batch['label'])
                
            else:
                # 使用演示数据
                test_texts = [
                    "这是一个关于科技进步的真实新闻报道",
                    "网传某地发生重大事故，但尚未得到官方确认",
                    "谣传某知名公司即将倒闭，已被官方辟谣",
                    "科学研究表明新药物具有显著疗效",
                    "未经证实的传言在社交媒体广泛传播"
                ]
                test_labels = [0, 2, 1, 0, 2]
            
            print(f"📝 测试数据: {len(test_texts)} 个样本")
            
            # 少样本例子
            few_shot_examples = [
                {'text': '官方发布的权威新闻报道', 'label': 'Non-rumor'},
                {'text': '网上流传的未证实谣言', 'label': 'Rumor'},
                {'text': '需要进一步核实的信息', 'label': 'Unverified'}
            ] if use_few_shot else None
            
            # 批量分类
            results = self.batch_classify(
                test_texts, 
                use_few_shot=use_few_shot, 
                examples=few_shot_examples,
                batch_size=4  # 减小批次大小
            )
            
            # 提取预测结果
            predictions = [r['predicted_label'] for r in results]
            
            # 计算评估指标
            accuracy = accuracy_score(test_labels, predictions)
            f1_macro = f1_score(test_labels, predictions, average='macro')
            f1_weighted = f1_score(test_labels, predictions, average='weighted')
            
            # 分类报告
            report = classification_report(
                test_labels, predictions,
                target_names=list(self.label_mapping.values()),
                output_dict=True
            )
            
            evaluation_result = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'classification_report': report,
                'num_samples': len(test_texts),
                'use_few_shot': use_few_shot,
                'model_name': self.model_name,
                'predictions': predictions,
                'true_labels': test_labels,
                'detailed_results': results
            }
            
            print(f"✅ 评估完成:")
            print(f"   准确率: {accuracy:.4f}")
            print(f"   F1分数(macro): {f1_macro:.4f}")
            print(f"   F1分数(weighted): {f1_weighted:.4f}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            return {'error': str(e)}
    
    def save_model(self, save_path: Optional[str] = None):
        """保存模型和配置"""
        if save_path is None:
            save_path = self.output_dir / f"qwen_rumor_classifier"
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 保存分词器
            self.tokenizer.save_pretrained(save_path / "tokenizer")
            
            # 保存LoRA模型（如果存在）
            if self.peft_model is not None:
                self.peft_model.save_pretrained(save_path / "lora_model")
                print(f"✅ LoRA模型已保存到: {save_path / 'lora_model'}")
            else:
                # 保存完整模型
                self.model.save_pretrained(save_path / "model")
                print(f"✅ 完整模型已保存到: {save_path / 'model'}")
            
            # 保存配置
            config = {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'use_lora': self.use_lora,
                'label_mapping': self.label_mapping,
                'device': str(self.device)
            }
            
            with open(save_path / "config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 分词器已保存到: {save_path / 'tokenizer'}")
            print(f"✅ 配置已保存到: {save_path / 'config.json'}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise


def create_qwen_classifier(use_lora: bool = True, 
                          load_in_4bit: bool = False) -> QwenRumorClassifier:
    """创建Qwen谣言分类器的便捷函数"""
    print("🚀 创建Qwen谣言分类器...")
    
    classifier = QwenRumorClassifier(
        model_name="Qwen/Qwen3-0.6B",
        use_lora=use_lora,
        load_in_4bit=load_in_4bit,
        max_length=512
    )
    
    if use_lora:
        classifier.setup_lora(r=16, lora_alpha=32, lora_dropout=0.1)
    
    return classifier


def demo_qwen_classification():
    """演示Qwen分类功能"""
    print("🎯 Qwen谣言检测演示")
    print("=" * 50)
    
    try:
        # 创建分类器
        classifier = create_qwen_classifier(use_lora=True, load_in_4bit=False)
        
        # 测试单个文本分类
        test_texts = [
            "科学家在实验室发现了新的治疗方法，经过严格的临床试验证实有效",
            "网传某地发生重大地震，但官方气象局尚未发布相关信息",
            "谣传新冠疫苗含有微芯片，这一说法已被多项科学研究证明为虚假信息"
        ]
        
        print("\n🔍 单个文本分类测试:")
        for i, text in enumerate(test_texts, 1):
            print(f"\n文本 {i}: {text}")
            result = classifier.classify_text(text)
            print(f"预测: {result['predicted_class']} (置信度: {result['confidence']:.2f})")
            print(f"原始响应: {result['raw_response']}")
        
        # 少样本学习测试
        print(f"\n🎯 少样本学习测试:")
        few_shot_examples = [
            {'text': '政府官方发布的权威声明', 'label': 'Non-rumor'},
            {'text': '网络上流传的未经证实的传言', 'label': 'Rumor'}
        ]
        
        test_text = "专家学者在学术期刊上发表的研究成果"
        result = classifier.classify_text(test_text, use_few_shot=True, examples=few_shot_examples)
        print(f"文本: {test_text}")
        print(f"少样本预测: {result['predicted_class']} (置信度: {result['confidence']:.2f})")
        
        # 数据集评估
        print(f"\n📊 数据集评估:")
        eval_result = classifier.evaluate_on_dataset(use_few_shot=False)
        print(f"标准提示准确率: {eval_result['accuracy']:.4f}")
        
        eval_result_few_shot = classifier.evaluate_on_dataset(use_few_shot=True)
        print(f"少样本提示准确率: {eval_result_few_shot['accuracy']:.4f}")
        
        # 保存模型
        print(f"\n💾 保存模型...")
        classifier.save_model()
        
        print(f"\n✅ Qwen分类演示完成!")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        raise


if __name__ == "__main__":
    demo_qwen_classification()