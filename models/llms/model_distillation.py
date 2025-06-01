#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# models/llms/model_distillation.py

"""
模型蒸馏和压缩模块
实现知识蒸馏、模型剪枝、量化等压缩技术
将大模型的知识迁移到小模型中，提升部署效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import sys
import logging
from tqdm import tqdm
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入项目模块
try:
    from data_utils.data_loaders import create_all_dataloaders
    from utils.config_manager import get_config_manager, get_output_path
    from models.llms.open_source_llms import QwenRumorClassifier
    USE_PROJECT_MODULES = True
    print("✅ 成功导入项目模块")
except ImportError as e:
    print(f"⚠️  导入项目模块失败: {e}")
    USE_PROJECT_MODULES = False


@dataclass
class DistillationConfig:
    """蒸馏配置"""
    teacher_model: str = "Qwen/Qwen3-0.6B"
    student_model: str = "Qwen/Qwen3-0.6B"  # 或者更小的模型
    temperature: float = 4.0
    alpha: float = 0.7  # 蒸馏损失权重
    beta: float = 0.3   # 真实标签损失权重
    max_length: int = 512
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 8
    save_dir: str = "outputs/distillation"
    use_quantization: bool = False
    use_pruning: bool = False


class KnowledgeDistillationDataset(Dataset):
    """知识蒸馏数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_length: 最大长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TeacherStudentLoss(nn.Module):
    """师生网络损失函数"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        """
        初始化损失函数
        
        Args:
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits: torch.Tensor, 
                teacher_logits: torch.Tensor,
                true_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏损失
        
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            true_labels: 真实标签
            
        Returns:
            损失字典
        """
        # 计算软目标损失 (蒸馏损失)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 计算硬目标损失 (真实标签损失)
        hard_loss = self.ce_loss(student_logits, true_labels)
        
        # 总损失
        total_loss = self.alpha * distillation_loss + self.beta * hard_loss
        
        return {
            'total_loss': total_loss,
            'distillation_loss': distillation_loss,
            'hard_loss': hard_loss
        }


class ModelPruner:
    """模型剪枝器"""
    
    def __init__(self, pruning_ratio: float = 0.1):
        """
        初始化剪枝器
        
        Args:
            pruning_ratio: 剪枝比例
        """
        self.pruning_ratio = pruning_ratio
    
    def magnitude_pruning(self, model: nn.Module) -> None:
        """
        基于权重大小的剪枝
        
        Args:
            model: 要剪枝的模型
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # 计算权重的绝对值
                weight_abs = torch.abs(module.weight.data)
                
                # 计算阈值
                threshold = torch.quantile(weight_abs.flatten(), self.pruning_ratio)
                
                # 创建掩码
                mask = weight_abs > threshold
                
                # 应用剪枝
                module.weight.data *= mask.float()
                
                # 如果有偏置，也进行剪枝
                if module.bias is not None:
                    bias_abs = torch.abs(module.bias.data)
                    bias_threshold = torch.quantile(bias_abs.flatten(), self.pruning_ratio)
                    bias_mask = bias_abs > bias_threshold
                    module.bias.data *= bias_mask.float()
        
        print(f"✅ 完成基于权重大小的剪枝 (比例: {self.pruning_ratio})")
    
    def structured_pruning(self, model: nn.Module, layers_to_prune: List[str]) -> None:
        """
        结构化剪枝
        
        Args:
            model: 要剪枝的模型
            layers_to_prune: 要剪枝的层名称列表
        """
        for layer_name in layers_to_prune:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                if isinstance(layer, nn.Linear):
                    # 计算每个神经元的重要性
                    neuron_importance = torch.norm(layer.weight.data, dim=0)
                    
                    # 确定要保留的神经元数量
                    num_neurons = layer.weight.size(1)
                    num_keep = int(num_neurons * (1 - self.pruning_ratio))
                    
                    # 选择最重要的神经元
                    _, important_indices = torch.topk(neuron_importance, num_keep)
                    
                    # 创建新的权重和偏置
                    new_weight = layer.weight.data[:, important_indices]
                    new_bias = layer.bias.data if layer.bias is not None else None
                    
                    # 替换层
                    new_layer = nn.Linear(num_keep, layer.weight.size(0), bias=layer.bias is not None)
                    new_layer.weight.data = new_weight
                    if new_bias is not None:
                        new_layer.bias.data = new_bias
                    
                    setattr(model, layer_name, new_layer)
        
        print(f"✅ 完成结构化剪枝")
    
    def get_model_sparsity(self, model: nn.Module) -> float:
        """
        计算模型稀疏度
        
        Args:
            model: 模型
            
        Returns:
            稀疏度
        """
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()
        
        sparsity = zero_params / total_params
        return sparsity


class ModelQuantizer:
    """模型量化器"""
    
    def __init__(self, quantization_type: str = "dynamic"):
        """
        初始化量化器
        
        Args:
            quantization_type: 量化类型 ("dynamic", "static", "qat")
        """
        self.quantization_type = quantization_type
    
    def dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """
        动态量化
        
        Args:
            model: 要量化的模型
            
        Returns:
            量化后的模型
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        
        print(f"✅ 完成动态量化")
        return quantized_model
    
    def static_quantization(self, model: nn.Module, 
                          calibration_loader: DataLoader) -> nn.Module:
        """
        静态量化
        
        Args:
            model: 要量化的模型
            calibration_loader: 校准数据加载器
            
        Returns:
            量化后的模型
        """
        # 设置量化配置
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备量化
        quantized_model = torch.quantization.prepare(model)
        
        # 校准
        quantized_model.eval()
        with torch.no_grad():
            for batch in tqdm(calibration_loader, desc="量化校准"):
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids')
                    if input_ids is not None:
                        quantized_model(input_ids)
        
        # 转换为量化模型
        quantized_model = torch.quantization.convert(quantized_model)
        
        print(f"✅ 完成静态量化")
        return quantized_model
    
    def get_model_size(self, model: nn.Module) -> float:
        """
        计算模型大小 (MB)
        
        Args:
            model: 模型
            
        Returns:
            模型大小 (MB)
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb


class KnowledgeDistiller:
    """知识蒸馏器"""
    
    def __init__(self, config: DistillationConfig):
        """
        初始化蒸馏器
        
        Args:
            config: 蒸馏配置
        """
        self.config = config
        
        # 初始化模型和分词器
        self.teacher_model = None
        self.student_model = None
        self.tokenizer = None
        
        # 初始化组件
        self.loss_fn = TeacherStudentLoss(
            temperature=config.temperature,
            alpha=config.alpha
        )
        self.pruner = ModelPruner() if config.use_pruning else None
        self.quantizer = ModelQuantizer() if config.use_quantization else None
        
        # 设置输出目录
        self.output_dir = Path(config.save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎓 知识蒸馏器初始化完成")
        print(f"   教师模型: {config.teacher_model}")
        print(f"   学生模型: {config.student_model}")
        print(f"   温度: {config.temperature}")
        print(f"   输出目录: {self.output_dir}")
    
    def load_models(self) -> None:
        """加载教师和学生模型"""
        try:
            print("📥 加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.teacher_model,
                trust_remote_code=True,
                pad_token='<|extra_0|>',
                eos_token='<|im_end|>'
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("📥 加载教师模型...")
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.config.teacher_model,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            print("📥 加载学生模型...")
            if self.config.student_model == self.config.teacher_model:
                # 如果是同一个模型，创建副本
                student_config = AutoConfig.from_pretrained(self.config.student_model)
                # 可以修改配置来创建更小的模型
                # student_config.num_hidden_layers = student_config.num_hidden_layers // 2
                
                self.student_model = AutoModelForCausalLM.from_config(
                    student_config,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            else:
                self.student_model = AutoModelForCausalLM.from_pretrained(
                    self.config.student_model,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            # 设置为评估/训练模式
            self.teacher_model.eval()
            self.student_model.train()
            
            # 冻结教师模型
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            print(f"✅ 模型加载完成")
            print(f"   教师模型参数: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
            print(f"   学生模型参数: {sum(p.numel() for p in self.student_model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def prepare_dataset(self) -> Tuple[Dataset, Dataset]:
        """准备训练和验证数据集"""
        if USE_PROJECT_MODULES:
            try:
                # 从真实数据集加载
                dataloaders = create_all_dataloaders(
                    batch_sizes={'train': 32, 'val': 32}
                )
                
                # 提取训练数据
                train_texts, train_labels = [], []
                for batch in dataloaders['train']:
                    texts = batch.get('text', batch.get('caption', []))
                    labels = batch.get('labels', batch.get('label', []))
                    
                    if hasattr(labels, 'tolist'):
                        labels = labels.tolist()
                    
                    train_texts.extend(texts)
                    train_labels.extend(labels)
                
                # 提取验证数据
                val_texts, val_labels = [], []
                for batch in dataloaders['val']:
                    texts = batch.get('text', batch.get('caption', []))
                    labels = batch.get('labels', batch.get('label', []))
                    
                    if hasattr(labels, 'tolist'):
                        labels = labels.tolist()
                    
                    val_texts.extend(texts)
                    val_labels.extend(labels)
                
                # 限制数据量以加快训练
                train_texts = train_texts[:1000]
                train_labels = train_labels[:1000]
                val_texts = val_texts[:200]
                val_labels = val_labels[:200]
                
            except Exception as e:
                logger.warning(f"加载真实数据集失败: {e}，使用演示数据")
                train_texts, train_labels, val_texts, val_labels = self._get_demo_data()
        else:
            train_texts, train_labels, val_texts, val_labels = self._get_demo_data()
        
        # 创建数据集
        train_dataset = KnowledgeDistillationDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_length
        )
        val_dataset = KnowledgeDistillationDataset(
            val_texts, val_labels, self.tokenizer, self.config.max_length
        )
        
        print(f"✅ 数据集准备完成")
        print(f"   训练样本: {len(train_dataset)}")
        print(f"   验证样本: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def _get_demo_data(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        """获取演示数据"""
        train_texts = [
            "政府官方发布新的政策公告",
            "科学期刊发表的研究成果",
            "网传某地发生重大事故",
            "谣传疫苗含有害物质",
            "据不完全统计市场反响良好",
            "专家学者的权威观点",
            "未经证实的网络传言",
            "官方媒体的新闻报道"
        ] * 25  # 重复以增加数据量
        
        train_labels = [0, 0, 1, 1, 2, 0, 1, 0] * 25
        
        val_texts = [
            "教育部发布高考改革方案",
            "网传明天将发生地震",
            "业内人士透露的消息"
        ] * 20
        
        val_labels = [0, 1, 2] * 20
        
        return train_texts, train_labels, val_texts, val_labels
    
    def distill_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步蒸馏"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        true_labels = batch['labels']
        
        # 教师模型前向传播
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits[:, -1, :3]  # 取最后一个token的前3维作为分类logits
        
        # 学生模型前向传播
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        student_logits = student_outputs.logits[:, -1, :3]  # 取最后一个token的前3维作为分类logits
        
        # 计算损失
        loss_dict = self.loss_fn(student_logits, teacher_logits, true_labels)
        
        return {
            'total_loss': loss_dict['total_loss'].item(),
            'distillation_loss': loss_dict['distillation_loss'].item(),
            'hard_loss': loss_dict['hard_loss'].item()
        }
    
    def train_distillation(self) -> Dict[str, Any]:
        """训练蒸馏模型"""
        print("🎓 开始知识蒸馏训练...")
        
        # 准备数据集
        train_dataset, val_dataset = self.prepare_dataset()
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # 避免多进程问题
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # 优化器
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate
        )
        
        # 训练历史
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'epochs': []
        }
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            print(f"\n📚 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # 训练阶段
            self.student_model.train()
            train_losses = []
            
            train_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}")
            for batch in train_bar:
                # 移动到设备
                device = next(self.student_model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 前向传播
                loss_dict = self.distill_step(batch)
                total_loss = loss_dict['total_loss']
                
                # 反向传播
                optimizer.zero_grad()
                
                # 重新计算损失用于反向传播
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                true_labels = batch['labels']
                
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    teacher_logits = teacher_outputs.logits[:, -1, :3]
                
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                student_logits = student_outputs.logits[:, -1, :3]
                
                loss_dict_bp = self.loss_fn(student_logits, teacher_logits, true_labels)
                loss_dict_bp['total_loss'].backward()
                
                optimizer.step()
                
                train_losses.append(total_loss)
                train_bar.set_postfix({'loss': f'{total_loss:.4f}'})
            
            avg_train_loss = np.mean(train_losses)
            
            # 验证阶段
            self.student_model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="验证"):
                    device = next(self.student_model.parameters()).device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    loss_dict = self.distill_step(batch)
                    val_losses.append(loss_dict['total_loss'])
            
            avg_val_loss = np.mean(val_losses)
            
            # 记录历史
            training_history['train_losses'].append(avg_train_loss)
            training_history['val_losses'].append(avg_val_loss)
            training_history['epochs'].append(epoch + 1)
            
            print(f"   训练损失: {avg_train_loss:.4f}")
            print(f"   验证损失: {avg_val_loss:.4f}")
        
        print(f"✅ 知识蒸馏训练完成")
        
        return training_history
    
    def compress_model(self) -> Dict[str, Any]:
        """压缩模型"""
        print("🗜️  开始模型压缩...")
        
        compression_results = {}
        
        # 原始模型大小
        if self.quantizer:
            original_size = self.quantizer.get_model_size(self.student_model)
            compression_results['original_size_mb'] = original_size
            print(f"   原始模型大小: {original_size:.2f} MB")
        
        # 剪枝
        if self.config.use_pruning and self.pruner:
            print("✂️  执行模型剪枝...")
            
            # 计算原始稀疏度
            original_sparsity = self.pruner.get_model_sparsity(self.student_model)
            
            # 执行剪枝
            self.pruner.magnitude_pruning(self.student_model)
            
            # 计算剪枝后稀疏度
            pruned_sparsity = self.pruner.get_model_sparsity(self.student_model)
            
            compression_results.update({
                'original_sparsity': original_sparsity,
                'pruned_sparsity': pruned_sparsity,
                'pruning_ratio': pruned_sparsity - original_sparsity
            })
            
            print(f"   剪枝前稀疏度: {original_sparsity:.4f}")
            print(f"   剪枝后稀疏度: {pruned_sparsity:.4f}")
        
        # 量化
        if self.config.use_quantization and self.quantizer:
            print("📏 执行模型量化...")
            
            # 动态量化
            quantized_model = self.quantizer.dynamic_quantization(self.student_model)
            
            # 计算量化后大小
            quantized_size = self.quantizer.get_model_size(quantized_model)
            compression_ratio = original_size / quantized_size
            
            compression_results.update({
                'quantized_size_mb': quantized_size,
                'compression_ratio': compression_ratio
            })
            
            print(f"   量化后大小: {quantized_size:.2f} MB")
            print(f"   压缩比: {compression_ratio:.2f}x")
            
            # 更新学生模型
            self.student_model = quantized_model
        
        return compression_results
    
    def evaluate_student_model(self) -> Dict[str, Any]:
        """评估学生模型"""
        print("📊 评估学生模型性能...")
        
        if USE_PROJECT_MODULES:
            try:
                # 使用真实测试数据
                dataloaders = create_all_dataloaders(
                    batch_sizes={'test': 16}
                )
                
                test_texts, test_labels = [], []
                for batch in dataloaders['test']:
                    texts = batch.get('text', batch.get('caption', []))
                    labels = batch.get('labels', batch.get('label', []))
                    
                    if hasattr(labels, 'tolist'):
                        labels = labels.tolist()
                    
                    test_texts.extend(texts)
                    test_labels.extend(labels)
                
                # 限制测试数量
                test_texts = test_texts[:100]
                test_labels = test_labels[:100]
                
            except Exception as e:
                logger.warning(f"加载测试数据失败: {e}，使用演示数据")
                test_texts, test_labels = self._get_test_demo_data()
        else:
            test_texts, test_labels = self._get_test_demo_data()
        
        # 创建测试数据集
        test_dataset = KnowledgeDistillationDataset(
            test_texts, test_labels, self.tokenizer, self.config.max_length
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # 评估
        self.student_model.eval()
        predictions = []
        true_labels_list = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估"):
                device = next(self.student_model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                true_labels = batch['labels']
                
                # 学生模型预测
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits[:, -1, :3]
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().tolist())
                true_labels_list.extend(true_labels.cpu().tolist())
        
        # 计算评估指标
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        accuracy = accuracy_score(true_labels_list, predictions)
        f1_macro = f1_score(true_labels_list, predictions, average='macro')
        
        label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
        report = classification_report(
            true_labels_list, predictions,
            target_names=list(label_mapping.values()),
            output_dict=True
        )
        
        evaluation_result = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'classification_report': report,
            'num_samples': len(test_texts),
            'predictions': predictions,
            'true_labels': true_labels_list
        }
        
        print(f"✅ 学生模型评估完成:")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   F1分数: {f1_macro:.4f}")
        
        return evaluation_result
    
    def _get_test_demo_data(self) -> Tuple[List[str], List[int]]:
        """获取测试演示数据"""
        test_texts = [
            "权威机构发布的官方声明",
            "网络上流传的未证实消息",
            "专家学者的研究观点",
            "社交媒体上的传言",
            "政府部门的政策公告"
        ] * 10
        
        test_labels = [0, 1, 0, 1, 0] * 10
        
        return test_texts, test_labels
    
    def save_distilled_model(self, model_name: str = "distilled_student") -> None:
        """保存蒸馏后的模型"""
        save_path = self.output_dir / model_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存学生模型
        self.student_model.save_pretrained(save_path / "model")
        self.tokenizer.save_pretrained(save_path / "tokenizer")
        
        # 保存配置
        config_dict = {
            'teacher_model': self.config.teacher_model,
            'student_model': self.config.student_model,
            'temperature': self.config.temperature,
            'alpha': self.config.alpha,
            'beta': self.config.beta,
            'use_pruning': self.config.use_pruning,
            'use_quantization': self.config.use_quantization
        }
        
        with open(save_path / "distillation_config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 蒸馏模型已保存到: {save_path}")
    
    def run_full_distillation(self) -> Dict[str, Any]:
        """运行完整的蒸馏流程"""
        print("🚀 开始完整的知识蒸馏流程...")
        
        results = {}
        
        try:
            # 1. 加载模型
            self.load_models()
            
            # 2. 训练蒸馏
            training_history = self.train_distillation()
            results['training_history'] = training_history
            
            # 3. 模型压缩
            compression_results = self.compress_model()
            results['compression_results'] = compression_results
            
            # 4. 评估学生模型
            evaluation_results = self.evaluate_student_model()
            results['evaluation_results'] = evaluation_results
            
            # 5. 保存模型
            self.save_distilled_model()
            
            # 6. 汇总结果
            results['summary'] = {
                'final_accuracy': evaluation_results['accuracy'],
                'final_f1': evaluation_results['f1_macro'],
                'training_epochs': len(training_history['epochs']),
                'final_train_loss': training_history['train_losses'][-1],
                'final_val_loss': training_history['val_losses'][-1]
            }
            
            if compression_results:
                results['summary'].update({
                    'compression_achieved': True,
                    'original_size_mb': compression_results.get('original_size_mb', 0),
                    'final_size_mb': compression_results.get('quantized_size_mb', compression_results.get('original_size_mb', 0)),
                    'compression_ratio': compression_results.get('compression_ratio', 1.0)
                })
            
            print(f"\n✅ 完整蒸馏流程完成!")
            print(f"   最终准确率: {results['summary']['final_accuracy']:.4f}")
            print(f"   最终F1分数: {results['summary']['final_f1']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"蒸馏流程失败: {e}")
            results['error'] = str(e)
            return results


def create_distillation_system(use_compression: bool = True) -> KnowledgeDistiller:
    """
    创建知识蒸馏系统的便捷函数
    
    Args:
        use_compression: 是否使用压缩技术
        
    Returns:
        知识蒸馏器实例
    """
    print("🚀 创建知识蒸馏系统...")
    
    config = DistillationConfig(
        teacher_model="Qwen/Qwen3-0.6B",
        student_model="Qwen/Qwen3-0.6B",
        temperature=4.0,
        alpha=0.7,
        num_epochs=3,
        batch_size=4,  # 减小批次大小
        use_pruning=use_compression,
        use_quantization=use_compression
    )
    
    distiller = KnowledgeDistiller(config)
    
    return distiller


def demo_model_distillation():
    """演示模型蒸馏功能"""
    print("🎓 模型蒸馏和压缩演示")
    print("=" * 60)
    
    try:
        # 创建蒸馏系统
        distiller = create_distillation_system(use_compression=True)
        
        # 运行完整蒸馏流程
        results = distiller.run_full_distillation()
        
        # 显示结果
        if 'error' not in results:
            summary = results['summary']
            
            print(f"\n📊 蒸馏结果汇总:")
            print(f"   训练轮数: {summary['training_epochs']}")
            print(f"   最终准确率: {summary['final_accuracy']:.4f}")
            print(f"   最终F1分数: {summary['final_f1']:.4f}")
            print(f"   最终训练损失: {summary['final_train_loss']:.4f}")
            print(f"   最终验证损失: {summary['final_val_loss']:.4f}")
            
            if summary.get('compression_achieved'):
                print(f"\n🗜️  模型压缩结果:")
                print(f"   原始大小: {summary['original_size_mb']:.2f} MB")
                print(f"   压缩后大小: {summary['final_size_mb']:.2f} MB")
                print(f"   压缩比: {summary['compression_ratio']:.2f}x")
            
            # 保存结果
            results_file = distiller.output_dir / "distillation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 结果已保存到: {results_file}")
        else:
            print(f"❌ 蒸馏过程中出现错误: {results['error']}")
        
        print(f"\n✅ 模型蒸馏演示完成!")
        
    except Exception as e:
        print(f"❌ 蒸馏演示失败: {e}")
        raise


if __name__ == "__main__":
    demo_model_distillation()