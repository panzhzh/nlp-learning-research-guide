# 训练配置 Training Configs

> 🏋️ **模型训练流程、优化器、学习率等训练策略配置**

## 📋 功能说明

`training_configs.yaml` 管理模型训练过程中的所有超参数，包括优化策略、学习率调度、早停机制等。

## 🎯 主要配置块

### general - 通用训练配置
- 任务类型和类别数设置
- 数据批次大小配置
- 训练轮数和学习率
- 早停和学习率调度

### traditional_ml - 传统机器学习训练
- 交叉验证配置
- 超参数搜索策略
- 网格搜索参数空间
- 评估指标设置

### neural_networks - 神经网络训练
- 优化器配置 (Adam, SGD)
- 损失函数选择
- 正则化参数
- 批归一化设置

### pretrained_finetuning - 预训练模型微调
- **BERT微调**: 学习率、warmup、层级学习率
- **多语言模型**: 语言适配和adapter配置
- **Dropout设置**: 各层dropout概率

### multimodal - 多模态训练
- **CLIP训练**: 对比学习参数
- **模态权重**: 文本/图像学习率倍数
- **融合策略**: early/late/hybrid融合

### graph_neural_networks - 图神经网络训练
- 图采样策略配置
- 邻居采样数量
- 图数据增强方法
- 注意力机制参数

### large_language_models - 大语言模型训练
- 生成参数配置
- 内存优化策略
- 梯度累积设置
- 混合精度训练

### parameter_efficient_finetuning - 参数高效微调
- **LoRA配置**: rank, alpha, target_modules
- **AdaLoRA**: 动态rank调整
- **Prefix Tuning**: 虚拟token数量

### distributed_training - 分布式训练
- 数据并行配置
- 模型并行设置
- DeepSpeed优化
- 多GPU策略

### experiment_tracking - 实验跟踪
- **WandB配置**: 项目名、标签设置
- **TensorBoard**: 日志目录配置
- **MLflow**: 实验管理设置

## 💡 使用场景

- 设置训练超参数
- 配置优化器和学习率
- 选择损失函数和评估指标
- 设置早停和模型保存策略
- 配置分布式训练参数

---

**[⬅️ 模型配置](model_configs.md) | [支持模型列表 ➡️](supported_models.md)**
