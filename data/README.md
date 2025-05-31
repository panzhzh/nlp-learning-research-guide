# 多模态谣言检测数据集说明

## 📊 数据集概览

本数据集为多模态谣言检测数据集MR2的缩减版，包含中英文混合的文本声明、相关图像以及通过多种检索方式获得的验证信息。数据集专门用于训练和评估谣言检测模型，注意该缩减数据集不能直接论文中使用。

缩减版数据集下载地址：
链接: https://pan.baidu.com/s/1sfUwsaeV2nfl54OkrfrKVw?pwd=jxhc 
提取码: jxhc 

## 📁 目录结构

```
data/
├── dataset_items_train.json           # 训练集数据项
├── dataset_items_val.json             # 验证集数据项
├── dataset_items_test.json            # 测试集数据项
├── train/                             # 训练集相关文件
│   ├── img/                          # 声明相关图像
│   ├── img_html_news/                # 基于标题检索的网页和图像
│   │   └── direct_annotation.json    # 直接检索标注信息
│   └── inverse_search/               # 基于图像反向检索的网页
│       └── inverse_annotation.json   # 反向检索标注信息
├── val/                              # 验证集相关文件
│   ├── img/
│   ├── img_html_news/
│   │   └── direct_annotation.json
│   └── inverse_search/
│       └── inverse_annotation.json
└── test/                             # 测试集相关文件
    ├── img/
    ├── img_html_news/
    │   └── direct_annotation.json
    └── inverse_search/
        └── inverse_annotation.json
```

## 🗃️ 数据格式说明

### 主数据文件 (dataset_items_*.json)

每个数据项包含以下字段：

```json
{
  "id": "claim_unique_id",
  "text": "声明文本内容（中英文混合）",
  "label": 1,
  "image_path": "img/claim_id.jpg",
  "language": "mixed/chinese/english",
  "timestamp": "2023-01-01T00:00:00Z",
  "source": "数据来源平台"
}
```

### 标签定义
| 标签值 | 含义 | 描述 |
|--------|------|------|
| **0** | Non-rumor | 非谣言，经过验证的真实信息 |
| **1** | Rumor | 谣言，经过验证的虚假信息 |
| **2** | Unverified | 未验证，无法确定真假的信息 |

### 直接检索标注 (direct_annotation.json)

基于声明标题检索相关网页和图像的结果：

```json
{
  "claim_id": {
    "img_link": "检索到的相关图像链接",
    "page_link": "检索到的网页链接",
    "domain": "网页域名",
    "snippet": "网页内容摘要",
    "image_path": "检索图像的本地路径",
    "html_path": "网页HTML文件的本地路径",
    "page_title": "网页标题"
  }
}
```

### 反向检索标注 (inverse_annotation.json)

基于声明中的图像进行反向搜索的结果：

```json
{
  "claim_id": {
    "entities": ["实体1", "实体2"],
    "entities_scores": [0.95, 0.87],
    "best_guess_lbl": "图像最可能的描述",
    "all_fully_matched_captions": [
      {
        "page_link": "完全匹配的网页链接",
        "image_link": "完全匹配的图像链接", 
        "html_path": "网页HTML本地路径",
        "title": "网页标题"
      }
    ],
    "all_partially_matched_captions": [
      {
        "page_link": "部分匹配的网页链接",
        "image_link": "部分匹配的图像链接",
        "html_path": "网页HTML本地路径", 
        "title": "网页标题"
      }
    ],
    "fully_matched_no_text": [
      {
        "page_link": "无文本完全匹配的网页链接",
        "image_link": "无文本完全匹配的图像链接",
        "html_path": "网页HTML本地路径",
        "title": "网页标题"
      }
    ]
  }
}
```

## 🔍 检索信息说明

### 直接检索 (Direct Search)
- **检索方式**: 基于声明的文本内容搜索相关网页和图像
- **用途**: 提供外部验证信息和上下文
- **数据来源**: 搜索引擎结果

### 反向检索 (Inverse Search)  
- **检索方式**: 基于声明中的图像进行反向图像搜索
- **用途**: 验证图像的原始来源和使用历史
- **实体识别**: 使用计算机视觉技术识别图像中的实体
- **匹配类型**:
  - **完全匹配**: 找到完全相同的图像和描述
  - **部分匹配**: 找到相似的图像或相关描述
  - **无文本匹配**: 找到相同图像但无相关文本描述

## 🎯 应用场景

### 1. 谣言检测任务
- **输入**: 文本声明 + 相关图像
- **输出**: 谣言/非谣言/未验证分类
- **评估指标**: 准确率、精确率、召回率、F1分数

### 2. 多模态验证
- **利用检索信息**: 结合外部检索结果进行验证
- **跨模态对齐**: 文本和图像内容一致性检查
- **时序分析**: 利用时间戳信息分析传播模式

### 3. 实体识别与事实核查
- **实体提取**: 利用图像实体识别结果
- **事实验证**: 使用检索到的权威来源进行核查
- **来源追踪**: 追溯图像和声明的原始来源

## 📋 数据使用协议

- ✅ **学术研究**: 允许用于学术研究和论文发表
- ✅ **开源项目**: 允许用于开源算法和模型开发
- ❌ **商业用途**: 禁止直接商业化使用
- ❌ **数据再分发**: 禁止重新分发原始数据集
- ⚠️ **隐私保护**: 使用时需遵守相关隐私保护规定

### 数据验证
- **格式检查**: 自动验证JSON格式和字段完整性
- **图像验证**: 检查图像文件完整性和可读性
- **链接验证**: 定期检查外部链接有效性

## 📚 致谢

感谢作者Hu Xuming的开源数据集，如果使用完整数据集，请引用以下论文：

```bibtex
@inproceedings{hu2023mr2,
  title={Mr2: A benchmark for multimodal retrieval-augmented rumor detection in social media},
  author={Hu, Xuming and Guo, Zhijiang and Chen, Junzhe and Wen, Lijie and Yu, Philip S},
  booktitle={Proceedings of the 46th international ACM SIGIR conference on research and development in information retrieval},
  pages={2901--2912},
  year={2023}
}
```

## 🔗 相关资源

- **数据集项目主页**: https://github.com/THU-BPM/MR2
```

这个README详细描述了数据集的结构、格式和使用方法，特别突出了谣言检测的应用场景和多模态检索信息的价值。本项目使用该数据集进行各种测试。