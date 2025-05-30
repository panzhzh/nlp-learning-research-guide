#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# datasets/mr2_analysis.py

"""
MR2多模态谣言检测数据集深度分析
全面分析数据集结构、分布特征，生成可视化图表
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class MR2DatasetAnalyzer:
    """MR2数据集分析器"""
    
    def __init__(self, data_dir: str = '../data', output_dir: str = 'outputs'):
        """
        初始化分析器
        
        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'charts'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'analysis'), exist_ok=True)
        
        # 标签映射
        self.label_mapping = {
            0: 'Non-rumor',
            1: 'Rumor', 
            2: 'Unverified'
        }
        
        # 存储分析结果
        self.analysis_results = {}
        
    def load_data(self):
        """加载所有数据集"""
        print("🔄 开始加载MR2数据集...")
        
        self.datasets = {}
        splits = ['train', 'val', 'test']
        
        for split in splits:
            file_path = os.path.join(self.data_dir, f'dataset_items_{split}.json')
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.datasets[split] = json.load(f)
                print(f"✅ 加载 {split} 数据: {len(self.datasets[split])} 条")
            else:
                print(f"⚠️  未找到 {split} 数据文件: {file_path}")
                
        return self.datasets
    
    def basic_statistics(self):
        """基础统计分析"""
        print("\n📊 === 基础统计分析 ===")
        
        stats = {}
        total_samples = 0
        
        for split, data in self.datasets.items():
            split_stats = {
                'total_samples': len(data),
                'label_distribution': Counter(),
                'has_image': 0,
                'has_direct_annotation': 0,
                'has_inverse_annotation': 0
            }
            
            for item_id, item in data.items():
                # 标签分布
                label = item.get('label', -1)
                split_stats['label_distribution'][label] += 1
                
                # 图像文件检查
                if 'image_path' in item:
                    image_path = os.path.join(self.data_dir, item['image_path'])
                    if os.path.exists(image_path):
                        split_stats['has_image'] += 1
                
                # 检索标注检查
                if 'direct_path' in item:
                    direct_path = os.path.join(self.data_dir, item['direct_path'], 'direct_annotation.json')
                    if os.path.exists(direct_path):
                        split_stats['has_direct_annotation'] += 1
                        
                if 'inv_path' in item:
                    inv_path = os.path.join(self.data_dir, item['inv_path'], 'inverse_annotation.json')
                    if os.path.exists(inv_path):
                        split_stats['has_inverse_annotation'] += 1
            
            stats[split] = split_stats
            total_samples += split_stats['total_samples']
            
            print(f"\n{split.upper()} 数据集:")
            print(f"  样本总数: {split_stats['total_samples']}")
            print(f"  标签分布: {dict(split_stats['label_distribution'])}")
            print(f"  有图像: {split_stats['has_image']}")
            print(f"  有直接检索: {split_stats['has_direct_annotation']}")
            print(f"  有反向检索: {split_stats['has_inverse_annotation']}")
        
        print(f"\n总样本数: {total_samples}")
        self.analysis_results['basic_stats'] = stats
        return stats
    
    def text_analysis(self):
        """文本内容分析"""
        print("\n📝 === 文本内容分析 ===")
        
        text_stats = {
            'total_texts': 0,
            'length_distribution': [],
            'language_distribution': Counter(),
            'word_count_distribution': [],
            'character_distribution': Counter(),
            'common_words': Counter(),
            'samples_by_length': {'short': [], 'medium': [], 'long': []}
        }
        
        all_texts = []
        
        for split, data in self.datasets.items():
            for item_id, item in data.items():
                caption = item.get('caption', '').strip()
                if caption:
                    all_texts.append(caption)
                    text_stats['total_texts'] += 1
                    
                    # 文本长度
                    text_length = len(caption)
                    text_stats['length_distribution'].append(text_length)
                    
                    # 词数统计
                    words = caption.split()
                    word_count = len(words)
                    text_stats['word_count_distribution'].append(word_count)
                    
                    # 常用词统计
                    for word in words:
                        cleaned_word = re.sub(r'[^\w]', '', word.lower())
                        if len(cleaned_word) > 2:  # 过滤短词
                            text_stats['common_words'][cleaned_word] += 1
                    
                    # 字符分布
                    for char in caption.lower():
                        if char.isalpha():
                            text_stats['character_distribution'][char] += 1
                    
                    # 语言检测 (简单规则)
                    if re.search(r'[\u4e00-\u9fff]', caption):
                        if re.search(r'[a-zA-Z]', caption):
                            text_stats['language_distribution']['mixed'] += 1
                        else:
                            text_stats['language_distribution']['chinese'] += 1
                    else:
                        text_stats['language_distribution']['english'] += 1
                    
                    # 按长度分类样本
                    if text_length < 30:
                        text_stats['samples_by_length']['short'].append(caption)
                    elif text_length < 100:
                        text_stats['samples_by_length']['medium'].append(caption)
                    else:
                        text_stats['samples_by_length']['long'].append(caption)
        
        # 统计摘要
        if text_stats['length_distribution']:
            print(f"文本总数: {text_stats['total_texts']}")
            print(f"平均长度: {np.mean(text_stats['length_distribution']):.1f} 字符")
            print(f"平均词数: {np.mean(text_stats['word_count_distribution']):.1f} 词")
            print(f"语言分布: {dict(text_stats['language_distribution'])}")
            print(f"最常见词汇: {text_stats['common_words'].most_common(10)}")
            
            # 展示不同长度的样本
            print(f"\n短文本样例 (<30字符): {text_stats['samples_by_length']['short'][:3]}")
            print(f"中等文本样例 (30-100字符): {text_stats['samples_by_length']['medium'][:3]}")
            print(f"长文本样例 (>100字符): {text_stats['samples_by_length']['long'][:2]}")
        
        self.analysis_results['text_stats'] = text_stats
        return text_stats
    
    def image_analysis(self):
        """图像数据分析"""
        print("\n🖼️  === 图像数据分析 ===")
        
        image_stats = {
            'total_images': 0,
            'valid_images': 0,
            'image_sizes': [],
            'image_formats': Counter(),
            'size_distribution': {'width': [], 'height': []},
            'file_sizes': []
        }
        
        for split, data in self.datasets.items():
            for item_id, item in data.items():
                if 'image_path' in item:
                    image_stats['total_images'] += 1
                    image_path = os.path.join(self.data_dir, item['image_path'])
                    
                    if os.path.exists(image_path):
                        try:
                            # 尝试打开图像
                            with Image.open(image_path) as img:
                                image_stats['valid_images'] += 1
                                
                                # 图像尺寸
                                width, height = img.size
                                image_stats['size_distribution']['width'].append(width)
                                image_stats['size_distribution']['height'].append(height)
                                image_stats['image_sizes'].append((width, height))
                                
                                # 图像格式
                                image_stats['image_formats'][img.format] += 1
                                
                                # 文件大小
                                file_size = os.path.getsize(image_path)
                                image_stats['file_sizes'].append(file_size)
                                
                        except Exception as e:
                            print(f"⚠️  无法读取图像 {image_path}: {e}")
        
        if image_stats['valid_images'] > 0:
            print(f"图像总数: {image_stats['total_images']}")
            print(f"有效图像: {image_stats['valid_images']}")
            print(f"图像格式: {dict(image_stats['image_formats'])}")
            print(f"平均尺寸: {np.mean(image_stats['size_distribution']['width']):.0f} x {np.mean(image_stats['size_distribution']['height']):.0f}")
            print(f"平均文件大小: {np.mean(image_stats['file_sizes'])/1024:.1f} KB")
        
        self.analysis_results['image_stats'] = image_stats
        return image_stats
    
    def annotation_analysis(self):
        """检索标注数据分析"""
        print("\n🔍 === 检索标注分析 ===")
        
        annotation_stats = {
            'direct_annotations': 0,
            'inverse_annotations': 0,
            'direct_stats': {
                'images_with_captions': [],
                'images_no_captions': [],
                'domains': Counter(),
                'total_retrieved_images': []
            },
            'inverse_stats': {
                'entities_count': [],
                'entity_scores': [],
                'best_guess_labels': [],
                'fully_matched': [],
                'partially_matched': [],
                'common_entities': Counter()
            }
        }
        
        for split, data in self.datasets.items():
            for item_id, item in data.items():
                # 直接检索分析
                if 'direct_path' in item:
                    direct_file = os.path.join(self.data_dir, item['direct_path'], 'direct_annotation.json')
                    if os.path.exists(direct_file):
                        try:
                            with open(direct_file, 'r', encoding='utf-8') as f:
                                direct_data = json.load(f)
                                annotation_stats['direct_annotations'] += 1
                                
                                # 统计检索结果
                                if 'images_with_captions' in direct_data:
                                    count = len(direct_data['images_with_captions'])
                                    annotation_stats['direct_stats']['images_with_captions'].append(count)
                                    
                                    # 域名统计
                                    for img_info in direct_data['images_with_captions']:
                                        domain = img_info.get('domain', 'unknown')
                                        annotation_stats['direct_stats']['domains'][domain] += 1
                                
                                if 'images_with_no_captions' in direct_data:
                                    count = len(direct_data['images_with_no_captions'])
                                    annotation_stats['direct_stats']['images_no_captions'].append(count)
                                
                                # 总检索图像数
                                total = len(direct_data.get('images_with_captions', [])) + len(direct_data.get('images_with_no_captions', []))
                                annotation_stats['direct_stats']['total_retrieved_images'].append(total)
                                
                        except Exception as e:
                            print(f"⚠️  读取直接检索文件失败 {direct_file}: {e}")
                
                # 反向检索分析
                if 'inv_path' in item:
                    inverse_file = os.path.join(self.data_dir, item['inv_path'], 'inverse_annotation.json')
                    if os.path.exists(inverse_file):
                        try:
                            with open(inverse_file, 'r', encoding='utf-8') as f:
                                inverse_data = json.load(f)
                                annotation_stats['inverse_annotations'] += 1
                                
                                # 实体分析
                                entities = inverse_data.get('entities', [])
                                annotation_stats['inverse_stats']['entities_count'].append(len(entities))
                                
                                for entity in entities:
                                    annotation_stats['inverse_stats']['common_entities'][entity] += 1
                                
                                # 实体分数
                                scores = inverse_data.get('entities_scores', [])
                                annotation_stats['inverse_stats']['entity_scores'].extend(scores)
                                
                                # 最佳猜测标签
                                best_guess = inverse_data.get('best_guess_lbl', [])
                                annotation_stats['inverse_stats']['best_guess_labels'].extend(best_guess)
                                
                                # 匹配结果统计
                                fully_matched = len(inverse_data.get('all_fully_matched_captions', []))
                                partially_matched = len(inverse_data.get('all_partially_matched_captions', []))
                                annotation_stats['inverse_stats']['fully_matched'].append(fully_matched)
                                annotation_stats['inverse_stats']['partially_matched'].append(partially_matched)
                                
                        except Exception as e:
                            print(f"⚠️  读取反向检索文件失败 {inverse_file}: {e}")
        
        # 输出统计结果
        print(f"直接检索标注数: {annotation_stats['direct_annotations']}")
        print(f"反向检索标注数: {annotation_stats['inverse_annotations']}")
        
        if annotation_stats['direct_stats']['total_retrieved_images']:
            print(f"平均检索图像数: {np.mean(annotation_stats['direct_stats']['total_retrieved_images']):.1f}")
            print(f"热门域名: {annotation_stats['direct_stats']['domains'].most_common(5)}")
        
        if annotation_stats['inverse_stats']['entities_count']:
            print(f"平均实体数: {np.mean(annotation_stats['inverse_stats']['entities_count']):.1f}")
            print(f"常见实体: {annotation_stats['inverse_stats']['common_entities'].most_common(10)}")
        
        self.analysis_results['annotation_stats'] = annotation_stats
        return annotation_stats
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("\n📊 === 生成可视化图表 ===")
        
        # 设置图表参数
        plt.style.use('default')
        fig_size = (15, 12)
        
        # 1. 数据集基础分布图
        self._plot_basic_distribution()
        
        # 2. 文本长度分布图
        self._plot_text_distribution()
        
        # 3. 图像尺寸分布图
        self._plot_image_distribution()
        
        # 4. 检索结果分析图
        self._plot_annotation_analysis()
        
        # 5. 综合分析仪表板
        self._create_dashboard()
        
        print("✅ 所有图表已生成完成")
    
    def _plot_basic_distribution(self):
        """绘制基础数据分布"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MR2数据集基础分布分析', fontsize=16, fontweight='bold')
        
        # 1. 数据集大小分布
        splits = list(self.analysis_results['basic_stats'].keys())
        sizes = [self.analysis_results['basic_stats'][split]['total_samples'] for split in splits]
        
        axes[0, 0].bar(splits, sizes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('各数据集大小分布')
        axes[0, 0].set_ylabel('样本数量')
        for i, v in enumerate(sizes):
            axes[0, 0].text(i, v + max(sizes)*0.01, str(v), ha='center', va='bottom')
        
        # 2. 标签分布 (合并所有split)
        all_labels = Counter()
        for split_stats in self.analysis_results['basic_stats'].values():
            all_labels.update(split_stats['label_distribution'])
        
        labels = [self.label_mapping.get(k, f'Unknown({k})') for k in all_labels.keys()]
        counts = list(all_labels.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        axes[0, 1].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
        axes[0, 1].set_title('标签分布')
        
        # 3. 数据完整性分析
        completeness_data = []
        for split, stats in self.analysis_results['basic_stats'].items():
            total = stats['total_samples']
            completeness_data.append({
                'Split': split,
                'Has Image': stats['has_image'] / total * 100,
                'Has Direct': stats['has_direct_annotation'] / total * 100,
                'Has Inverse': stats['has_inverse_annotation'] / total * 100
            })
        
        df_completeness = pd.DataFrame(completeness_data)
        x = np.arange(len(df_completeness))
        width = 0.25
        
        axes[1, 0].bar(x - width, df_completeness['Has Image'], width, label='有图像', color='#FF6B6B')
        axes[1, 0].bar(x, df_completeness['Has Direct'], width, label='有直接检索', color='#4ECDC4')
        axes[1, 0].bar(x + width, df_completeness['Has Inverse'], width, label='有反向检索', color='#45B7D1')
        
        axes[1, 0].set_ylabel('完整性 (%)')
        axes[1, 0].set_title('数据完整性分析')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(df_completeness['Split'])
        axes[1, 0].legend()
        
        # 4. 按split的标签分布
        split_label_data = []
        for split, stats in self.analysis_results['basic_stats'].items():
            for label, count in stats['label_distribution'].items():
                split_label_data.append({
                    'Split': split,
                    'Label': self.label_mapping.get(label, f'Unknown({label})'),
                    'Count': count
                })
        
        df_split_labels = pd.DataFrame(split_label_data)
        pivot_df = df_split_labels.pivot(index='Split', columns='Label', values='Count').fillna(0)
        
        pivot_df.plot(kind='bar', ax=axes[1, 1], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 1].set_title('各数据集标签分布')
        axes[1, 1].set_ylabel('样本数量')
        axes[1, 1].legend(title='标签')
        axes[1, 1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'charts', 'basic_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_text_distribution(self):
        """绘制文本分布分析"""
        if 'text_stats' not in self.analysis_results:
            return
            
        text_stats = self.analysis_results['text_stats']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('文本内容分析', fontsize=16, fontweight='bold')
        
        # 1. 文本长度分布
        axes[0, 0].hist(text_stats['length_distribution'], bins=30, color='#FF6B6B', alpha=0.7)
        axes[0, 0].set_title('文本长度分布')
        axes[0, 0].set_xlabel('字符数')
        axes[0, 0].set_ylabel('频次')
        
        # 2. 词数分布
        axes[0, 1].hist(text_stats['word_count_distribution'], bins=20, color='#4ECDC4', alpha=0.7)
        axes[0, 1].set_title('词数分布')
        axes[0, 1].set_xlabel('词数')
        axes[0, 1].set_ylabel('频次')
        
        # 3. 语言分布
        lang_data = text_stats['language_distribution']
        axes[0, 2].pie(lang_data.values(), labels=lang_data.keys(), autopct='%1.1f%%',
                      colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 2].set_title('语言分布')
        
        # 4. 常用词云形式的柱状图
        common_words = text_stats['common_words'].most_common(15)
        words, counts = zip(*common_words)
        
        axes[1, 0].barh(range(len(words)), counts, color='#96CEB4')
        axes[1, 0].set_yticks(range(len(words)))
        axes[1, 0].set_yticklabels(words)
        axes[1, 0].set_title('最常见词汇 (Top 15)')
        axes[1, 0].set_xlabel('出现次数')
        
        # 5. 字符频率分布
        char_freq = text_stats['character_distribution'].most_common(20)
        chars, freqs = zip(*char_freq)
        
        axes[1, 1].bar(range(len(chars)), freqs, color='#FFEAA7')
        axes[1, 1].set_xticks(range(len(chars)))
        axes[1, 1].set_xticklabels(chars)
        axes[1, 1].set_title('字符频率分布 (Top 20)')
        axes[1, 1].set_ylabel('出现次数')
        
        # 6. 文本长度统计摘要
        length_stats = {
            '平均长度': np.mean(text_stats['length_distribution']),
            '中位数': np.median(text_stats['length_distribution']),
            '最大长度': np.max(text_stats['length_distribution']),
            '最小长度': np.min(text_stats['length_distribution']),
            '标准差': np.std(text_stats['length_distribution'])
        }
        
        axes[1, 2].axis('off')
        stats_text = '\n'.join([f'{k}: {v:.1f}' for k, v in length_stats.items()])
        axes[1, 2].text(0.1, 0.5, f'文本长度统计:\n\n{stats_text}', 
                       transform=axes[1, 2].transAxes, fontsize=12,
                       verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'charts', 'text_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_image_distribution(self):
        """绘制图像分布分析"""
        if 'image_stats' not in self.analysis_results:
            return
            
        image_stats = self.analysis_results['image_stats']
        
        if image_stats['valid_images'] == 0:
            print("⚠️  没有有效图像数据，跳过图像分析图表")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('图像数据分析', fontsize=16, fontweight='bold')
        
        # 1. 图像尺寸散点图
        widths = image_stats['size_distribution']['width']
        heights = image_stats['size_distribution']['height']
        
        axes[0, 0].scatter(widths, heights, alpha=0.6, color='#FF6B6B')
        axes[0, 0].set_xlabel('宽度 (像素)')
        axes[0, 0].set_ylabel('高度 (像素)')
        axes[0, 0].set_title('图像尺寸分布')
        
        # 2. 宽度分布直方图
        axes[0, 1].hist(widths, bins=20, color='#4ECDC4', alpha=0.7)
        axes[0, 1].set_xlabel('宽度 (像素)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('图像宽度分布')
        
        # 3. 高度分布直方图
        axes[1, 0].hist(heights, bins=20, color='#45B7D1', alpha=0.7)
        axes[1, 0].set_xlabel('高度 (像素)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('图像高度分布')
        
        # 4. 图像格式分布
        format_data = image_stats['image_formats']
        axes[1, 1].pie(format_data.values(), labels=format_data.keys(), autopct='%1.1f%%',
                      colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1, 1].set_title('图像格式分布')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'charts', 'image_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_annotation_analysis(self):
        """绘制检索标注分析"""
        if 'annotation_stats' not in self.analysis_results:
            return
            
        annotation_stats = self.analysis_results['annotation_stats']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('检索标注数据分析', fontsize=16, fontweight='bold')
        
        # 1. 检索数量对比
        annotation_counts = [
            annotation_stats['direct_annotations'],
            annotation_stats['inverse_annotations']
        ]
        
        axes[0, 0].bar(['直接检索', '反向检索'], annotation_counts, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 0].set_title('检索标注数量')
        axes[0, 0].set_ylabel('标注数量')
        for i, v in enumerate(annotation_counts):
            axes[0, 0].text(i, v + max(annotation_counts)*0.01, str(v), ha='center', va='bottom')
        
        # 2. 直接检索图像数量分布
        if annotation_stats['direct_stats']['total_retrieved_images']:
            axes[0, 1].hist(annotation_stats['direct_stats']['total_retrieved_images'], 
                           bins=15, color='#45B7D1', alpha=0.7)
            axes[0, 1].set_title('直接检索图像数量分布')