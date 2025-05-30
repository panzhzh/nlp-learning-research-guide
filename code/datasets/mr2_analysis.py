#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# code/datasets/mr2_analysis.py

"""
MR2多模态谣言检测数据集深度分析
全面分析数据集结构、分布特征，生成可视化图表
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体避免图表乱码，其他输出保持中文
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 添加配置管理器路径
current_file = Path(__file__).resolve()
code_root = current_file.parent.parent
sys.path.append(str(code_root))

try:
    from utils.config_manager import get_config_manager, get_output_path, get_analysis_config, get_label_mapping, get_data_dir
    USE_CONFIG_MANAGER = True
except ImportError:
    print("⚠️  配置管理器不可用，使用默认配置")
    USE_CONFIG_MANAGER = False


class MR2DatasetAnalyzer:
    """MR2数据集分析器"""
    
    def __init__(self, data_dir: str = 'data', output_dir: str = 'outputs'):
        """
        初始化分析器
        
        Args:
            data_dir: 数据目录路径 (相对于code目录)
            output_dir: 输出目录路径 (相对于code目录)
        """
        if USE_CONFIG_MANAGER:
            # 使用配置管理器
            self.config_manager = get_config_manager()
            self.config_manager.create_output_directories()
            
            self.data_dir = get_data_dir()
            self.charts_dir = get_output_path('datasets', 'charts')
            self.reports_dir = get_output_path('datasets', 'reports')
            self.analysis_dir = get_output_path('datasets', 'analysis')
            
            analysis_config = get_analysis_config()
            viz_config = analysis_config.get('visualization', {})
            self.colors = viz_config.get('colors', self._default_colors())
            self.label_mapping = get_label_mapping()
            
            print(f"🔧 使用配置管理器")
            print(f"🔧 数据目录: {self.data_dir}")
            print(f"🔧 输出目录: {self.charts_dir.parent}")
            
        else:
            # 使用默认配置
            self.data_dir = data_dir
            self.output_dir = output_dir
            self.charts_dir = Path(output_dir) / 'charts'
            self.reports_dir = Path(output_dir) / 'reports'
            self.analysis_dir = Path(output_dir) / 'analysis'
            
            # 创建输出目录
            for dir_path in [self.charts_dir, self.reports_dir, self.analysis_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            self.colors = self._default_colors()
            self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
            
            print(f"🔧 使用默认配置")
            print(f"🔧 数据目录: {self.data_dir}")
            print(f"🔧 输出目录: {self.output_dir}")
        
        # 存储分析结果
        self.analysis_results = {}
    
    def _default_colors(self):
        """默认颜色配置"""
        return {
            'primary': '#FF6B6B',
            'secondary': '#4ECDC4', 
            'tertiary': '#45B7D1',
            'accent': '#96CEB4',
            'warning': '#FFEAA7',
            'info': '#DDA0DD',
            'success': '#98FB98'
        }
        
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
            print(f"\n短文本样例 (<30字符):")
            for i, text in enumerate(text_stats['samples_by_length']['short'][:5]):
                print(f"  {i+1}. {text}")
            
            print(f"\n中等文本样例 (30-100字符):")
            for i, text in enumerate(text_stats['samples_by_length']['medium'][:5]):
                print(f"  {i+1}. {text}")
                
            print(f"\n长文本样例 (>100字符):")
            for i, text in enumerate(text_stats['samples_by_length']['long'][:3]):
                print(f"  {i+1}. {text}")
        
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
        fig.suptitle('MR2 Dataset Basic Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. 数据集大小分布
        splits = list(self.analysis_results['basic_stats'].keys())
        sizes = [self.analysis_results['basic_stats'][split]['total_samples'] for split in splits]
        
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
        axes[0, 0].bar(splits, sizes, color=colors)
        axes[0, 0].set_title('Dataset Size Distribution')
        axes[0, 0].set_ylabel('Number of Samples')
        for i, v in enumerate(sizes):
            axes[0, 0].text(i, v + max(sizes)*0.01, str(v), ha='center', va='bottom')
        
        # 2. 标签分布 (合并所有split)
        all_labels = Counter()
        for split_stats in self.analysis_results['basic_stats'].values():
            all_labels.update(split_stats['label_distribution'])
        
        labels = [self.label_mapping.get(k, f'Unknown({k})') for k in all_labels.keys()]
        counts = list(all_labels.values())
        
        axes[0, 1].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
        axes[0, 1].set_title('Label Distribution')
        
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
        
        axes[1, 0].bar(x - width, df_completeness['Has Image'], width, label='Has Image', color=self.colors['primary'])
        axes[1, 0].bar(x, df_completeness['Has Direct'], width, label='Has Direct Search', color=self.colors['secondary'])
        axes[1, 0].bar(x + width, df_completeness['Has Inverse'], width, label='Has Inverse Search', color=self.colors['tertiary'])
        
        axes[1, 0].set_ylabel('Completeness (%)')
        axes[1, 0].set_title('Data Completeness Analysis')
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
        
        pivot_df.plot(kind='bar', ax=axes[1, 1], color=colors)
        axes[1, 1].set_title('Label Distribution by Split')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].legend(title='Labels')
        axes[1, 1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'basic_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_text_distribution(self):
        """绘制文本分布分析"""
        if 'text_stats' not in self.analysis_results:
            return
            
        text_stats = self.analysis_results['text_stats']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Text Content Analysis', fontsize=16, fontweight='bold')
        
        # 1. 文本长度分布
        axes[0, 0].hist(text_stats['length_distribution'], bins=30, color=self.colors['primary'], alpha=0.7)
        axes[0, 0].set_title('Text Length Distribution')
        axes[0, 0].set_xlabel('Number of Characters')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. 词数分布
        axes[0, 1].hist(text_stats['word_count_distribution'], bins=20, color=self.colors['secondary'], alpha=0.7)
        axes[0, 1].set_title('Word Count Distribution')
        axes[0, 1].set_xlabel('Number of Words')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. 语言分布
        lang_data = text_stats['language_distribution']
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
        axes[0, 2].pie(lang_data.values(), labels=lang_data.keys(), autopct='%1.1f%%', colors=colors)
        axes[0, 2].set_title('Language Distribution')
        
        # 4. 常用词云形式的柱状图
        common_words = text_stats['common_words'].most_common(15)
        if common_words:
            words, counts = zip(*common_words)
            
            axes[1, 0].barh(range(len(words)), counts, color=self.colors['accent'])
            axes[1, 0].set_yticks(range(len(words)))
            axes[1, 0].set_yticklabels(words)
            axes[1, 0].set_title('Most Common Words (Top 15)')
            axes[1, 0].set_xlabel('Frequency')
        
        # 5. 字符频率分布
        char_freq = text_stats['character_distribution'].most_common(20)
        if char_freq:
            chars, freqs = zip(*char_freq)
            
            axes[1, 1].bar(range(len(chars)), freqs, color=self.colors['warning'])
            axes[1, 1].set_xticks(range(len(chars)))
            axes[1, 1].set_xticklabels(chars)
            axes[1, 1].set_title('Character Frequency Distribution (Top 20)')
            axes[1, 1].set_ylabel('Frequency')
        
        # 6. 文本长度统计摘要
        if text_stats['length_distribution']:
            length_stats = {
                'Average Length': np.mean(text_stats['length_distribution']),
                'Median': np.median(text_stats['length_distribution']),
                'Max Length': np.max(text_stats['length_distribution']),
                'Min Length': np.min(text_stats['length_distribution']),
                'Std Dev': np.std(text_stats['length_distribution'])
            }
            
            axes[1, 2].axis('off')
            stats_text = '\n'.join([f'{k}: {v:.1f}' for k, v in length_stats.items()])
            axes[1, 2].text(0.1, 0.5, f'Text Length Statistics:\n\n{stats_text}', 
                           transform=axes[1, 2].transAxes, fontsize=12,
                           verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'text_distribution.png', dpi=300, bbox_inches='tight')
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
        fig.suptitle('Image Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. 图像尺寸散点图
        widths = image_stats['size_distribution']['width']
        heights = image_stats['size_distribution']['height']
        
        axes[0, 0].scatter(widths, heights, alpha=0.6, color=self.colors['primary'])
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Height (pixels)')
        axes[0, 0].set_title('Image Size Distribution')
        
        # 2. 宽度分布直方图
        axes[0, 1].hist(widths, bins=20, color=self.colors['secondary'], alpha=0.7)
        axes[0, 1].set_xlabel('Width (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Image Width Distribution')
        
        # 3. 高度分布直方图
        axes[1, 0].hist(heights, bins=20, color=self.colors['tertiary'], alpha=0.7)
        axes[1, 0].set_xlabel('Height (pixels)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Image Height Distribution')
        
        # 4. 图像格式分布
        format_data = image_stats['image_formats']
        if format_data:
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary'], self.colors['accent']]
            axes[1, 1].pie(format_data.values(), labels=format_data.keys(), autopct='%1.1f%%', colors=colors)
            axes[1, 1].set_title('Image Format Distribution')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'image_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_annotation_analysis(self):
        """绘制检索标注分析"""
        if 'annotation_stats' not in self.analysis_results:
            return
            
        annotation_stats = self.analysis_results['annotation_stats']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Retrieval Annotation Analysis', fontsize=16, fontweight='bold')
        
        # 1. 检索数量对比
        annotation_counts = [
            annotation_stats['direct_annotations'],
            annotation_stats['inverse_annotations']
        ]
        
        if max(annotation_counts) > 0:
            axes[0, 0].bar(['Direct Search', 'Inverse Search'], annotation_counts, 
                          color=[self.colors['primary'], self.colors['secondary']])
            axes[0, 0].set_title('Annotation Count')
            axes[0, 0].set_ylabel('Number of Annotations')
            for i, v in enumerate(annotation_counts):
                axes[0, 0].text(i, v + max(annotation_counts)*0.01, str(v), ha='center', va='bottom')
        
        # 2. 直接检索图像数量分布
        if annotation_stats['direct_stats']['total_retrieved_images']:
            axes[0, 1].hist(annotation_stats['direct_stats']['total_retrieved_images'], 
                           bins=15, color=self.colors['tertiary'], alpha=0.7)
            axes[0, 1].set_title('Direct Search Image Count Distribution')
            axes[0, 1].set_xlabel('Number of Retrieved Images')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. 热门域名分布
        if annotation_stats['direct_stats']['domains']:
            top_domains = annotation_stats['direct_stats']['domains'].most_common(10)
            domains, counts = zip(*top_domains)
            
            axes[0, 2].barh(range(len(domains)), counts, color=self.colors['accent'])
            axes[0, 2].set_yticks(range(len(domains)))
            axes[0, 2].set_yticklabels(domains)
            axes[0, 2].set_title('Top Retrieval Domains (Top 10)')
            axes[0, 2].set_xlabel('Frequency')
        
        # 4. 反向检索实体数量分布
        if annotation_stats['inverse_stats']['entities_count']:
            axes[1, 0].hist(annotation_stats['inverse_stats']['entities_count'],
                           bins=15, color=self.colors['warning'], alpha=0.7)
            axes[1, 0].set_title('Inverse Search Entity Count Distribution')
            axes[1, 0].set_xlabel('Number of Entities')
            axes[1, 0].set_ylabel('Frequency')
        
        # 5. 实体置信度分布
        if annotation_stats['inverse_stats']['entity_scores']:
            axes[1, 1].hist(annotation_stats['inverse_stats']['entity_scores'],
                           bins=20, color=self.colors['info'], alpha=0.7)
            axes[1, 1].set_title('Entity Confidence Score Distribution')
            axes[1, 1].set_xlabel('Confidence Score')
            axes[1, 1].set_ylabel('Frequency')
        
        # 6. 常见实体词云
        if annotation_stats['inverse_stats']['common_entities']:
            top_entities = annotation_stats['inverse_stats']['common_entities'].most_common(15)
            entities, counts = zip(*top_entities)
            
            axes[1, 2].barh(range(len(entities)), counts, color=self.colors['success'])
            axes[1, 2].set_yticks(range(len(entities)))
            axes[1, 2].set_yticklabels(entities)
            axes[1, 2].set_title('Common Detected Entities (Top 15)')
            axes[1, 2].set_xlabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'annotation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_dashboard(self):
        """创建综合分析仪表板"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('MR2 Dataset Comprehensive Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 数据集概览
        ax1 = fig.add_subplot(gs[0, :2])
        stats_summary = []
        for split, stats in self.analysis_results['basic_stats'].items():
            total = stats['total_samples']
            stats_summary.append({
                'Dataset': split.upper(),
                'Samples': total,
                'Rumor': stats['label_distribution'].get(1, 0),
                'Non-rumor': stats['label_distribution'].get(0, 0),
                'Unverified': stats['label_distribution'].get(2, 0),
                'Images': stats['has_image'],
                'Direct': stats['has_direct_annotation'],
                'Inverse': stats['has_inverse_annotation']
            })
        
        df_summary = pd.DataFrame(stats_summary)
        colors = list(self.colors.values())[:7]
        df_summary.set_index('Dataset').plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Dataset Detailed Statistics', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=0)
        
        # 2. 文本长度vs标签关系
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'text_stats' in self.analysis_results:
            # 按标签分组的文本长度分布
            text_by_label = {0: [], 1: [], 2: []}
            for split, data in self.datasets.items():
                for item_id, item in data.items():
                    caption = item.get('caption', '')
                    label = item.get('label', -1)
                    if caption and label in text_by_label:
                        text_by_label[label].append(len(caption))
            
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
            for label, lengths in text_by_label.items():
                if lengths:
                    ax2.hist(lengths, bins=20, alpha=0.6, 
                            label=self.label_mapping.get(label, f'Label {label}'),
                            color=colors[label % len(colors)])
            
            ax2.set_title('Text Length Distribution by Label', fontweight='bold')
            ax2.set_xlabel('Text Length (characters)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
        
        # 3. 图像尺寸聚类分析
        ax3 = fig.add_subplot(gs[1, :2])
        if 'image_stats' in self.analysis_results and self.analysis_results['image_stats']['valid_images'] > 0:
            widths = self.analysis_results['image_stats']['size_distribution']['width']
            heights = self.analysis_results['image_stats']['size_distribution']['height']
            
            # 计算宽高比
            aspect_ratios = [w/h for w, h in zip(widths, heights)]
            
            scatter = ax3.scatter(widths, heights, c=aspect_ratios, cmap='viridis', alpha=0.6)
            ax3.set_xlabel('Width (pixels)')
            ax3.set_ylabel('Height (pixels)')
            ax3.set_title('Image Size Clustering (color = aspect ratio)', fontweight='bold')
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax3, label='Aspect Ratio')
        
        # 4. 检索效果分析
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'annotation_stats' in self.analysis_results:
            retrieval_data = {
                'Direct Search Coverage': self.analysis_results['annotation_stats']['direct_annotations'],
                'Inverse Search Coverage': self.analysis_results['annotation_stats']['inverse_annotations']
            }
            
            # 计算总样本数
            total_samples = sum(stats['total_samples'] for stats in self.analysis_results['basic_stats'].values())
            
            coverage_rates = [v/total_samples*100 for v in retrieval_data.values()]
            
            bars = ax4.bar(retrieval_data.keys(), coverage_rates, 
                          color=[self.colors['primary'], self.colors['secondary']])
            ax4.set_title('Retrieval Annotation Coverage Rate', fontweight='bold')
            ax4.set_ylabel('Coverage Rate (%)')
            
            for i, v in enumerate(coverage_rates):
                ax4.text(i, v + max(coverage_rates)*0.01, f'{v:.1f}%', ha='center', va='bottom')
        
        # 5. 语言分布饼图
        ax5 = fig.add_subplot(gs[2, 0])
        if 'text_stats' in self.analysis_results:
            lang_data = self.analysis_results['text_stats']['language_distribution']
            if lang_data:
                colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
                ax5.pie(lang_data.values(), labels=lang_data.keys(), autopct='%1.1f%%', colors=colors)
                ax5.set_title('Text Language Distribution', fontweight='bold')
        
        # 6. 标签不平衡分析
        ax6 = fig.add_subplot(gs[2, 1])
        all_labels = Counter()
        for split_stats in self.analysis_results['basic_stats'].values():
            all_labels.update(split_stats['label_distribution'])
        
        if all_labels:
            labels = [self.label_mapping.get(k, f'Label {k}') for k in all_labels.keys()]
            counts = list(all_labels.values())
            
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
            bars = ax6.bar(labels, counts, color=colors)
            ax6.set_title('Label Imbalance Analysis', fontweight='bold')
            ax6.set_ylabel('Number of Samples')
            ax6.tick_params(axis='x', rotation=45)
            
            # 添加不平衡比例
            max_count = max(counts)
            for i, (label, count) in enumerate(zip(labels, counts)):
                ratio = count / max_count
                ax6.text(i, count + max_count*0.01, f'{ratio:.2f}', ha='center', va='bottom')
        
        # 7. 数据质量评估
        ax7 = fig.add_subplot(gs[2, 2])
        quality_metrics = []
        for split, stats in self.analysis_results['basic_stats'].items():
            total = stats['total_samples']
            quality_score = (
                stats['has_image'] / total * 0.4 +
                stats['has_direct_annotation'] / total * 0.3 +
                stats['has_inverse_annotation'] / total * 0.3
            ) * 100
            quality_metrics.append(quality_score)
        
        splits = list(self.analysis_results['basic_stats'].keys())
        colors = [self.colors['accent'], self.colors['warning'], self.colors['info']]
        bars = ax7.bar(splits, quality_metrics, color=colors)
        ax7.set_title('Data Quality Score', fontweight='bold')
        ax7.set_ylabel('Quality Score (%)')
        
        for bar, score in zip(bars, quality_metrics):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.1f}%', ha='center', va='bottom')
        
        # 8. 关键统计指标
        ax8 = fig.add_subplot(gs[2, 3])
        ax8.axis('off')
        
        # 计算关键指标
        total_samples = sum(stats['total_samples'] for stats in self.analysis_results['basic_stats'].values())
        total_images = sum(stats['has_image'] for stats in self.analysis_results['basic_stats'].values())
        total_annotations = self.analysis_results.get('annotation_stats', {}).get('direct_annotations', 0)
        
        avg_text_length = 0
        if 'text_stats' in self.analysis_results:
            avg_text_length = np.mean(self.analysis_results['text_stats']['length_distribution'])
        
        key_metrics = f"""
Key Metrics Summary:

📊 Total Samples: {total_samples:,}
🖼️ Total Images: {total_images:,}
🔍 Total Annotations: {total_annotations:,}
📝 Avg Text Length: {avg_text_length:.1f} chars

💡 Dataset Features:
• Multi-modal: Text+Image+Retrieval
• Multi-lingual: Chinese+English mix
• Multi-source retrieval: Direct+Inverse
• Three-class: Rumor detection task
        """
        
        ax8.text(0.05, 0.95, key_metrics, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
        
        plt.savefig(self.charts_dir / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """生成完整的分析报告"""
        print("\n📄 === 生成分析报告 ===")
        
        report_content = []
        report_content.append("# MR2多模态谣言检测数据集分析报告\n")
        report_content.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_content.append("---\n")
        
        # 1. 执行摘要
        total_samples = sum(stats['total_samples'] for stats in self.analysis_results['basic_stats'].values())
        
        report_content.append("## 执行摘要\n")
        report_content.append(f"MR2数据集包含 **{total_samples:,}** 个多模态样本，专门用于谣言检测任务。")
        report_content.append("数据集具有以下特点：\n")
        report_content.append("- 🔄 **多模态数据**: 结合文本、图像和检索信息")
        report_content.append("- 🌏 **多语言支持**: 中英文混合文本")
        report_content.append("- 🔍 **丰富标注**: 直接检索和反向图像搜索")
        report_content.append("- 📊 **三分类任务**: 谣言、非谣言、未验证\n")
        
        # 2. 数据集统计
        report_content.append("## 数据集统计\n")
        report_content.append("### 数据量分布\n")
        for split, stats in self.analysis_results['basic_stats'].items():
            report_content.append(f"- **{split.upper()}**: {stats['total_samples']:,} 样本")
        
        report_content.append("\n### 标签分布\n")
        all_labels = Counter()
        for split_stats in self.analysis_results['basic_stats'].values():
            all_labels.update(split_stats['label_distribution'])
        
        for label, count in all_labels.items():
            label_name = self.label_mapping.get(label, f'Unknown({label})')
            percentage = count / total_samples * 100
            report_content.append(f"- **{label_name}**: {count:,} ({percentage:.1f}%)")
        
        # 3. 文本分析结果
        if 'text_stats' in self.analysis_results:
            text_stats = self.analysis_results['text_stats']
            report_content.append("\n## 文本内容分析\n")
            report_content.append(f"- **文本总数**: {text_stats['total_texts']:,}")
            report_content.append(f"- **平均长度**: {np.mean(text_stats['length_distribution']):.1f} 字符")
            report_content.append(f"- **平均词数**: {np.mean(text_stats['word_count_distribution']):.1f} 词")
            
            report_content.append("\n### 语言分布")
            for lang, count in text_stats['language_distribution'].items():
                percentage = count / text_stats['total_texts'] * 100
                report_content.append(f"- **{lang}**: {count:,} ({percentage:.1f}%)")
            
            report_content.append("\n### 常见词汇 (Top 10)")
            for word, count in text_stats['common_words'].most_common(10):
                report_content.append(f"- **{word}**: {count} 次")
        
        # 4. 图像分析结果
        if 'image_stats' in self.analysis_results:
            image_stats = self.analysis_results['image_stats']
            report_content.append("\n## 图像数据分析\n")
            report_content.append(f"- **图像总数**: {image_stats['total_images']:,}")
            report_content.append(f"- **有效图像**: {image_stats['valid_images']:,}")
            
            if image_stats['valid_images'] > 0:
                report_content.append(f"- **平均尺寸**: {np.mean(image_stats['size_distribution']['width']):.0f} × {np.mean(image_stats['size_distribution']['height']):.0f} 像素")
                report_content.append(f"- **平均文件大小**: {np.mean(image_stats['file_sizes'])/1024:.1f} KB")
                
                report_content.append("\n### 图像格式分布")
                for format_type, count in image_stats['image_formats'].items():
                    percentage = count / image_stats['valid_images'] * 100
                    report_content.append(f"- **{format_type}**: {count:,} ({percentage:.1f}%)")
        
        # 5. 检索标注分析
        if 'annotation_stats' in self.analysis_results:
            annotation_stats = self.analysis_results['annotation_stats']
            report_content.append("\n## 检索标注分析\n")
            report_content.append(f"- **直接检索标注**: {annotation_stats['direct_annotations']:,}")
            report_content.append(f"- **反向检索标注**: {annotation_stats['inverse_annotations']:,}")
            
            if annotation_stats['direct_stats']['total_retrieved_images']:
                avg_images = np.mean(annotation_stats['direct_stats']['total_retrieved_images'])
                report_content.append(f"- **平均检索图像数**: {avg_images:.1f}")
            
            if annotation_stats['inverse_stats']['entities_count']:
                avg_entities = np.mean(annotation_stats['inverse_stats']['entities_count'])
                report_content.append(f"- **平均识别实体数**: {avg_entities:.1f}")
        
        # 6. 数据质量评估
        report_content.append("\n## 数据质量评估\n")
        for split, stats in self.analysis_results['basic_stats'].items():
            total = stats['total_samples']
            image_rate = stats['has_image'] / total * 100
            direct_rate = stats['has_direct_annotation'] / total * 100
            inverse_rate = stats['has_inverse_annotation'] / total * 100
            
            report_content.append(f"### {split.upper()} 数据集")
            report_content.append(f"- **图像完整性**: {image_rate:.1f}%")
            report_content.append(f"- **直接检索完整性**: {direct_rate:.1f}%")
            report_content.append(f"- **反向检索完整性**: {inverse_rate:.1f}%")
        
        # 7. 建议和结论
        report_content.append("\n## 建议和结论\n")
        report_content.append("### 数据集优势")
        report_content.append("1. **多模态特性**: 提供了文本、图像和检索信息的全面结合")
        report_content.append("2. **真实场景**: 来源于真实的社交媒体谣言检测场景")
        report_content.append("3. **丰富标注**: 包含详细的检索验证信息")
        
        report_content.append("\n### 潜在挑战")
        report_content.append("1. **标签不平衡**: 需要考虑类别不平衡的处理策略")
        report_content.append("2. **多语言处理**: 中英文混合文本需要特殊的预处理方法")
        report_content.append("3. **图像质量**: 部分图像可能存在质量或格式问题")
        
        report_content.append("\n### 建议方法")
        report_content.append("1. **多模态融合**: 设计有效的文本-图像融合策略")
        report_content.append("2. **检索增强**: 利用检索信息进行模型增强")
        report_content.append("3. **预训练模型**: 使用中英文多语言预训练模型")
        
        # 保存报告
        report_text = '\n'.join(report_content)
        report_file = self.reports_dir / 'mr2_dataset_analysis_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✅ 分析报告已保存到: {report_file}")
        return report_text
    
    def run_complete_analysis(self):
        """运行完整的数据集分析流程"""
        print("🚀 === 开始MR2数据集完整分析 ===")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 基础统计分析
        self.basic_statistics()
        
        # 3. 文本分析
        self.text_analysis()
        
        # 4. 图像分析
        self.image_analysis()
        
        # 5. 检索标注分析
        self.annotation_analysis()
        
        # 6. 创建可视化图表
        self.create_visualizations()
        
        # 7. 生成分析报告
        self.generate_report()
        
        print(f"\n🎉 === 分析完成! ===")
        
        if USE_CONFIG_MANAGER:
            print(f"📁 输出目录: {self.charts_dir.parent}")
            print(f"📊 图表目录: {self.charts_dir}")
            print(f"📄 报告目录: {self.reports_dir}")
        else:
            print(f"📁 输出目录: {self.output_dir}")
            print(f"📊 图表目录: {self.charts_dir}")
            print(f"📄 报告目录: {self.reports_dir}")
        
        return self.analysis_results


# 使用示例
if __name__ == "__main__":
    # 创建分析器实例 (自动检测配置)
    analyzer = MR2DatasetAnalyzer()
    
    # 运行完整分析
    results = analyzer.run_complete_analysis()
    
    print("\n" + "="*50)
    print("分析结果预览:")
    print("="*50)
    
    # 打印关键统计信息
    for split, stats in results['basic_stats'].items():
        print(f"\n{split.upper()} 数据集: {stats['total_samples']} 样本")
        print(f"  标签分布: {dict(stats['label_distribution'])}")
        print(f"  数据完整性: 图像({stats['has_image']}) 直接检索({stats['has_direct_annotation']}) 反向检索({stats['has_inverse_annotation']})")