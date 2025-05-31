#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# code/datasets/mr2_analysis.py

"""
MR2多模态谣言检测数据集深度分析
全面分析数据集结构、分布特征，生成可视化图表
修复版本：增强错误处理和数据为空时的处理
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
    """MR2数据集分析器 - 增强版"""
    
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
            self.data_dir = Path(data_dir)
            self.output_dir = Path(output_dir)
            self.charts_dir = self.output_dir / 'charts'
            self.reports_dir = self.output_dir / 'reports'
            self.analysis_dir = self.output_dir / 'analysis'
            
            # 创建输出目录 - 关键修复
            for dir_path in [self.charts_dir, self.reports_dir, self.analysis_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            self.colors = self._default_colors()
            self.label_mapping = {0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'}
            
            print(f"🔧 使用默认配置")
            print(f"🔧 数据目录: {self.data_dir}")
            print(f"🔧 输出目录: {self.output_dir}")
        
        # 存储分析结果
        self.analysis_results = {}
        self.datasets = {}
    
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
    
    def get_color(self, color_key: str, default_color: str = '#666666') -> str:
        """
        安全获取颜色，避免KeyError
        
        Args:
            color_key: 颜色键名
            default_color: 默认颜色
            
        Returns:
            颜色值
        """
        return self.colors.get(color_key, default_color)
    
    def check_data_availability(self) -> Dict[str, bool]:
        """检查数据文件是否可用"""
        print("🔍 检查数据文件可用性...")
        
        availability = {}
        splits = ['train', 'val', 'test']
        
        for split in splits:
            file_path = self.data_dir / f'dataset_items_{split}.json'
            availability[split] = file_path.exists()
            if availability[split]:
                print(f"✅ 找到 {split} 数据文件")
            else:
                print(f"❌ 未找到 {split} 数据文件: {file_path}")
        
        return availability
    
    def create_demo_data(self):
        """创建演示数据（当真实数据不可用时）"""
        print("🔧 创建演示数据...")
        
        # 确保数据目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建演示数据
        demo_texts = [
            "这是一个谣言检测的演示文本",
            "This is a demo text for rumor detection",
            "混合语言的演示文本 mixed language demo",
            "Breaking news about technology advancement",
            "关于新技术发展的重要消息",
            "Fake news spreads faster than real news",
            "虚假信息传播速度比真实信息更快",
            "AI technology is revolutionizing the world",
            "人工智能技术正在改变世界",
            "Climate change affects global economy"
        ]
        
        for split in ['train', 'val', 'test']:
            demo_data = {}
            num_samples = {'train': 8, 'val': 3, 'test': 4}[split]
            
            for i in range(num_samples):
                demo_data[str(i)] = {
                    'caption': demo_texts[i % len(demo_texts)],
                    'label': i % 3,  # 0, 1, 2 循环
                    'language': 'mixed',
                    'image_path': f'{split}/img/{i}.jpg'
                }
            
            # 保存到文件
            demo_file = self.data_dir / f'dataset_items_{split}.json'
            with open(demo_file, 'w', encoding='utf-8') as f:
                json.dump(demo_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 创建 {split} 演示数据: {len(demo_data)} 样本")
        
        return True
        
    def load_data(self):
        """加载所有数据集"""
        print("🔄 开始加载MR2数据集...")
        
        # 检查数据可用性
        availability = self.check_data_availability()
        
        # 如果没有任何数据，创建演示数据
        if not any(availability.values()):
            print("❓ 没有找到真实数据，是否创建演示数据？")
            try:
                self.create_demo_data()
                # 重新检查可用性
                availability = self.check_data_availability()
            except Exception as e:
                print(f"❌ 创建演示数据失败: {e}")
        
        self.datasets = {}
        splits = ['train', 'val', 'test']
        
        for split in splits:
            if availability.get(split, False):
                file_path = self.data_dir / f'dataset_items_{split}.json'
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.datasets[split] = json.load(f)
                    print(f"✅ 加载 {split} 数据: {len(self.datasets[split])} 条")
                except Exception as e:
                    print(f"❌ 加载 {split} 数据失败: {e}")
                    
        return self.datasets
    
    def basic_statistics(self):
        """基础统计分析"""
        print("\n📊 === 基础统计分析 ===")
        
        if not self.datasets:
            print("⚠️  没有可用数据进行统计分析")
            self.analysis_results['basic_stats'] = {}
            return {}
        
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
                    image_path = self.data_dir / item['image_path']
                    if image_path.exists():
                        split_stats['has_image'] += 1
                
                # 检索标注检查
                if 'direct_path' in item:
                    direct_path = self.data_dir / item['direct_path'] / 'direct_annotation.json'
                    if direct_path.exists():
                        split_stats['has_direct_annotation'] += 1
                        
                if 'inv_path' in item:
                    inv_path = self.data_dir / item['inv_path'] / 'inverse_annotation.json'
                    if inv_path.exists():
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
        
        if not self.datasets:
            print("⚠️  没有可用数据进行文本分析")
            self.analysis_results['text_stats'] = {}
            return {}
        
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
                        if len(text_stats['samples_by_length']['short']) < 5:
                            text_stats['samples_by_length']['short'].append(caption)
                    elif text_length < 100:
                        if len(text_stats['samples_by_length']['medium']) < 5:
                            text_stats['samples_by_length']['medium'].append(caption)
                    else:
                        if len(text_stats['samples_by_length']['long']) < 3:
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
        
        if not self.datasets:
            print("⚠️  没有可用数据进行图像分析")
            self.analysis_results['image_stats'] = {}
            return {}
        
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
                    image_path = self.data_dir / item['image_path']
                    
                    if image_path.exists():
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
                                file_size = image_path.stat().st_size
                                image_stats['file_sizes'].append(file_size)
                                
                        except Exception as e:
                            print(f"⚠️  无法读取图像 {image_path}: {e}")
        
        if image_stats['valid_images'] > 0:
            print(f"图像总数: {image_stats['total_images']}")
            print(f"有效图像: {image_stats['valid_images']}")
            print(f"图像格式: {dict(image_stats['image_formats'])}")
            print(f"平均尺寸: {np.mean(image_stats['size_distribution']['width']):.0f} x {np.mean(image_stats['size_distribution']['height']):.0f}")
            print(f"平均文件大小: {np.mean(image_stats['file_sizes'])/1024:.1f} KB")
        else:
            print("⚠️  没有找到有效的图像文件")
        
        self.analysis_results['image_stats'] = image_stats
        return image_stats
    
    def annotation_analysis(self):
        """检索标注数据分析"""
        print("\n🔍 === 检索标注分析 ===")
        
        if not self.datasets:
            print("⚠️  没有可用数据进行标注分析")
            self.analysis_results['annotation_stats'] = {
                'direct_annotations': 0,
                'inverse_annotations': 0,
                'direct_stats': {'total_retrieved_images': [], 'domains': Counter()},
                'inverse_stats': {'entities_count': [], 'common_entities': Counter()}
            }
            return self.analysis_results['annotation_stats']
        
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
                    direct_file = self.data_dir / item['direct_path'] / 'direct_annotation.json'
                    if direct_file.exists():
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
                    inverse_file = self.data_dir / item['inv_path'] / 'inverse_annotation.json'
                    if inverse_file.exists():
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
        
        # 确保输出目录存在
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否有数据
        if not self.analysis_results.get('basic_stats'):
            print("⚠️  没有分析结果，跳过图表生成")
            return
        
        # 设置图表参数
        plt.style.use('default')
        
        try:
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
            
        except Exception as e:
            print(f"❌ 生成图表时出错: {e}")
            print("这可能是由于数据不足或格式问题导致的")
    
    def _plot_basic_distribution(self):
        """绘制基础数据分布 - 增强错误处理"""
        try:
            if not self.analysis_results.get('basic_stats'):
                print("⚠️  跳过基础分布图：没有统计数据")
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('MR2 Dataset Basic Distribution Analysis', fontsize=16, fontweight='bold')
            
            # 1. 数据集大小分布
            stats = self.analysis_results['basic_stats']
            if stats:
                splits = list(stats.keys())
                sizes = [stats[split]['total_samples'] for split in splits]
                
                colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
                axes[0, 0].bar(splits, sizes, color=colors[:len(splits)])
                axes[0, 0].set_title('Dataset Size Distribution')
                axes[0, 0].set_ylabel('Number of Samples')
                for i, v in enumerate(sizes):
                    if v > 0:  # 只在有数据时添加标签
                        axes[0, 0].text(i, v + max(sizes)*0.01, str(v), ha='center', va='bottom')
            else:
                axes[0, 0].text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Dataset Size Distribution')
            
            # 2. 标签分布 (合并所有split)
            all_labels = Counter()
            for split_stats in stats.values():
                all_labels.update(split_stats['label_distribution'])
            
            if all_labels and sum(all_labels.values()) > 0:
                labels = [self.label_mapping.get(k, f'Unknown({k})') for k in all_labels.keys()]
                counts = list(all_labels.values())
                
                axes[0, 1].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)])
                axes[0, 1].set_title('Label Distribution')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Label Data', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Label Distribution')
            
            # 3. 数据完整性分析 - 修复关键错误
            if stats:
                completeness_data = []
                for split, split_stats in stats.items():
                    total = split_stats['total_samples']
                    if total > 0:  # 避免除零错误
                        completeness_data.append({
                            'Split': split,
                            'Has Image': split_stats['has_image'] / total * 100,
                            'Has Direct': split_stats['has_direct_annotation'] / total * 100,
                            'Has Inverse': split_stats['has_inverse_annotation'] / total * 100
                        })
                
                if completeness_data:
                    df_completeness = pd.DataFrame(completeness_data)
                    x = np.arange(len(df_completeness))
                    width = 0.25
                    
                    # 安全访问DataFrame列
                    try:
                        axes[1, 0].bar(x - width, df_completeness['Has Image'], width, label='Has Image', color=self.colors['primary'])
                        axes[1, 0].bar(x, df_completeness['Has Direct'], width, label='Has Direct Search', color=self.colors['secondary'])
                        axes[1, 0].bar(x + width, df_completeness['Has Inverse'], width, label='Has Inverse Search', color=self.colors['tertiary'])
                        
                        axes[1, 0].set_ylabel('Completeness (%)')
                        axes[1, 0].set_title('Data Completeness Analysis')
                        axes[1, 0].set_xticks(x)
                        axes[1, 0].set_xticklabels(df_completeness['Split'])
                        axes[1, 0].legend()
                    except KeyError as e:
                        print(f"⚠️  DataFrame列访问错误: {e}")
                        axes[1, 0].text(0.5, 0.5, 'Data Processing Error', ha='center', va='center', transform=axes[1, 0].transAxes)
                        axes[1, 0].set_title('Data Completeness Analysis')
                else:
                    axes[1, 0].text(0.5, 0.5, 'No Completeness Data', ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Data Completeness Analysis')
            
            # 4. 按split的标签分布 - 增强错误处理
            if stats and all_labels:
                try:
                    split_label_data = []
                    for split, split_stats in stats.items():
                        for label, count in split_stats['label_distribution'].items():
                            split_label_data.append({
                                'Split': split,
                                'Label': self.label_mapping.get(label, f'Unknown({label})'),
                                'Count': count
                            })
                    
                    if split_label_data:
                        df_split_labels = pd.DataFrame(split_label_data)
                        pivot_df = df_split_labels.pivot(index='Split', columns='Label', values='Count').fillna(0)
                        
                        pivot_df.plot(kind='bar', ax=axes[1, 1], color=colors[:len(pivot_df.columns)])
                        axes[1, 1].set_title('Label Distribution by Split')
                        axes[1, 1].set_ylabel('Number of Samples')
                        axes[1, 1].legend(title='Labels')
                        axes[1, 1].tick_params(axis='x', rotation=0)
                    else:
                        axes[1, 1].text(0.5, 0.5, 'No Split Label Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                        axes[1, 1].set_title('Label Distribution by Split')
                except Exception as e:
                    print(f"⚠️  处理标签分布时出错: {e}")
                    axes[1, 1].text(0.5, 0.5, 'Label Processing Error', ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Label Distribution by Split')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Label Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Label Distribution by Split')
            
            plt.tight_layout()
            
            # 安全保存图表
            try:
                output_file = self.charts_dir / 'basic_distribution.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"✅ 基础分布图已保存: {output_file}")
                plt.show()
            except Exception as e:
                print(f"❌ 保存基础分布图失败: {e}")
                plt.show()
                
        except Exception as e:
            print(f"❌ 生成基础分布图失败: {e}")
    
    def _plot_text_distribution(self):
        """绘制文本分布分析 - 增强错误处理"""
        try:
            text_stats = self.analysis_results.get('text_stats')
            if not text_stats or not text_stats.get('length_distribution'):
                print("⚠️  跳过文本分布图：没有文本数据")
                return
                
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Text Content Analysis', fontsize=16, fontweight='bold')
            
            # 1. 文本长度分布
            if text_stats['length_distribution']:
                axes[0, 0].hist(text_stats['length_distribution'], bins=30, color=self.colors['primary'], alpha=0.7)
                axes[0, 0].set_title('Text Length Distribution')
                axes[0, 0].set_xlabel('Number of Characters')
                axes[0, 0].set_ylabel('Frequency')
            else:
                axes[0, 0].text(0.5, 0.5, 'No Text Length Data', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Text Length Distribution')
            
            # 2. 词数分布
            if text_stats['word_count_distribution']:
                axes[0, 1].hist(text_stats['word_count_distribution'], bins=20, color=self.colors['secondary'], alpha=0.7)
                axes[0, 1].set_title('Word Count Distribution')
                axes[0, 1].set_xlabel('Number of Words')
                axes[0, 1].set_ylabel('Frequency')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Word Count Data', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Word Count Distribution')
            
            # 3. 语言分布
            lang_data = text_stats.get('language_distribution')
            if lang_data and sum(lang_data.values()) > 0:
                colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
                axes[0, 2].pie(lang_data.values(), labels=lang_data.keys(), autopct='%1.1f%%', colors=colors[:len(lang_data)])
                axes[0, 2].set_title('Language Distribution')
            else:
                axes[0, 2].text(0.5, 0.5, 'No Language Data', ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Language Distribution')
            
            # 4. 常用词云形式的柱状图
            common_words = text_stats.get('common_words', Counter()).most_common(15)
            if common_words:
                words, counts = zip(*common_words)
                
                axes[1, 0].barh(range(len(words)), counts, color=self.get_color('accent', '#96CEB4'))
                axes[1, 0].set_yticks(range(len(words)))
                axes[1, 0].set_yticklabels(words)
                axes[1, 0].set_title('Most Common Words (Top 15)')
                axes[1, 0].set_xlabel('Frequency')
            else:
                axes[1, 0].text(0.5, 0.5, 'No Word Data', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Most Common Words (Top 15)')
            
            # 5. 字符频率分布
            char_freq = text_stats.get('character_distribution', Counter()).most_common(20)
            if char_freq:
                chars, freqs = zip(*char_freq)
                
                axes[1, 1].bar(range(len(chars)), freqs, color=self.get_color('warning', '#FFEAA7'))
                axes[1, 1].set_xticks(range(len(chars)))
                axes[1, 1].set_xticklabels(chars)
                axes[1, 1].set_title('Character Frequency Distribution (Top 20)')
                axes[1, 1].set_ylabel('Frequency')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Character Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Character Frequency Distribution (Top 20)')
            
            # 6. 文本长度统计摘要
            if text_stats.get('length_distribution'):
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
            else:
                axes[1, 2].text(0.5, 0.5, 'No Statistics Available', ha='center', va='center', transform=axes[1, 2].transAxes)
            
            plt.tight_layout()
            
            # 安全保存图表
            try:
                output_file = self.charts_dir / 'text_distribution.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"✅ 文本分布图已保存: {output_file}")
                plt.show()
            except Exception as e:
                print(f"❌ 保存文本分布图失败: {e}")
                plt.show()
                
        except Exception as e:
            print(f"❌ 生成文本分布图失败: {e}")
    
    def _plot_image_distribution(self):
        """绘制图像分布分析 - 增强错误处理"""
        try:
            image_stats = self.analysis_results.get('image_stats')
            if not image_stats or image_stats.get('valid_images', 0) == 0:
                print("⚠️  跳过图像分布图：没有有效图像数据")
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Image Data Analysis', fontsize=16, fontweight='bold')
            
            # 1. 图像尺寸散点图
            widths = image_stats.get('size_distribution', {}).get('width', [])
            heights = image_stats.get('size_distribution', {}).get('height', [])
            
            if widths and heights:
                axes[0, 0].scatter(widths, heights, alpha=0.6, color=self.colors['primary'])
                axes[0, 0].set_xlabel('Width (pixels)')
                axes[0, 0].set_ylabel('Height (pixels)')
                axes[0, 0].set_title('Image Size Distribution')
            else:
                axes[0, 0].text(0.5, 0.5, 'No Size Data', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Image Size Distribution')
            
            # 2. 宽度分布直方图
            if widths:
                axes[0, 1].hist(widths, bins=20, color=self.colors['secondary'], alpha=0.7)
                axes[0, 1].set_xlabel('Width (pixels)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Image Width Distribution')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Width Data', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Image Width Distribution')
            
            # 3. 高度分布直方图
            if heights:
                axes[1, 0].hist(heights, bins=20, color=self.colors['tertiary'], alpha=0.7)
                axes[1, 0].set_xlabel('Height (pixels)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Image Height Distribution')
            else:
                axes[1, 0].text(0.5, 0.5, 'No Height Data', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Image Height Distribution')
            
            # 4. 图像格式分布
            format_data = image_stats.get('image_formats')
            if format_data and sum(format_data.values()) > 0:
                colors = [
                    self.get_color('primary', '#FF6B6B'), 
                    self.get_color('secondary', '#4ECDC4'), 
                    self.get_color('tertiary', '#45B7D1'), 
                    self.get_color('accent', '#96CEB4')
                ]
                axes[1, 1].pie(format_data.values(), labels=format_data.keys(), autopct='%1.1f%%', colors=colors[:len(format_data)])
                axes[1, 1].set_title('Image Format Distribution')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Format Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Image Format Distribution')
            
            plt.tight_layout()
            
            # 安全保存图表
            try:
                output_file = self.charts_dir / 'image_distribution.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"✅ 图像分布图已保存: {output_file}")
                plt.show()
            except Exception as e:
                print(f"❌ 保存图像分布图失败: {e}")
                plt.show()
                
        except Exception as e:
            print(f"❌ 生成图像分布图失败: {e}")
    
    def _plot_annotation_analysis(self):
        """绘制检索标注分析 - 修复版本，显示有意义的数据"""
        try:
            annotation_stats = self.analysis_results.get('annotation_stats')
            if not annotation_stats:
                print("⚠️  跳过标注分析图：没有标注数据")
                return
                
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Retrieval Annotation Analysis', fontsize=16, fontweight='bold')
            
            # 1. 检索数量对比 (保持原有的正确内容)
            annotation_counts = [
                annotation_stats.get('direct_annotations', 0),
                annotation_stats.get('inverse_annotations', 0)
            ]
            
            if max(annotation_counts) > 0:
                axes[0, 0].bar(['Direct Search', 'Inverse Search'], annotation_counts, 
                              color=[self.colors['primary'], self.colors['secondary']])
                axes[0, 0].set_title('Annotation Count')
                axes[0, 0].set_ylabel('Number of Annotations')
                for i, v in enumerate(annotation_counts):
                    if v > 0:
                        axes[0, 0].text(i, v + max(annotation_counts)*0.01, str(v), ha='center', va='bottom')
            else:
                axes[0, 0].text(0.5, 0.5, 'No Annotation Data', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Annotation Count')
            
            # 2. 直接检索文件可用性
            if annotation_stats.get('direct_available_files', 0) > 0 or annotation_stats.get('inverse_available_files', 0) > 0:
                file_counts = [
                    annotation_stats.get('direct_available_files', 0),
                    annotation_stats.get('inverse_available_files', 0)
                ]
                axes[0, 1].bar(['Direct Files', 'Inverse Files'], file_counts,
                              color=[self.colors['tertiary'], self.colors['accent']])
                axes[0, 1].set_title('Available Annotation Files')
                axes[0, 1].set_ylabel('Number of Files')
                for i, v in enumerate(file_counts):
                    if v > 0:
                        axes[0, 1].text(i, v + 0.1, str(v), ha='center', va='bottom')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Files Found', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Available Annotation Files')
            
            # 3. 域名分布 (如果有直接检索数据)
            domains = annotation_stats.get('direct_stats', {}).get('domains', Counter())
            if domains and sum(domains.values()) > 0:
                top_domains = domains.most_common(5)
                domain_names, domain_counts = zip(*top_domains)
                
                axes[0, 2].barh(range(len(domain_names)), domain_counts, 
                               color=self.get_color('warning', '#FFEAA7'))
                axes[0, 2].set_yticks(range(len(domain_names)))
                axes[0, 2].set_yticklabels(domain_names)
                axes[0, 2].set_title('Top Domains (Direct Search)')
                axes[0, 2].set_xlabel('Count')
            else:
                axes[0, 2].text(0.5, 0.5, 'No Domain Data', ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Top Domains (Direct Search)')
            
            # 4. 实体统计 (如果有反向检索数据)
            entities_count = annotation_stats.get('inverse_stats', {}).get('entities_count', [])
            if entities_count:
                axes[1, 0].hist(entities_count, bins=10, color=self.get_color('info', '#DDA0DD'), alpha=0.7)
                axes[1, 0].set_title('Entities Count Distribution')
                axes[1, 0].set_xlabel('Number of Entities')
                axes[1, 0].set_ylabel('Frequency')
            else:
                axes[1, 0].text(0.5, 0.5, 'No Entity Data', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Entities Count Distribution')
            
            # 5. 常见实体词云 (如果有实体数据)
            common_entities = annotation_stats.get('inverse_stats', {}).get('common_entities', Counter())
            if common_entities and sum(common_entities.values()) > 0:
                top_entities = common_entities.most_common(10)
                entity_names, entity_counts = zip(*top_entities)
                
                axes[1, 1].barh(range(len(entity_names)), entity_counts,
                               color=self.get_color('success', '#98FB98'))
                axes[1, 1].set_yticks(range(len(entity_names)))
                axes[1, 1].set_yticklabels(entity_names)
                axes[1, 1].set_title('Most Common Entities')
                axes[1, 1].set_xlabel('Frequency')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Entity Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Most Common Entities')
            
            # 6. 匹配类型分布
            fully_matched = annotation_stats.get('inverse_stats', {}).get('fully_matched', [])
            partially_matched = annotation_stats.get('inverse_stats', {}).get('partially_matched', [])
            
            if fully_matched or partially_matched:
                match_types = ['Fully Matched', 'Partially Matched']
                match_counts = [
                    sum(fully_matched) if fully_matched else 0,
                    sum(partially_matched) if partially_matched else 0
                ]
                
                if sum(match_counts) > 0:
                    axes[1, 2].pie(match_counts, labels=match_types, autopct='%1.1f%%',
                                  colors=[self.colors['primary'], self.colors['secondary']])
                    axes[1, 2].set_title('Match Types Distribution')
                else:
                    axes[1, 2].text(0.5, 0.5, 'No Match Data', ha='center', va='center', transform=axes[1, 2].transAxes)
                    axes[1, 2].set_title('Match Types Distribution')
            else:
                axes[1, 2].text(0.5, 0.5, 'No Match Data', ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Match Types Distribution')
            
            plt.tight_layout()
            
            # 安全保存图表
            try:
                output_file = self.charts_dir / 'annotation_analysis.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"✅ 标注分析图已保存: {output_file}")
                plt.show()
            except Exception as e:
                print(f"❌ 保存标注分析图失败: {e}")
                plt.show()
                
        except Exception as e:
            print(f"❌ 生成标注分析图失败: {e}")
    
    def _create_dashboard(self):
        """创建综合分析仪表板 - 修复版本，显示完整的仪表板"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('MR2 Dataset Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # 获取基础统计数据
            stats = self.analysis_results.get('basic_stats', {})
            text_stats = self.analysis_results.get('text_stats', {})
            image_stats = self.analysis_results.get('image_stats', {})
            
            # 1. 数据集概览 (保持原有的正确内容)
            if stats:
                splits = list(stats.keys())
                sizes = [stats[split]['total_samples'] for split in splits]
                
                if sizes and max(sizes) > 0:
                    colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
                    axes[0, 0].bar(splits, sizes, color=colors[:len(splits)])
                    axes[0, 0].set_title('Dataset Overview')
                    axes[0, 0].set_ylabel('Samples')
                    # 添加数值标签
                    for i, v in enumerate(sizes):
                        axes[0, 0].text(i, v + max(sizes)*0.01, str(v), ha='center', va='bottom')
                else:
                    axes[0, 0].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[0, 0].transAxes)
                    axes[0, 0].set_title('Dataset Overview')
            else:
                axes[0, 0].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Dataset Overview')
            
            # 2. 标签分布 (保持原有的正确内容)
            all_labels = Counter()
            for split_stats in stats.values():
                all_labels.update(split_stats.get('label_distribution', {}))
            
            if all_labels and sum(all_labels.values()) > 0:
                labels = [self.label_mapping.get(k, f'Label {k}') for k in all_labels.keys()]
                counts = list(all_labels.values())
                colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
                
                axes[0, 1].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)])
                axes[0, 1].set_title('Label Distribution')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Labels', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Label Distribution')
            
            # 3. 文本长度分布 (保持原有的正确内容)
            if text_stats.get('length_distribution'):
                axes[0, 2].hist(text_stats['length_distribution'], bins=20, color=self.get_color('accent', '#96CEB4'), alpha=0.7)
                axes[0, 2].set_title('Text Length Distribution')
                axes[0, 2].set_xlabel('Characters')
                axes[0, 2].set_ylabel('Frequency')
            else:
                axes[0, 2].text(0.5, 0.5, 'No Text Data', ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Text Length Distribution')
            
            # 4. 语言分析 - 修复为真实的语言分布
            lang_data = text_stats.get('language_distribution', {})
            if lang_data and sum(lang_data.values()) > 0:
                languages = list(lang_data.keys())
                lang_counts = list(lang_data.values())
                colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
                
                axes[1, 0].pie(lang_counts, labels=languages, autopct='%1.1f%%', colors=colors[:len(languages)])
                axes[1, 0].set_title('Language Analysis')
            else:
                axes[1, 0].text(0.5, 0.5, 'No Language Data', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Language Analysis')
            
            # 5. 质量指标 - 显示数据完整性
            if stats:
                quality_metrics = []
                quality_values = []
                
                # 计算各种完整性指标
                total_samples = sum(s.get('total_samples', 0) for s in stats.values())
                total_images = sum(s.get('has_image', 0) for s in stats.values())
                total_direct = sum(s.get('has_direct_annotation', 0) for s in stats.values())
                total_inverse = sum(s.get('has_inverse_annotation', 0) for s in stats.values())
                
                if total_samples > 0:
                    quality_metrics = ['Image\nCompleteness', 'Direct\nRetrieval', 'Inverse\nRetrieval', 'Text\nAvailability']
                    quality_values = [
                        (total_images / total_samples) * 100,
                        (total_direct / total_samples) * 100,
                        (total_inverse / total_samples) * 100,
                        (text_stats.get('total_texts', 0) / total_samples) * 100 if total_samples > 0 else 0
                    ]
                    
                    bars = axes[1, 1].bar(quality_metrics, quality_values, 
                                         color=[self.colors['primary'], self.colors['secondary'], 
                                               self.colors['tertiary'], self.colors['accent']])
                    axes[1, 1].set_title('Quality Metrics')
                    axes[1, 1].set_ylabel('Completeness (%)')
                    axes[1, 1].set_ylim(0, 100)
                    
                    # 添加百分比标签
                    for bar, value in zip(bars, quality_values):
                        height = bar.get_height()
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                                        f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
                else:
                    axes[1, 1].text(0.5, 0.5, 'No Quality Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Quality Metrics')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Quality Data', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Quality Metrics')
            
            # 6. 汇总统计 - 显示关键数字
            axes[1, 2].axis('off')
            
            # 收集汇总信息
            summary_info = []
            if stats:
                total_samples = sum(s.get('total_samples', 0) for s in stats.values())
                summary_info.append(f"Total Samples: {total_samples:,}")
                
                # 标签分布摘要
                if all_labels:
                    max_label = max(all_labels, key=all_labels.get)
                    max_label_name = self.label_mapping.get(max_label, f'Label {max_label}')
                    summary_info.append(f"Most Common: {max_label_name}")
            
            if text_stats:
                total_texts = text_stats.get('total_texts', 0)
                summary_info.append(f"Text Items: {total_texts:,}")
                
                if text_stats.get('length_distribution'):
                    avg_length = np.mean(text_stats['length_distribution'])
                    summary_info.append(f"Avg Length: {avg_length:.0f} chars")
            
            if image_stats:
                valid_images = image_stats.get('valid_images', 0)
                summary_info.append(f"Valid Images: {valid_images:,}")
            
            # 显示汇总信息
            if summary_info:
                summary_text = '\n'.join(summary_info)
                axes[1, 2].text(0.1, 0.7, f'Summary Statistics:\n\n{summary_text}', 
                               transform=axes[1, 2].transAxes, fontsize=14,
                               verticalalignment='top', 
                               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
            else:
                axes[1, 2].text(0.5, 0.5, 'No Summary Data', ha='center', va='center', transform=axes[1, 2].transAxes)
            
            axes[1, 2].set_title('Summary Statistics')
            
            plt.tight_layout()
            
            # 安全保存图表
            try:
                output_file = self.charts_dir / 'comprehensive_dashboard.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"✅ 综合仪表板已保存: {output_file}")
                plt.show()
            except Exception as e:
                print(f"❌ 保存综合仪表板失败: {e}")
                plt.show()
                
        except Exception as e:
            print(f"❌ 生成综合仪表板失败: {e}")
    
    def generate_report(self):
        """生成完整的分析报告 - 增强错误处理"""
        print("\n📄 === 生成分析报告 ===")
        
        try:
            report_content = []
            report_content.append("# MR2多模态谣言检测数据集分析报告\n")
            report_content.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_content.append("---\n")
            
            # 基础统计
            stats = self.analysis_results.get('basic_stats', {})
            if stats:
                total_samples = sum(s.get('total_samples', 0) for s in stats.values())
                report_content.append("## 执行摘要\n")
                report_content.append(f"MR2数据集包含 **{total_samples:,}** 个多模态样本。\n")
                
                # 数据集统计
                report_content.append("## 数据集统计\n")
                report_content.append("### 数据量分布\n")
                for split, split_stats in stats.items():
                    report_content.append(f"- **{split.upper()}**: {split_stats.get('total_samples', 0):,} 样本")
                
                # 标签分布
                all_labels = Counter()
                for split_stats in stats.values():
                    all_labels.update(split_stats.get('label_distribution', {}))
                
                if all_labels:
                    report_content.append("\n### 标签分布\n")
                    for label, count in all_labels.items():
                        label_name = self.label_mapping.get(label, f'Unknown({label})')
                        percentage = count / total_samples * 100 if total_samples > 0 else 0
                        report_content.append(f"- **{label_name}**: {count:,} ({percentage:.1f}%)")
            else:
                report_content.append("## 执行摘要\n")
                report_content.append("没有找到有效的数据集文件。\n")
            
            # 文本分析结果
            text_stats = self.analysis_results.get('text_stats', {})
            if text_stats and text_stats.get('total_texts', 0) > 0:
                report_content.append("\n## 文本内容分析\n")
                report_content.append(f"- **文本总数**: {text_stats['total_texts']:,}")
                
                if text_stats.get('length_distribution'):
                    avg_length = np.mean(text_stats['length_distribution'])
                    report_content.append(f"- **平均长度**: {avg_length:.1f} 字符")
                
                if text_stats.get('word_count_distribution'):
                    avg_words = np.mean(text_stats['word_count_distribution'])
                    report_content.append(f"- **平均词数**: {avg_words:.1f} 词")
                
                # 语言分布
                lang_dist = text_stats.get('language_distribution', {})
                if lang_dist:
                    report_content.append("\n### 语言分布")
                    total_texts = text_stats['total_texts']
                    for lang, count in lang_dist.items():
                        percentage = count / total_texts * 100 if total_texts > 0 else 0
                        report_content.append(f"- **{lang}**: {count:,} ({percentage:.1f}%)")
            
            # 数据质量评估
            if stats:
                report_content.append("\n## 数据质量评估\n")
                for split, split_stats in stats.items():
                    total = split_stats.get('total_samples', 0)
                    if total > 0:
                        image_rate = split_stats.get('has_image', 0) / total * 100
                        direct_rate = split_stats.get('has_direct_annotation', 0) / total * 100
                        inverse_rate = split_stats.get('has_inverse_annotation', 0) / total * 100
                        
                        report_content.append(f"### {split.upper()} 数据集")
                        report_content.append(f"- **图像完整性**: {image_rate:.1f}%")
                        report_content.append(f"- **直接检索完整性**: {direct_rate:.1f}%")
                        report_content.append(f"- **反向检索完整性**: {inverse_rate:.1f}%")
            
            # 建议和结论
            report_content.append("\n## 建议和结论\n")
            report_content.append("### 数据集特点")
            report_content.append("1. **多模态数据**: 结合文本、图像和检索信息")
            report_content.append("2. **多语言支持**: 中英文混合文本")
            report_content.append("3. **丰富标注**: 包含检索验证信息")
            
            # 保存报告
            report_text = '\n'.join(report_content)
            
            # 确保报告目录存在
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            report_file = self.reports_dir / 'mr2_dataset_analysis_report.md'
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"✅ 分析报告已保存到: {report_file}")
            return report_text
            
        except Exception as e:
            print(f"❌ 生成分析报告失败: {e}")
            return ""
    
    def run_complete_analysis(self):
        """运行完整的数据集分析流程"""
        print("🚀 === 开始MR2数据集完整分析 ===")
        
        try:
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
            print(f"📁 输出目录: {self.charts_dir.parent}")
            print(f"📊 图表目录: {self.charts_dir}")
            print(f"📄 报告目录: {self.reports_dir}")
            
            return self.analysis_results
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            print("这可能是由于数据文件缺失或格式问题导致的")
            return {}

# 主执行代码
if __name__ == "__main__":
    print("🔍 MR2数据集深度分析工具")
    
    try:
        # 创建分析器实例 (自动检测配置)
        analyzer = MR2DatasetAnalyzer()
        
        # 运行完整分析
        results = analyzer.run_complete_analysis()
        
        # 显示结果摘要
        if results and results.get('basic_stats'):
            print("\n" + "="*50)
            print("分析结果预览:")
            print("="*50)
            
            for split, stats in results['basic_stats'].items():
                print(f"\n{split.upper()} 数据集: {stats['total_samples']} 样本")
                print(f"  标签分布: {dict(stats['label_distribution'])}")
                print(f"  数据完整性: 图像({stats['has_image']}) 直接检索({stats['has_direct_annotation']}) 反向检索({stats['has_inverse_annotation']})")
        else:
            print("\n⚠️  分析未完成或没有有效数据")
            print("请检查数据文件是否存在于正确的位置")
    
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        print("请检查:")
        print("1. 数据文件是否存在")
        print("2. 依赖包是否安装完整")
        print("3. 输出目录是否有写入权限")
    
    print("\n✅ 程序执行完成")