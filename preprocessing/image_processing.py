#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# preprocessing/image_processing.py

"""
MR2图像预处理模块
专门处理MR2数据集中的图像数据，包括：
- 图像加载和验证
- 尺寸标准化和缩放
- 数据增强
- 特征提取
- 批量处理
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 配置管理
try:
    from utils.config_manager import get_data_config, get_data_dir
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False

import logging
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    图像预处理器
    专门为MR2数据集设计的图像处理工具
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        初始化图像处理器
        
        Args:
            target_size: 目标图像尺寸 (height, width)
        """
        self.target_size = target_size
        
        # 加载配置
        if USE_CONFIG:
            try:
                config = get_data_config()
                self.processing_config = config.get('processing', {}).get('image', {})
                self.data_dir = get_data_dir()
            except:
                self.processing_config = {}
                self.data_dir = Path('data')
        else:
            self.processing_config = {}
            self.data_dir = Path('data')
        
        # 设置处理参数
        self.normalize_mean = self.processing_config.get('normalize_mean', [0.485, 0.456, 0.406])
        self.normalize_std = self.processing_config.get('normalize_std', [0.229, 0.224, 0.225])
        self.quality_threshold = self.processing_config.get('quality_threshold', 0.3)
        
        # 初始化变换（放在参数设置之后）
        self.setup_transforms()
        
        print(f"🖼️  图像处理器初始化完成")
        print(f"   目标尺寸: {self.target_size}")
        print(f"   数据目录: {self.data_dir}")
    
    def setup_transforms(self):
        """设置图像变换"""
        
        # 基础变换 - 用于训练
        self.train_transforms = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
        
        # 验证变换 - 用于验证和测试
        self.val_transforms = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
        
        # 无标准化变换 - 用于可视化
        self.visual_transforms = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])
    
    def load_image(self, image_path: Union[str, Path], mode: str = 'RGB') -> Optional[Image.Image]:
        """
        加载图像文件
        
        Args:
            image_path: 图像文件路径
            mode: 图像模式 ('RGB', 'RGBA', 'L')
            
        Returns:
            PIL Image对象，如果失败返回None
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                logger.error(f"图像文件不存在: {image_path}")
                return None
            
            # 加载图像
            image = Image.open(image_path)
            
            # 转换模式
            if image.mode != mode:
                image = image.convert(mode)
            
            # 验证图像质量
            if not self.validate_image(image):
                logger.warning(f"图像质量检查未通过: {image_path}")
                return None
            
            return image
            
        except Exception as e:
            logger.error(f"加载图像失败 {image_path}: {e}")
            return None
    
    def validate_image(self, image: Image.Image) -> bool:
        """
        验证图像质量
        
        Args:
            image: PIL Image对象
            
        Returns:
            是否通过验证
        """
        # 检查图像尺寸
        width, height = image.size
        if width < 50 or height < 50:
            return False
        
        # 检查图像数据 - 修复验证方法
        try:
            # 使用copy()方法验证图像数据而不是verify()
            _ = image.copy()
            return True
        except Exception:
            return False
    
    def get_image_info(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取图像基本信息
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像信息字典
        """
        try:
            with Image.open(image_path) as image:
                file_size = os.path.getsize(image_path)
                
                return {
                    'path': str(image_path),
                    'filename': os.path.basename(image_path),
                    'format': image.format,
                    'mode': image.mode,
                    'size': image.size,
                    'width': image.width,
                    'height': image.height,
                    'file_size': file_size,
                    'file_size_mb': round(file_size / (1024 * 1024), 2),
                    'aspect_ratio': round(image.width / image.height, 2)
                }
        except Exception as e:
            logger.error(f"获取图像信息失败 {image_path}: {e}")
            return {}
    
    def resize_image(self, image: Image.Image, size: Optional[Tuple[int, int]] = None, 
                     method: str = 'lanczos') -> Image.Image:
        """
        调整图像尺寸
        
        Args:
            image: PIL Image对象
            size: 目标尺寸，默认使用self.target_size
            method: 重采样方法 ('lanczos', 'bilinear', 'bicubic')
            
        Returns:
            调整尺寸后的图像
        """
        if size is None:
            size = self.target_size
        
        # 选择重采样方法
        resample_methods = {
            'lanczos': Image.Resampling.LANCZOS,
            'bilinear': Image.Resampling.BILINEAR,
            'bicubic': Image.Resampling.BICUBIC
        }
        resample = resample_methods.get(method, Image.Resampling.LANCZOS)
        
        return image.resize(size, resample)
    
    def apply_augmentation(self, image: Image.Image, augment_type: str = 'light') -> Image.Image:
        """
        应用数据增强
        
        Args:
            image: PIL Image对象
            augment_type: 增强类型 ('light', 'medium', 'heavy')
            
        Returns:
            增强后的图像
        """
        if augment_type == 'light':
            # 轻度增强
            if np.random.random() > 0.5:
                image = F.hflip(image)  # 水平翻转
            if np.random.random() > 0.7:
                angle = np.random.uniform(-5, 5)
                image = F.rotate(image, angle)  # 小角度旋转
                
        elif augment_type == 'medium':
            # 中度增强
            if np.random.random() > 0.5:
                image = F.hflip(image)
            if np.random.random() > 0.6:
                angle = np.random.uniform(-10, 10)
                image = F.rotate(image, angle)
            if np.random.random() > 0.6:
                # 亮度调整
                enhancer = ImageEnhance.Brightness(image)
                factor = np.random.uniform(0.8, 1.2)
                image = enhancer.enhance(factor)
                
        elif augment_type == 'heavy':
            # 重度增强
            if np.random.random() > 0.5:
                image = F.hflip(image)
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                image = F.rotate(image, angle)
            if np.random.random() > 0.5:
                # 亮度调整
                enhancer = ImageEnhance.Brightness(image)
                factor = np.random.uniform(0.7, 1.3)
                image = enhancer.enhance(factor)
            if np.random.random() > 0.5:
                # 对比度调整
                enhancer = ImageEnhance.Contrast(image)
                factor = np.random.uniform(0.8, 1.2)
                image = enhancer.enhance(factor)
            if np.random.random() > 0.3:
                # 添加轻微模糊
                image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return image
    
    def process_single_image(self, image_path: Union[str, Path], 
                           transform_type: str = 'val',
                           apply_augment: bool = False,
                           augment_type: str = 'light') -> Optional[torch.Tensor]:
        """
        处理单张图像
        
        Args:
            image_path: 图像文件路径
            transform_type: 变换类型 ('train', 'val', 'visual')
            apply_augment: 是否应用数据增强
            augment_type: 增强类型
            
        Returns:
            处理后的tensor，失败返回None
        """
        try:
            # 加载图像
            image = self.load_image(image_path)
            if image is None:
                return None
            
            # 应用数据增强
            if apply_augment:
                image = self.apply_augmentation(image, augment_type)
            
            # 选择变换
            if transform_type == 'train':
                transforms_fn = self.train_transforms
            elif transform_type == 'val':
                transforms_fn = self.val_transforms
            elif transform_type == 'visual':
                transforms_fn = self.visual_transforms
            else:
                transforms_fn = self.val_transforms
            
            # 应用变换
            tensor = transforms_fn(image)
            return tensor
            
        except Exception as e:
            logger.error(f"处理图像失败 {image_path}: {e}")
            return None
    
    def extract_image_features(self, image: Image.Image) -> Dict[str, float]:
        """
        提取图像特征 - 修复计算错误
        
        Args:
            image: PIL Image对象
            
        Returns:
            图像特征字典
        """
        try:
            # 确保图像是PIL Image对象
            if not isinstance(image, Image.Image):
                logger.error(f"输入不是PIL Image对象: {type(image)}")
                return {}
            
            # 转换为numpy数组
            img_array = np.array(image)
            
            # 检查数组是否有效
            if img_array is None or img_array.size == 0:
                logger.error("转换为numpy数组失败")
                return {}
            
            features = {}
            
            # 基本特征
            features['width'] = float(image.width)
            features['height'] = float(image.height)
            features['aspect_ratio'] = float(image.width / image.height)
            features['total_pixels'] = float(image.width * image.height)
            
            # 颜色特征
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:  # 彩色图像
                # RGB均值
                features['mean_r'] = float(np.mean(img_array[:, :, 0]))
                features['mean_g'] = float(np.mean(img_array[:, :, 1]))
                features['mean_b'] = float(np.mean(img_array[:, :, 2]))
                
                # RGB标准差
                features['std_r'] = float(np.std(img_array[:, :, 0]))
                features['std_g'] = float(np.std(img_array[:, :, 1]))
                features['std_b'] = float(np.std(img_array[:, :, 2]))
                
                # 整体亮度
                features['brightness'] = float(np.mean(img_array))
                
                # 对比度（简单估计）
                features['contrast'] = float(np.std(img_array))
                
                # 转换为灰度用于边缘检测
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
            else:  # 灰度图像或单通道
                if len(img_array.shape) == 3:
                    img_array = img_array[:, :, 0]  # 取第一个通道
                
                features['mean_gray'] = float(np.mean(img_array))
                features['std_gray'] = float(np.std(img_array))
                features['brightness'] = float(np.mean(img_array))
                features['contrast'] = float(np.std(img_array))
                
                gray = img_array.astype(np.uint8)
            
            # 边缘密度
            try:
                edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
                features['edge_density'] = float(np.sum(edges > 0) / edges.size)
            except:
                features['edge_density'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"提取图像特征失败: {e}")
            return {}
    
    def process_mr2_dataset(self, splits: List[str] = ['train', 'val', 'test'], 
                           save_features: bool = True) -> Dict[str, Dict]:
        """
        处理MR2数据集的所有图像
        
        Args:
            splits: 要处理的数据划分
            save_features: 是否保存特征到文件
            
        Returns:
            处理结果字典
        """
        print(f"🔄 开始处理MR2数据集图像...")
        
        results = {}
        
        for split in splits:
            print(f"\n📂 处理 {split} 数据集")
            
            # 加载数据集信息
            dataset_file = self.data_dir / f'dataset_items_{split}.json'
            if not dataset_file.exists():
                print(f"⚠️  数据集文件不存在: {dataset_file}")
                continue
            
            with open(dataset_file, 'r', encoding='utf-8') as f:
                dataset_items = json.load(f)
            
            split_results = {
                'total_items': len(dataset_items),
                'processed_images': 0,
                'failed_images': 0,
                'image_info': {},
                'image_features': {}
            }
            
            # 处理每个数据项
            for item_id, item_data in dataset_items.items():
                if 'image_path' not in item_data:
                    continue
                
                # 构建完整图像路径
                image_path = self.data_dir / item_data['image_path']
                
                if not image_path.exists():
                    print(f"⚠️  图像文件不存在: {image_path}")
                    split_results['failed_images'] += 1
                    continue
                
                try:
                    # 获取图像信息
                    img_info = self.get_image_info(image_path)
                    split_results['image_info'][item_id] = img_info
                    
                    # 加载图像并提取特征 - 修复这里的处理逻辑
                    image = self.load_image(image_path)
                    if image is not None:
                        features = self.extract_image_features(image)
                        split_results['image_features'][item_id] = features
                        split_results['processed_images'] += 1
                        
                        if split_results['processed_images'] % 50 == 0:
                            print(f"  已处理 {split_results['processed_images']} 张图像")
                    else:
                        split_results['failed_images'] += 1
                        
                except Exception as e:
                    logger.error(f"处理图像失败 {image_path}: {e}")
                    split_results['failed_images'] += 1
            
            results[split] = split_results
            
            print(f"✅ {split} 数据集处理完成:")
            print(f"  总数: {split_results['total_items']}")
            print(f"  成功: {split_results['processed_images']}")
            print(f"  失败: {split_results['failed_images']}")
            
            # 保存特征
            if save_features:
                self.save_image_features(split_results, split)
        
        return results
    
    def save_image_features(self, features_data: Dict, split: str):
        """
        保存图像特征到文件
        
        Args:
            features_data: 特征数据
            split: 数据划分名称
        """
        # 创建processed目录
        processed_dir = self.data_dir / 'processed'
        processed_dir.mkdir(exist_ok=True)
        
        # 保存图像信息
        info_file = processed_dir / f'{split}_image_info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(features_data['image_info'], f, indent=2, ensure_ascii=False)
        
        # 保存图像特征
        features_file = processed_dir / f'{split}_image_features.json'
        with open(features_file, 'w', encoding='utf-8') as f:
            json.dump(features_data['image_features'], f, indent=2, ensure_ascii=False)
        
        print(f"💾 图像特征已保存到: {processed_dir}")
    
    def create_image_statistics(self, results: Dict) -> Dict[str, Any]:
        """
        创建图像统计信息
        
        Args:
            results: 处理结果
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_images': 0,
            'successful_images': 0,
            'failed_images': 0,
            'size_distribution': {
                'widths': [],
                'heights': [],
                'file_sizes': []
            },
            'format_distribution': {},
            'feature_statistics': {}
        }
        
        for split, split_data in results.items():
            stats['total_images'] += split_data['total_items']
            stats['successful_images'] += split_data['processed_images']
            stats['failed_images'] += split_data['failed_images']
            
            # 收集尺寸信息
            for item_id, img_info in split_data['image_info'].items():
                stats['size_distribution']['widths'].append(img_info.get('width', 0))
                stats['size_distribution']['heights'].append(img_info.get('height', 0))
                stats['size_distribution']['file_sizes'].append(img_info.get('file_size', 0))
                
                # 格式分布
                img_format = img_info.get('format', 'Unknown')
                stats['format_distribution'][img_format] = stats['format_distribution'].get(img_format, 0) + 1
        
        # 计算统计值
        if stats['size_distribution']['widths']:
            stats['avg_width'] = np.mean(stats['size_distribution']['widths'])
            stats['avg_height'] = np.mean(stats['size_distribution']['heights'])
            stats['avg_file_size'] = np.mean(stats['size_distribution']['file_sizes'])
        
        return stats
    
    def batch_process_images(self, image_paths: List[Union[str, Path]], 
                           transform_type: str = 'val',
                           batch_size: int = 32) -> List[torch.Tensor]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            transform_type: 变换类型
            batch_size: 批次大小
            
        Returns:
            处理后的tensor列表
        """
        processed_tensors = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                tensor = self.process_single_image(path, transform_type)
                if tensor is not None:
                    batch_tensors.append(tensor)
            
            if batch_tensors:
                # 堆叠tensor
                batch_tensor = torch.stack(batch_tensors)
                processed_tensors.append(batch_tensor)
        
        return processed_tensors


# 使用示例和测试代码
if __name__ == "__main__":
    print("🖼️  测试图像处理模块")
    
    # 创建图像处理器
    processor = ImageProcessor(target_size=(224, 224))
    
    # 测试单张图像处理
    print("\n📝 === 单张图像处理测试 ===")
    
    # 假设的图像路径（你需要替换为实际路径）
    test_image_dir = Path("data/train/img")
    if test_image_dir.exists():
        # 获取第一张图像进行测试
        image_files = list(test_image_dir.glob("*.jpg")) + list(test_image_dir.glob("*.png"))
        if image_files:
            test_image = image_files[0]
            print(f"测试图像: {test_image}")
            
            # 获取图像信息
            img_info = processor.get_image_info(test_image)
            print(f"图像信息: {img_info}")
            
            # 处理图像
            tensor = processor.process_single_image(test_image, transform_type='val')
            if tensor is not None:
                print(f"处理结果tensor形状: {tensor.shape}")
            
            # 提取特征
            image = processor.load_image(test_image)
            if image is not None:
                features = processor.extract_image_features(image)
                print(f"图像特征: {list(features.keys())}")
                print(f"亮度: {features.get('brightness', 0):.2f}")
                print(f"对比度: {features.get('contrast', 0):.2f}")
        else:
            print("未找到测试图像文件")
    else:
        print(f"图像目录不存在: {test_image_dir}")
    
    # 测试数据集处理
    print("\n🔄 === 数据集处理测试 ===")
    print("开始处理MR2数据集（这可能需要一些时间）...")
    
    try:
        # 处理数据集
        results = processor.process_mr2_dataset(splits=['train'], save_features=True)
        
        # 创建统计信息
        stats = processor.create_image_statistics(results)
        
        print("\n📊 === 处理统计 ===")
        print(f"总图像数: {stats['total_images']}")
        print(f"成功处理: {stats['successful_images']}")
        print(f"处理失败: {stats['failed_images']}")
        
        if 'avg_width' in stats:
            print(f"平均宽度: {stats['avg_width']:.1f}px")
            print(f"平均高度: {stats['avg_height']:.1f}px")
            print(f"平均文件大小: {stats['avg_file_size']/1024:.1f}KB")
        
        print(f"格式分布: {stats['format_distribution']}")
        
    except Exception as e:
        print(f"数据集处理失败: {e}")
        print("请确保数据目录和文件存在")
    
    print("\n✅ 图像处理模块测试完成")