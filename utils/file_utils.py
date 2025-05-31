#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# utils/file_utils.py

"""
文件操作工具模块
提供统一的文件读写、格式转换、路径处理等功能
"""

import json
import csv
import pickle
import os
import shutil
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class FileUtils:
    """文件操作工具类"""
    
    @staticmethod
    def read_json(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        读取JSON文件
        
        Args:
            file_path: JSON文件路径
            encoding: 文件编码
            
        Returns:
            JSON数据字典
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON文件不存在: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON文件格式错误: {file_path}, 错误: {e}")
            raise
        except Exception as e:
            logger.error(f"读取JSON文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def write_json(data: Dict[str, Any], file_path: Union[str, Path], 
                   encoding: str = 'utf-8', indent: int = 2, ensure_ascii: bool = False) -> None:
        """
        写入JSON文件
        
        Args:
            data: 要写入的数据
            file_path: 输出文件路径
            encoding: 文件编码
            indent: 缩进空格数
            ensure_ascii: 是否确保ASCII编码
        """
        try:
            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding=encoding) as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            
            logger.info(f"成功写入JSON文件: {file_path}")
            
        except Exception as e:
            logger.error(f"写入JSON文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def read_yaml(file_path: Union[str, Path], encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        读取YAML文件
        
        Args:
            file_path: YAML文件路径
            encoding: 文件编码
            
        Returns:
            YAML数据字典
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"YAML文件不存在: {file_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML文件格式错误: {file_path}, 错误: {e}")
            raise
        except Exception as e:
            logger.error(f"读取YAML文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def write_yaml(data: Dict[str, Any], file_path: Union[str, Path], 
                   encoding: str = 'utf-8') -> None:
        """
        写入YAML文件
        
        Args:
            data: 要写入的数据
            file_path: 输出文件路径
            encoding: 文件编码
        """
        try:
            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding=encoding) as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"成功写入YAML文件: {file_path}")
            
        except Exception as e:
            logger.error(f"写入YAML文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def read_csv(file_path: Union[str, Path], encoding: str = 'utf-8', **kwargs) -> pd.DataFrame:
        """
        读取CSV文件
        
        Args:
            file_path: CSV文件路径
            encoding: 文件编码
            **kwargs: pandas.read_csv的其他参数
            
        Returns:
            DataFrame对象
        """
        try:
            return pd.read_csv(file_path, encoding=encoding, **kwargs)
        except FileNotFoundError:
            logger.error(f"CSV文件不存在: {file_path}")
            raise
        except Exception as e:
            logger.error(f"读取CSV文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def write_csv(df: pd.DataFrame, file_path: Union[str, Path], 
                  encoding: str = 'utf-8', index: bool = False, **kwargs) -> None:
        """
        写入CSV文件
        
        Args:
            df: DataFrame对象
            file_path: 输出文件路径
            encoding: 文件编码
            index: 是否包含索引
            **kwargs: pandas.to_csv的其他参数
        """
        try:
            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(file_path, encoding=encoding, index=index, **kwargs)
            logger.info(f"成功写入CSV文件: {file_path}")
            
        except Exception as e:
            logger.error(f"写入CSV文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def read_pickle(file_path: Union[str, Path]) -> Any:
        """
        读取Pickle文件
        
        Args:
            file_path: Pickle文件路径
            
        Returns:
            反序列化的对象
        """
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Pickle文件不存在: {file_path}")
            raise
        except Exception as e:
            logger.error(f"读取Pickle文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def write_pickle(obj: Any, file_path: Union[str, Path]) -> None:
        """
        写入Pickle文件
        
        Args:
            obj: 要序列化的对象
            file_path: 输出文件路径
        """
        try:
            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
            
            logger.info(f"成功写入Pickle文件: {file_path}")
            
        except Exception as e:
            logger.error(f"写入Pickle文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def read_text(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """
        读取文本文件
        
        Args:
            file_path: 文本文件路径
            encoding: 文件编码
            
        Returns:
            文件内容字符串
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"文本文件不存在: {file_path}")
            raise
        except Exception as e:
            logger.error(f"读取文本文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def write_text(text: str, file_path: Union[str, Path], 
                   encoding: str = 'utf-8', mode: str = 'w') -> None:
        """
        写入文本文件
        
        Args:
            text: 要写入的文本
            file_path: 输出文件路径
            encoding: 文件编码
            mode: 写入模式 ('w' 覆盖, 'a' 追加)
        """
        try:
            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, mode, encoding=encoding) as f:
                f.write(text)
            
            logger.info(f"成功写入文本文件: {file_path}")
            
        except Exception as e:
            logger.error(f"写入文本文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def read_lines(file_path: Union[str, Path], encoding: str = 'utf-8', 
                   strip: bool = True) -> List[str]:
        """
        按行读取文本文件
        
        Args:
            file_path: 文本文件路径
            encoding: 文件编码
            strip: 是否去除每行的首尾空白
            
        Returns:
            行列表
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
                return [line.strip() for line in lines] if strip else lines
        except FileNotFoundError:
            logger.error(f"文本文件不存在: {file_path}")
            raise
        except Exception as e:
            logger.error(f"按行读取文本文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def write_lines(lines: List[str], file_path: Union[str, Path], 
                    encoding: str = 'utf-8', add_newline: bool = True) -> None:
        """
        按行写入文本文件
        
        Args:
            lines: 行列表
            file_path: 输出文件路径
            encoding: 文件编码
            add_newline: 是否在每行末尾添加换行符
        """
        try:
            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding=encoding) as f:
                for line in lines:
                    if add_newline and not line.endswith('\n'):
                        line += '\n'
                    f.write(line)
            
            logger.info(f"成功按行写入文本文件: {file_path}")
            
        except Exception as e:
            logger.error(f"按行写入文本文件失败: {file_path}, 错误: {e}")
            raise


class PathUtils:
    """路径操作工具类"""
    
    @staticmethod
    def ensure_dir(dir_path: Union[str, Path]) -> Path:
        """
        确保目录存在，不存在则创建
        
        Args:
            dir_path: 目录路径
            
        Returns:
            Path对象
        """
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """
        获取文件大小（字节）
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件大小
        """
        try:
            return os.path.getsize(file_path)
        except FileNotFoundError:
            logger.error(f"文件不存在: {file_path}")
            raise
    
    @staticmethod
    def get_file_extension(file_path: Union[str, Path]) -> str:
        """
        获取文件扩展名
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件扩展名（包含点）
        """
        return Path(file_path).suffix
    
    @staticmethod
    def change_file_extension(file_path: Union[str, Path], new_ext: str) -> Path:
        """
        更改文件扩展名
        
        Args:
            file_path: 原文件路径
            new_ext: 新扩展名（可包含或不包含点）
            
        Returns:
            新的Path对象
        """
        path = Path(file_path)
        if not new_ext.startswith('.'):
            new_ext = '.' + new_ext
        return path.with_suffix(new_ext)
    
    @staticmethod
    def list_files(dir_path: Union[str, Path], pattern: str = "*", 
                   recursive: bool = False) -> List[Path]:
        """
        列出目录中的文件
        
        Args:
            dir_path: 目录路径
            pattern: 文件名模式（支持通配符）
            recursive: 是否递归搜索子目录
            
        Returns:
            文件路径列表
        """
        path = Path(dir_path)
        if not path.exists():
            logger.warning(f"目录不存在: {dir_path}")
            return []
        
        if recursive:
            return list(path.rglob(pattern))
        else:
            return list(path.glob(pattern))
    
    @staticmethod
    def copy_file(src: Union[str, Path], dst: Union[str, Path], 
                  create_dirs: bool = True) -> None:
        """
        复制文件
        
        Args:
            src: 源文件路径
            dst: 目标文件路径
            create_dirs: 是否自动创建目标目录
        """
        try:
            if create_dirs:
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src, dst)
            logger.info(f"成功复制文件: {src} -> {dst}")
            
        except Exception as e:
            logger.error(f"复制文件失败: {src} -> {dst}, 错误: {e}")
            raise
    
    @staticmethod
    def move_file(src: Union[str, Path], dst: Union[str, Path], 
                  create_dirs: bool = True) -> None:
        """
        移动文件
        
        Args:
            src: 源文件路径
            dst: 目标文件路径
            create_dirs: 是否自动创建目标目录
        """
        try:
            if create_dirs:
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(src), str(dst))
            logger.info(f"成功移动文件: {src} -> {dst}")
            
        except Exception as e:
            logger.error(f"移动文件失败: {src} -> {dst}, 错误: {e}")
            raise
    
    @staticmethod
    def delete_file(file_path: Union[str, Path], ignore_errors: bool = False) -> None:
        """
        删除文件
        
        Args:
            file_path: 文件路径
            ignore_errors: 是否忽略错误
        """
        try:
            Path(file_path).unlink()
            logger.info(f"成功删除文件: {file_path}")
            
        except FileNotFoundError:
            if not ignore_errors:
                logger.error(f"文件不存在: {file_path}")
                raise
        except Exception as e:
            if not ignore_errors:
                logger.error(f"删除文件失败: {file_path}, 错误: {e}")
                raise


class ImageUtils:
    """图像文件操作工具类"""
    
    @staticmethod
    def load_image(image_path: Union[str, Path], mode: str = 'RGB') -> Image.Image:
        """
        加载图像文件
        
        Args:
            image_path: 图像文件路径
            mode: 图像模式 ('RGB', 'RGBA', 'L' 等)
            
        Returns:
            PIL Image对象
        """
        try:
            image = Image.open(image_path)
            if mode and image.mode != mode:
                image = image.convert(mode)
            return image
        except FileNotFoundError:
            logger.error(f"图像文件不存在: {image_path}")
            raise
        except Exception as e:
            logger.error(f"加载图像失败: {image_path}, 错误: {e}")
            raise
    
    @staticmethod
    def save_image(image: Image.Image, save_path: Union[str, Path], 
                   quality: int = 95, **kwargs) -> None:
        """
        保存图像文件
        
        Args:
            image: PIL Image对象
            save_path: 保存路径
            quality: 图像质量 (1-100)
            **kwargs: 其他保存参数
        """
        try:
            # 确保目录存在
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            image.save(save_path, quality=quality, **kwargs)
            logger.info(f"成功保存图像: {save_path}")
            
        except Exception as e:
            logger.error(f"保存图像失败: {save_path}, 错误: {e}")
            raise
    
    @staticmethod
    def get_image_info(image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取图像基本信息
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像信息字典
        """
        try:
            with Image.open(image_path) as image:
                file_size = PathUtils.get_file_size(image_path)
                
                return {
                    'path': str(image_path),
                    'format': image.format,
                    'mode': image.mode,
                    'size': image.size,
                    'width': image.width,
                    'height': image.height,
                    'file_size': file_size,
                    'file_size_mb': round(file_size / (1024 * 1024), 2)
                }
        except Exception as e:
            logger.error(f"获取图像信息失败: {image_path}, 错误: {e}")
            raise
    
    @staticmethod
    def resize_image(image: Image.Image, size: tuple, 
                     resample: int = Image.Resampling.LANCZOS) -> Image.Image:
        """
        调整图像大小
        
        Args:
            image: PIL Image对象
            size: 目标尺寸 (width, height)
            resample: 重采样方法
            
        Returns:
            调整后的图像
        """
        try:
            return image.resize(size, resample)
        except Exception as e:
            logger.error(f"调整图像大小失败: {e}")
            raise
    
    @staticmethod
    def is_valid_image(file_path: Union[str, Path]) -> bool:
        """
        检查文件是否为有效图像
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为有效图像
        """
        try:
            with Image.open(file_path) as image:
                image.verify()
            return True
        except Exception:
            return False


class DatasetUtils:
    """数据集文件操作工具类"""
    
    @staticmethod
    def load_mr2_dataset(data_dir: Union[str, Path], split: str) -> Dict[str, Any]:
        """
        加载MR2数据集
        
        Args:
            data_dir: 数据目录
            split: 数据划分 ('train', 'val', 'test')
            
        Returns:
            数据集字典
        """
        dataset_file = Path(data_dir) / f'dataset_items_{split}.json'
        return FileUtils.read_json(dataset_file)
    
    @staticmethod
    def save_processed_features(features: Any, data_dir: Union[str, Path], 
                               split: str, feature_type: str = 'features') -> None:
        """
        保存处理后的特征
        
        Args:
            features: 特征数据
            data_dir: 数据目录
            split: 数据划分
            feature_type: 特征类型名
        """
        processed_dir = Path(data_dir) / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        feature_file = processed_dir / f'{split}_{feature_type}.pkl'
        FileUtils.write_pickle(features, feature_file)
    
    @staticmethod
    def load_processed_features(data_dir: Union[str, Path], split: str, 
                               feature_type: str = 'features') -> Any:
        """
        加载处理后的特征
        
        Args:
            data_dir: 数据目录
            split: 数据划分
            feature_type: 特征类型名
            
        Returns:
            特征数据
        """
        processed_dir = Path(data_dir) / 'processed'
        feature_file = processed_dir / f'{split}_{feature_type}.pkl'
        
        if not feature_file.exists():
            raise FileNotFoundError(f"处理后的特征文件不存在: {feature_file}")
        
        return FileUtils.read_pickle(feature_file)
    
    @staticmethod
    def get_annotation_file(data_dir: Union[str, Path], split: str, 
                           item_id: str, annotation_type: str) -> Path:
        """
        获取标注文件路径
        
        Args:
            data_dir: 数据目录
            split: 数据划分
            item_id: 数据项ID
            annotation_type: 标注类型 ('direct', 'inverse')
            
        Returns:
            标注文件路径
        """
        if annotation_type == 'direct':
            return Path(data_dir) / split / 'img_html_news' / item_id / 'direct_annotation.json'
        elif annotation_type == 'inverse':
            return Path(data_dir) / split / 'inverse_search' / item_id / 'inverse_annotation.json'
        else:
            raise ValueError(f"不支持的标注类型: {annotation_type}")
    
    @staticmethod
    def batch_process_files(file_list: List[Union[str, Path]], 
                           process_func: callable, 
                           output_dir: Optional[Union[str, Path]] = None,
                           **kwargs) -> List[Any]:
        """
        批量处理文件
        
        Args:
            file_list: 文件路径列表
            process_func: 处理函数
            output_dir: 输出目录（可选）
            **kwargs: 处理函数的额外参数
            
        Returns:
            处理结果列表
        """
        results = []
        
        for file_path in file_list:
            try:
                result = process_func(file_path, **kwargs)
                
                # 如果指定了输出目录，保存结果
                if output_dir and result is not None:
                    output_path = Path(output_dir) / f"{Path(file_path).stem}_processed.pkl"
                    FileUtils.write_pickle(result, output_path)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"处理文件失败: {file_path}, 错误: {e}")
                results.append(None)
        
        return results


# 便捷函数
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    根据文件扩展名自动选择加载方法
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置数据字典
    """
    ext = PathUtils.get_file_extension(config_path).lower()
    
    if ext in ['.json']:
        return FileUtils.read_json(config_path)
    elif ext in ['.yaml', '.yml']:
        return FileUtils.read_yaml(config_path)
    else:
        raise ValueError(f"不支持的配置文件格式: {ext}")


def save_results(results: Dict[str, Any], output_path: Union[str, Path], 
                format: str = 'auto') -> None:
    """
    保存实验结果
    
    Args:
        results: 结果数据
        output_path: 输出路径
        format: 保存格式 ('json', 'yaml', 'pickle', 'auto')
    """
    if format == 'auto':
        ext = PathUtils.get_file_extension(output_path).lower()
        if ext in ['.json']:
            format = 'json'
        elif ext in ['.yaml', '.yml']:
            format = 'yaml'
        elif ext in ['.pkl', '.pickle']:
            format = 'pickle'
        else:
            format = 'json'  # 默认
    
    if format == 'json':
        FileUtils.write_json(results, output_path)
    elif format == 'yaml':
        FileUtils.write_yaml(results, output_path)
    elif format == 'pickle':
        FileUtils.write_pickle(results, output_path)
    else:
        raise ValueError(f"不支持的保存格式: {format}")


# 使用示例
if __name__ == "__main__":
    # 测试文件操作功能
    print("🧪 测试文件操作工具")
    
    # 测试JSON读写
    test_data = {"name": "测试", "value": 123, "list": [1, 2, 3]}
    FileUtils.write_json(test_data, "test_output.json")
    loaded_data = FileUtils.read_json("test_output.json")
    print(f"JSON测试: {loaded_data}")
    
    # 测试路径操作
    test_dir = PathUtils.ensure_dir("test_dir/subdir")
    print(f"创建目录: {test_dir}")
    
    # 测试图像信息获取（如果有图像文件）
    try:
        # 这里需要实际的图像文件路径
        # image_info = ImageUtils.get_image_info("path/to/image.jpg")
        # print(f"图像信息: {image_info}")
        pass
    except:
        print("跳过图像测试（无图像文件）")
    
    print("✅ 文件操作工具测试完成")