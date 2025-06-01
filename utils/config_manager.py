#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# utils/config_manager.py

"""
使用项目根目录作为基础路径
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import platform


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """初始化配置管理器"""
        # 直接使用当前项目目录作为根目录
        self.project_root = Path.cwd()
        
        # 设置配置目录 - 直接在项目根目录下
        if config_dir is None:
            self.config_dir = self.project_root / "config"
        else:
            self.config_dir = Path(config_dir)
            
        print(f"🔧 运行环境: {platform.system()}")
        print(f"🔧 Project root: {self.project_root}")
        print(f"🔧 Config dir: {self.config_dir}")
        
        # 加载配置
        self.configs = {}
        self._load_all_configs()
        
        # 验证数据目录
        self._validate_data_directory()
    
    def _validate_data_directory(self):
        """验证数据目录是否存在"""
        data_dir = self.get_data_dir()
        
        if not data_dir.exists():
            raise FileNotFoundError(
                f"❌ 数据目录不存在: {data_dir}\n"
                f"请确保MR2数据集已下载并解压到正确位置。\n"
                f"期望的数据目录结构:\n"
                f"{data_dir}/\n"
                f"├── dataset_items_train.json\n"
                f"├── dataset_items_val.json\n"
                f"├── dataset_items_test.json\n"
                f"└── train/\n"
                f"    └── img/\n"
                f"下载链接: https://pan.baidu.com/s/1sfUwsaeV2nfl54OkrfrKVw?pwd=jxhc"
            )
        
        # 检查必要的数据文件
        required_files = [
            "dataset_items_train.json",
            "dataset_items_val.json", 
            "dataset_items_test.json"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = data_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(
                f"❌ 缺少必要的数据文件: {missing_files}\n"
                f"数据目录: {data_dir}\n"
                f"请确保所有数据文件都已正确解压到数据目录中。"
            )
        
        print(f"✅ 数据目录验证通过: {data_dir}")
    
    def _load_all_configs(self):
        """加载所有配置文件"""
        config_files = {
            'data': 'data_configs.yaml',
            'training': 'training_configs.yaml',
            'supported_models': 'supported_models.yaml',
            'model': 'model_configs.yaml'
        }
        
        for config_name, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self.configs[config_name] = yaml.safe_load(f)
                    print(f"✅ 加载配置文件: {config_path}")
                except Exception as e:
                    print(f"⚠️  加载配置文件失败 {config_path}: {e}")
                    self.configs[config_name] = {}
            else:
                print(f"⚠️  配置文件不存在: {config_path}")
                self.configs[config_name] = {}
    
    def get_config(self, config_type: str) -> Dict[str, Any]:
        """获取指定类型的配置"""
        return self.configs.get(config_type, {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.get_config('data')
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.get_config('training')
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.get_config('model')
    
    def get_label_mapping(self) -> Dict[int, str]:
        """获取标签映射"""
        data_config = self.get_data_config()
        labels = data_config.get('dataset', {}).get('labels', {
            0: 'Non-rumor', 1: 'Rumor', 2: 'Unverified'
        })
        return {int(k): v for k, v in labels.items()}
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """获取分析配置"""
        data_config = self.get_data_config()
        return data_config.get('analysis', {
            'visualization': {
                'colors': {
                    'primary': '#FF6B6B',
                    'secondary': '#4ECDC4', 
                    'tertiary': '#45B7D1'
                }
            }
        })
    
    def get_data_dir(self) -> Path:
        """
        获取数据目录路径 - 简化版本
        """
        data_config = self.get_data_config()
        dataset_paths = data_config.get('dataset', {}).get('paths', {})
        base_dir = dataset_paths.get('base_dir', 'auto_detect')
        
        if base_dir == 'auto_detect':
            # 直接在项目根目录下查找data目录
            data_dir = self.project_root / 'data'
            if data_dir.exists():
                print(f"🔍 找到数据目录: {data_dir}")
                return data_dir
            else:
                # 如果不存在，返回默认路径 (会在验证时报错)
                return data_dir
        else:
            # 使用配置中指定的路径
            if os.path.isabs(base_dir):
                return Path(base_dir)
            else:
                return self.project_root / base_dir
    
    def get_output_path(self, module: str, subdir: str) -> Path:
        """获取输出路径并自动创建目录"""
        output_dir = self.project_root / 'outputs' / module / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def create_output_directories(self):
        """创建输出目录结构"""
        output_dirs = [
            'outputs/data_utils/charts',
            'outputs/data_utils/reports',
            'outputs/data_utils/analysis',
            'outputs/models',
            'outputs/logs'
        ]
        
        created_count = 0
        for dir_path in output_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                created_count += 1
        
        print(f"✅ 创建输出目录: {created_count} 个")
        return created_count
    
    def check_data_requirements(self) -> bool:
        """检查数据要求是否满足"""
        data_config = self.get_data_config()
        requirements = data_config.get('dataset', {}).get('requirements', {})
        
        enforce_real_data = requirements.get('enforce_real_data', True)
        min_samples = requirements.get('min_samples_per_split', 10)
        
        if enforce_real_data:
            # 强制检查真实数据集
            data_dir = self.get_data_dir()
            splits = ['train', 'val', 'test']
            
            for split in splits:
                file_path = data_dir / f'dataset_items_{split}.json'
                if not file_path.exists():
                    raise FileNotFoundError(f"❌ 必需的数据文件不存在: {file_path}")
                
                # 检查文件是否有足够的样本
                try:
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if len(data) < min_samples:
                        raise ValueError(f"❌ {split} 数据集样本数不足: {len(data)} < {min_samples}")
                    
                    print(f"✅ {split} 数据集验证通过: {len(data)} 样本")
                    
                except Exception as e:
                    raise ValueError(f"❌ 验证 {split} 数据集失败: {e}")
        
        return True
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """将相对路径转换为绝对路径"""
        if os.path.isabs(relative_path):
            return Path(relative_path)
        else:
            return self.project_root / relative_path


# 全局配置管理器实例
_config_manager = None

def get_config_manager():
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

# 便捷函数
def get_data_config():
    """获取数据配置"""
    return get_config_manager().get_data_config()

def get_training_config():
    """获取训练配置"""
    return get_config_manager().get_training_config()

def get_model_config():
    """获取模型配置"""
    return get_config_manager().get_model_config()

def get_output_path(module: str, subdir: str) -> Path:
    """获取输出路径"""
    return get_config_manager().get_output_path(module, subdir)

def get_label_mapping():
    """获取标签映射"""
    return get_config_manager().get_label_mapping()

def get_analysis_config():
    """获取分析配置"""
    return get_config_manager().get_analysis_config()

def get_data_dir():
    """获取数据目录"""
    return get_config_manager().get_data_dir()

def create_output_directories():
    """创建输出目录"""
    return get_config_manager().create_output_directories()

def check_data_requirements():
    """检查数据要求"""
    return get_config_manager().check_data_requirements()


# 测试代码
if __name__ == "__main__":
    print("🔧 测试简化配置管理器")
    
    try:
        # 创建配置管理器
        config_mgr = ConfigManager()
        
        # 检查数据要求
        config_mgr.check_data_requirements()
        
        # 创建输出目录
        created_dirs = config_mgr.create_output_directories()
        print(f"创建目录数量: {created_dirs}")
        
        # 测试各种配置获取
        print(f"数据目录: {config_mgr.get_data_dir()}")
        print(f"标签映射: {config_mgr.get_label_mapping()}")
        print(f"图表输出路径: {config_mgr.get_output_path('data_utils', 'charts')}")
        
        print("✅ 简化配置管理器测试完成")
        
    except Exception as e:
        print(f"❌ 配置管理器测试失败: {e}")
        print("请检查数据集是否已正确下载和解压")