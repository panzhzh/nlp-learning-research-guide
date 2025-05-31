#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# code/utils/config_manager.py

"""
简化的配置管理器
专注于核心功能，易于理解和维护
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """简化的配置管理器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """初始化配置管理器"""
        # 智能检测项目结构
        self.project_root, self.code_root = self._detect_project_structure()
        
        # 设置配置目录
        if config_dir is None:
            self.config_dir = self.code_root / "config"
        else:
            self.config_dir = Path(config_dir)
            
        print(f"🔧 Project root: {self.project_root}")
        print(f"🔧 Code root: {self.code_root}")
        print(f"🔧 Config dir: {self.config_dir}")
        
        # 加载配置
        self.configs = {}
        self._load_all_configs()
    
    def _detect_project_structure(self) -> tuple[Path, Path]:
        """智能检测项目结构"""
        # 获取当前文件所在目录
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        
        # 情况1: 当前在 code/utils/ 下
        if current_dir.name == "utils" and current_dir.parent.name == "code":
            code_root = current_dir.parent
            project_root = code_root.parent
            return project_root, code_root
        
        # 情况2: 从其他位置运行，查找code目录
        search_dir = Path.cwd()
        for _ in range(5):  # 最多向上查找5层
            code_candidate = search_dir / "code"
            if code_candidate.exists() and code_candidate.is_dir():
                # 检查是否有预期的子目录
                expected_dirs = ["config", "utils", "datasets"]
                if any((code_candidate / d).exists() for d in expected_dirs):
                    return search_dir, code_candidate
            
            parent = search_dir.parent
            if parent == search_dir:  # 已到根目录
                break
            search_dir = parent
        
        # 情况3: 默认假设当前目录就是项目根目录
        project_root = Path.cwd()
        code_root = project_root / "code"
        return project_root, code_root
    
    def _load_all_configs(self):
        """加载所有配置文件"""
        config_files = {
            'data': 'data_configs.yaml',
            'training': 'training_configs.yaml',
            'supported_models': 'supported_models.yaml'
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
        """获取训练配置 - 修复缺失的函数"""
        return self.get_config('training')
    
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
        """获取数据目录路径"""
        # 首先检查配置文件中的设置
        data_config = self.get_data_config()
        dataset_paths = data_config.get('dataset', {}).get('paths', {})
        base_dir = dataset_paths.get('base_dir', 'data')
        
        # 转换为绝对路径
        if not os.path.isabs(base_dir):
            data_dir = self.code_root / base_dir
        else:
            data_dir = Path(base_dir)
        
        return data_dir
    
    def get_output_path(self, module: str, subdir: str) -> Path:
        """获取输出路径并自动创建目录"""
        output_dir = self.code_root / 'outputs' / module / subdir
        output_dir.mkdir(parents=True, exist_ok=True)  # 自动创建目录
        return output_dir
    
    def create_output_directories(self):
        """创建输出目录结构"""
        output_dirs = [
            'outputs/datasets/charts',
            'outputs/datasets/reports',
            'outputs/datasets/analysis',
            'outputs/models',
            'outputs/logs'
        ]
        
        created_count = 0
        for dir_path in output_dirs:
            full_path = self.code_root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                created_count += 1
        
        print(f"✅ 创建输出目录: {created_count} 个")
        return created_count


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


# 测试代码
if __name__ == "__main__":
    print("🔧 测试配置管理器")
    
    # 创建配置管理器
    config_mgr = ConfigManager()
    
    # 创建输出目录
    created_dirs = config_mgr.create_output_directories()
    print(f"创建目录数量: {created_dirs}")
    
    # 测试各种配置获取
    print(f"数据目录: {config_mgr.get_data_dir()}")
    print(f"标签映射: {config_mgr.get_label_mapping()}")
    print(f"图表输出路径: {config_mgr.get_output_path('datasets', 'charts')}")
    
    print("✅ 配置管理器测试完成")