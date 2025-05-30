#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: ipanzhzh
# code/utils/config_manager.py

"""
配置管理器
统一管理项目配置，支持YAML配置文件读取和路径管理
自动检测运行环境，智能处理路径
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """配置管理器 - 智能路径处理"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录，如果为None则自动检测
        """
        # 自动检测项目根目录和code目录
        self.project_root, self.code_root = self._detect_project_structure()
        
        # 设置配置目录
        if config_dir is None:
            self.config_dir = self.code_root / "config"
        else:
            self.config_dir = Path(config_dir)
            
        print(f"🔧 Project root: {self.project_root}")
        print(f"🔧 Code root: {self.code_root}")
        print(f"🔧 Config dir: {self.config_dir}")
        
        self.configs = {}
        self._load_all_configs()
    
    def _detect_project_structure(self) -> tuple[Path, Path]:
        """
        智能检测项目结构
        
        Returns:
            (project_root, code_root) 路径元组
        """
        # 获取当前文件所在目录
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        
        # 情况1: 当前在 code/utils/ 下
        if current_dir.name == "utils" and current_dir.parent.name == "code":
            code_root = current_dir.parent
            project_root = code_root.parent
            return project_root, code_root
        
        # 情况2: 从项目根目录或其他位置运行，需要查找code目录
        # 向上查找，直到找到包含code目录的父目录
        search_dir = Path.cwd()
        
        for _ in range(5):  # 最多向上查找5层
            code_candidate = search_dir / "code"
            if code_candidate.exists() and code_candidate.is_dir():
                # 检查code目录下是否有预期的子目录
                expected_dirs = ["config", "utils", "data", "datasets"]
                if any((code_candidate / d).exists() for d in expected_dirs):
                    return search_dir, code_candidate
            
            parent = search_dir.parent
            if parent == search_dir:  # 已经到达根目录
                break
            search_dir = parent
        
        # 情况3: 如果找不到，假设当前目录就是项目根目录
        project_root = Path.cwd()
        code_root = project_root / "code"
        
        # 如果code目录不存在，创建它
        if not code_root.exists():
            print(f"⚠️  Code directory not found, creating: {code_root}")
            code_root.mkdir(exist_ok=True)
            
        return project_root, code_root
    
    def _load_all_configs(self):
        """加载所有配置文件"""
        config_files = {
            'data': 'data_configs.yaml',
            'model': 'model_configs.yaml', 
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
        """
        获取指定类型的配置
        
        Args:
            config_type: 配置类型 ('data', 'model', 'training', 'supported_models')
            
        Returns:
            配置字典
        """
        return self.configs.get(config_type, {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.get_config('data')
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.get_config('model')
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.get_config('training')
    
    def get_output_paths(self) -> Dict[str, str]:
        """获取输出路径配置"""
        data_config = self.get_data_config()
        output_structure = data_config.get('output', {}).get('structure', {})
        
        # 将相对路径转换为基于code目录的绝对路径
        resolved_paths = {}
        for category, paths in output_structure.items():
            if isinstance(paths, dict):
                resolved_paths[category] = {}
                for path_name, path_value in paths.items():
                    resolved_paths[category][path_name] = str(self.code_root / path_value)
            elif isinstance(paths, str):
                resolved_paths[category] = str(self.code_root / paths)
        
        return resolved_paths
    
    def get_dataset_paths(self) -> Dict[str, str]:
        """获取数据集路径配置"""
        data_config = self.get_data_config()
        dataset_paths = data_config.get('dataset', {}).get('paths', {})
        
        # 解析相对于code目录的路径
        resolved_paths = {}
        for key, value in dataset_paths.items():
            if isinstance(value, str):
                resolved_paths[key] = str(self.code_root / value)
            else:
                resolved_paths[key] = value
                
        return resolved_paths
    
    def get_label_mapping(self) -> Dict[int, str]:
        """获取标签映射"""
        data_config = self.get_data_config()
        labels = data_config.get('dataset', {}).get('labels', {})
        # 确保键是整数类型
        return {int(k): v for k, v in labels.items()}
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """获取分析配置"""
        data_config = self.get_data_config()
        return data_config.get('analysis', {})
    
    def create_output_directories(self):
        """创建输出目录结构"""
        output_paths = self.get_output_paths()
        
        created_dirs = []
        for category, paths in output_paths.items():
            if isinstance(paths, dict):
                for path_name, path_value in paths.items():
                    dir_path = Path(path_value)
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(dir_path))
            elif isinstance(paths, str):
                dir_path = Path(paths)
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
        
        print(f"✅ 创建输出目录: {len(created_dirs)} 个")
        return created_dirs
    
    def get_data_dir(self) -> Path:
        """获取数据目录路径"""
        dataset_paths = self.get_dataset_paths()
        base_dir = dataset_paths.get('base_dir', 'data')
        return Path(base_dir)
    
    def get_full_path(self, relative_path: str, base_type: str = 'data') -> Path:
        """
        获取完整路径
        
        Args:
            relative_path: 相对路径
            base_type: 基础路径类型 ('data', 'code', 'project')
            
        Returns:
            完整路径
        """
        if base_type == 'data':
            base_dir = self.get_data_dir()
        elif base_type == 'code':
            base_dir = self.code_root
        elif base_type == 'project':
            base_dir = self.project_root
        else:
            base_dir = Path('.')
            
        return base_dir / relative_path
    
    def get_output_path(self, module: str, subdir: str) -> Path:
        """
        获取特定模块的输出路径
        
        Args:
            module: 模块名 (如 'datasets', 'preprocessing')
            subdir: 子目录名 (如 'charts', 'reports')
            
        Returns:
            输出路径
        """
        output_paths = self.get_output_paths()
        if module in output_paths and subdir in output_paths[module]:
            return Path(output_paths[module][subdir])
        else:
            # 默认路径
            return self.code_root / 'outputs' / module / subdir
    
    def save_config(self, config_data: Dict[str, Any], config_type: str):
        """
        保存配置到文件
        
        Args:
            config_data: 配置数据
            config_type: 配置类型
        """
        config_files = {
            'data': 'data_configs.yaml',
            'model': 'model_configs.yaml',
            'training': 'training_configs.yaml',
            'supported_models': 'supported_models.yaml'
        }
        
        if config_type not in config_files:
            raise ValueError(f"不支持的配置类型: {config_type}")
        
        config_path = self.config_dir / config_files[config_type]
        
        # 确保配置目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 保存配置到: {config_path}")
    
    def update_config(self, config_type: str, updates: Dict[str, Any]):
        """
        更新配置
        
        Args:
            config_type: 配置类型
            updates: 更新的配置项
        """
        if config_type not in self.configs:
            self.configs[config_type] = {}
        
        def deep_update(base_dict, update_dict):
            """深度更新字典"""
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.configs[config_type], updates)
        self.save_config(self.configs[config_type], config_type)


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


def get_output_path(module: str, subdir: str) -> Path:
    """获取输出路径"""
    return get_config_manager().get_output_path(module, subdir)


def get_label_mapping():
    """获取标签映射"""
    return get_config_manager().get_label_mapping()


def create_output_directories():
    """创建输出目录"""
    return get_config_manager().create_output_directories()


def get_analysis_config():
    """获取分析配置"""
    return get_config_manager().get_analysis_config()


def get_data_dir():
    """获取数据目录"""
    return get_config_manager().get_data_dir()


# 使用示例
if __name__ == "__main__":
    # 测试配置管理器
    print("🔧 测试配置管理器")
    
    # 创建配置管理器
    config_mgr = ConfigManager()
    
    # 创建输出目录
    created_dirs = config_mgr.create_output_directories()
    print(f"创建目录数量: {len(created_dirs)}")
    
    # 测试配置获取
    data_config = config_mgr.get_data_config()
    print(f"数据集名称: {data_config.get('dataset', {}).get('name', 'Unknown')}")
    
    # 测试路径获取
    charts_path = config_mgr.get_output_path('datasets', 'charts')
    print(f"图表输出路径: {charts_path}")
    
    # 测试数据目录
    data_dir = config_mgr.get_data_dir()
    print(f"数据目录: {data_dir}")
    
    # 测试标签映射
    labels = config_mgr.get_label_mapping()
    print(f"标签映射: {labels}")
    
    print("✅ 配置管理器测试完成")