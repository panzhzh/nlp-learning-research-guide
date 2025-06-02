#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP项目文档统计脚本
用于统计项目中的文档数量、字数等信息，并生成JSON数据文件
"""

import os
import json
import re
import fnmatch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set

class DocumentStats:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
        # 要统计的文件扩展名
        self.include_extensions = {
            '.md', '.txt', '.py', '.js', '.html', '.css', 
            '.rst', '.tex', '.ipynb', '.json', '.yaml', '.yml'
        }
        
        # 从 .gitignore 文件读取排除规则
        self.gitignore_patterns = self.load_gitignore_patterns()
        
        # 统计结果
        self.stats = {
            'total_files': 0,
            'total_words': 0,
            'total_chars': 0,
            'total_lines': 0,
            'file_types': {},
            'largest_files': [],
            'update_time': '',
            'project_structure': {}
        }
    
    def load_gitignore_patterns(self) -> List[str]:
        """读取 .gitignore 文件并解析排除模式"""
        gitignore_path = self.project_root / '.gitignore'
        patterns = []
        
        if not gitignore_path.exists():
            print(f"警告: 未找到 .gitignore 文件，将使用默认排除规则")
            # 如果没有 .gitignore 文件，使用一些基本的默认规则
            default_patterns = [
                '__pycache__/',
                '*.pyc',
                '.git/',
                '.vscode/',
                '.idea/',
                'node_modules/',
                '*.log'
            ]
            return default_patterns
        
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释行
                    if not line or line.startswith('#'):
                        continue
                    patterns.append(line)
            
            print(f"已加载 {len(patterns)} 条 .gitignore 规则")
            return patterns
            
        except Exception as e:
            print(f"警告: 读取 .gitignore 文件失败: {e}")
            return []
    
    def should_exclude_path(self, file_path: Path) -> bool:
        """根据 .gitignore 规则判断是否应该排除某个路径"""
        # 获取相对于项目根目录的路径
        try:
            relative_path = file_path.relative_to(self.project_root)
        except ValueError:
            # 如果路径不在项目根目录下，排除它
            return True
        
        path_str = str(relative_path)
        path_parts = relative_path.parts
        
        # 检查每个 gitignore 模式
        for pattern in self.gitignore_patterns:
            if self.matches_gitignore_pattern(path_str, path_parts, pattern):
                return True
        
        return False
    
    def matches_gitignore_pattern(self, path_str: str, path_parts: tuple, pattern: str) -> bool:
        """检查路径是否匹配 gitignore 模式"""
        # 处理否定模式 (以 ! 开头)
        if pattern.startswith('!'):
            # 否定模式暂不处理，比较复杂
            return False
        
        # 处理以 / 结尾的目录模式
        if pattern.endswith('/'):
            pattern = pattern[:-1]
            # 检查是否有任何路径部分匹配这个目录名
            for part in path_parts[:-1]:  # 排除文件名本身
                if fnmatch.fnmatch(part, pattern):
                    return True
            return False
        
        # 处理以 / 开头的根目录模式
        if pattern.startswith('/'):
            pattern = pattern[1:]
            return fnmatch.fnmatch(path_str, pattern)
        
        # 处理包含 / 的路径模式
        if '/' in pattern:
            return fnmatch.fnmatch(path_str, pattern)
        
        # 处理文件名模式 - 检查路径的任何部分
        # 1. 检查完整路径
        if fnmatch.fnmatch(path_str, pattern):
            return True
        
        # 2. 检查文件名
        if fnmatch.fnmatch(path_parts[-1], pattern):
            return True
        
        # 3. 检查任何目录名
        for part in path_parts[:-1]:
            if fnmatch.fnmatch(part, pattern):
                return True
        
        # 4. 检查是否匹配路径中的任何段
        if '*' in pattern or '?' in pattern:
            for part in path_parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
        
        return False
    
    def count_words(self, text: str) -> int:
        """统计字数（中英文混合）"""
        # 中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        # 英文单词
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        # 数字
        numbers = len(re.findall(r'\d+', text))
        
        return chinese_chars + english_words + numbers
    
    def analyze_file(self, file_path: Path) -> Dict:
        """分析单个文件"""
        try:
            # 尝试不同的编码方式
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"警告: 无法解码文件 {file_path}")
                return None
            
            words = self.count_words(content)
            chars = len(content)
            lines = len(content.splitlines())
            
            file_info = {
                'path': str(file_path.relative_to(self.project_root)),
                'words': words,
                'chars': chars,
                'lines': lines,
                'size': file_path.stat().st_size,
                'extension': file_path.suffix.lower()
            }
            
            # 调试信息：打印前几个文件的统计
            if len(getattr(self, '_debug_count', [])) < 5:
                if not hasattr(self, '_debug_count'):
                    self._debug_count = []
                self._debug_count.append(file_info)
                print(f"调试: {file_info['path']} - {words}字, {chars}字符, {lines}行")
            
            return file_info
            
        except Exception as e:
            print(f"警告: 无法读取文件 {file_path}: {e}")
            return None
    
    def scan_project(self) -> None:
        """扫描整个项目"""
        print(f"正在扫描项目: {self.project_root.absolute()}")
        print(f"使用 {len(self.gitignore_patterns)} 条 .gitignore 规则")
        
        file_details = []
        total_scanned = 0
        excluded_count = 0
        
        for file_path in self.project_root.rglob('*'):
            if not file_path.is_file():
                continue
            
            total_scanned += 1
            
            # 检查是否应该排除
            if self.should_exclude_path(file_path):
                excluded_count += 1
                if total_scanned % 100 == 0:  # 每100个文件打印一次进度
                    print(f"已扫描 {total_scanned} 个文件，排除了 {excluded_count} 个")
                continue
                
            # 检查文件扩展名
            if file_path.suffix.lower() not in self.include_extensions:
                continue
            
            # 分析文件
            file_info = self.analyze_file(file_path)
            if file_info:
                file_details.append(file_info)
        
        print(f"扫描完成！总共扫描 {total_scanned} 个文件，排除 {excluded_count} 个")
        print(f"统计 {len(file_details)} 个符合条件的文档文件")
        
        # 统计汇总
        self.stats['total_files'] = len(file_details)
        self.stats['total_words'] = sum(f['words'] for f in file_details)
        self.stats['total_chars'] = sum(f['chars'] for f in file_details)
        self.stats['total_lines'] = sum(f['lines'] for f in file_details)
        
        # 调试信息
        print(f"\n=== 调试信息 ===")
        print(f"文件详情列表长度: {len(file_details)}")
        if file_details:
            print(f"第一个文件示例: {file_details[0]}")
            sample_words = [f['words'] for f in file_details[:5]]
            print(f"前5个文件字数: {sample_words}")
        
        # 按文件类型统计
        for file_info in file_details:
            ext = file_info['extension']
            if ext not in self.stats['file_types']:
                self.stats['file_types'][ext] = {
                    'count': 0,
                    'words': 0,
                    'chars': 0,
                    'lines': 0
                }
            
            self.stats['file_types'][ext]['count'] += 1
            self.stats['file_types'][ext]['words'] += file_info['words']
            self.stats['file_types'][ext]['chars'] += file_info['chars']
            self.stats['file_types'][ext]['lines'] += file_info['lines']
        
        # 最大的文件（按字数排序）
        self.stats['largest_files'] = sorted(
            file_details, 
            key=lambda x: x['words'], 
            reverse=True
        )[:10]
        
        # 更新时间
        self.stats['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n=== 最终统计结果 ===")
        print(f"总文件数: {self.stats['total_files']}")
        print(f"总字数: {self.stats['total_words']:,}")
        print(f"总字符数: {self.stats['total_chars']:,}")
        print(f"总行数: {self.stats['total_lines']:,}")
        
        # 显示文件类型分布
        if self.stats['file_types']:
            print(f"\n文件类型分布:")
            for ext, info in self.stats['file_types'].items():
                print(f"  {ext}: {info['count']}个文件, {info['words']:,}字")
        
        # 显示最大的几个文件
        if self.stats['largest_files']:
            print(f"\n字数最多的5个文件:")
            for i, file_info in enumerate(self.stats['largest_files'][:5], 1):
                print(f"  {i}. {file_info['path']}: {file_info['words']:,}字")
    
    def save_stats(self, output_file: str = "docs/stats.json") -> None:
        """保存统计结果到JSON文件"""
        output_path = self.project_root / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 调试：检查保存前的数据
        print(f"\n=== 保存调试信息 ===")
        print(f"准备保存的统计数据:")
        print(f"  total_files: {self.stats['total_files']}")
        print(f"  total_words: {self.stats['total_words']}")
        print(f"  total_chars: {self.stats['total_chars']}")
        print(f"  file_types数量: {len(self.stats['file_types'])}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        # 验证保存的文件
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            print(f"验证保存的数据:")
            print(f"  total_files: {saved_data.get('total_files', 'Missing')}")
            print(f"  total_words: {saved_data.get('total_words', 'Missing')}")
        except Exception as e:
            print(f"验证保存数据时出错: {e}")
        
        print(f"统计结果已保存到: {output_path}")
    
    def generate_markdown_report(self, output_file: str = "docs/stats.md") -> None:
        """生成Markdown格式的统计报告"""
        output_path = self.project_root / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        md_content = f"""# 📊 项目文档统计报告

> 最后更新时间: {self.stats['update_time']}

## 📈 总体统计

| 指标 | 数值 |
|------|------|
| 📁 文档文件总数 | {self.stats['total_files']:,} |
| 📝 总字数 | {self.stats['total_words']:,} |
| 📄 总字符数 | {self.stats['total_chars']:,} |
| 📋 总行数 | {self.stats['total_lines']:,} |
| 📊 平均字数/文件 | {self.stats['total_words'] // max(self.stats['total_files'], 1):,} |

## 📂 文件类型分布

| 文件类型 | 文件数 | 字数 | 字符数 | 行数 |
|----------|--------|------|--------|------|
"""
        
        for ext, info in sorted(self.stats['file_types'].items()):
            md_content += f"| {ext} | {info['count']} | {info['words']:,} | {info['chars']:,} | {info['lines']:,} |\n"
        
        md_content += f"""
## 📋 主要文档文件 (按字数排序)

| 文件路径 | 字数 | 字符数 | 行数 |
|----------|------|--------|------|
"""
        
        for file_info in self.stats['largest_files'][:15]:
            md_content += f"| {file_info['path']} | {file_info['words']:,} | {file_info['chars']:,} | {file_info['lines']:,} |\n"
        
        md_content += f"""
---
*此报告由项目文档统计脚本自动生成*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"Markdown报告已生成: {output_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NLP项目文档统计工具')
    parser.add_argument('--root', '-r', default='.', help='项目根目录路径')
    parser.add_argument('--output', '-o', default='docs/stats.json', help='输出JSON文件路径')
    parser.add_argument('--markdown', '-m', help='生成Markdown报告路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    # 创建统计器
    stats = DocumentStats(args.root)
    
    # 扫描项目
    stats.scan_project()
    
    # 保存JSON数据
    stats.save_stats(args.output)
    
    # 生成Markdown报告
    if args.markdown:
        stats.generate_markdown_report(args.markdown)
    else:
        stats.generate_markdown_report()  # 使用默认路径
    
    # 显示详细信息
    if args.verbose:
        print("\n=== 详细统计信息 ===")
        print(json.dumps(stats.stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()