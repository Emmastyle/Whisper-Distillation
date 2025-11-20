#!/usr/bin/env python3
"""
Phase 2 设置验证脚本
验证所有 Phase 2 配置文件和脚本是否正确创建
"""

import os
import sys
import yaml
from pathlib import Path


def test_config_files():
    """测试配置文件"""
    print("测试 Phase 2 配置文件...")
    
    required_configs = [
        "configs/data_100h.yaml",
        "configs/teacher_small.yaml",
        "configs/small_ce_s2.yaml",
        "configs/small_ce_s3.yaml",
        "configs/small_ce_s4.yaml",
        "configs/small_kd_s2.yaml",
        "configs/small_kd_s3.yaml",
        "configs/small_kd_s4.yaml",
    ]
    
    all_pass = True
    for config_path in required_configs:
        if not os.path.exists(config_path):
            print(f"  ✗ 配置文件不存在: {config_path}")
            all_pass = False
            continue
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # 验证基本结构
            if 'small' in config_path and config_path != 'configs/teacher_small.yaml':
                assert 'model' in config, f"缺少 model 配置"
                assert 'data' in config, f"缺少 data 配置"
                assert 'training' in config, f"缺少 training 配置"
                
                if 'kd' in config_path:
                    assert 'kd' in config, f"缺少 kd 配置"
            
            print(f"  ✓ {config_path}")
            
        except Exception as e:
            print(f"  ✗ 配置文件错误 {config_path}: {str(e)}")
            all_pass = False
    
    return all_pass


def test_scripts():
    """测试脚本文件"""
    print("\n测试 Phase 2 脚本...")
    
    required_scripts = [
        "scripts/analyze_data_100h.py",
        "scripts/test_teacher_small.py",
        "scripts/train_phase2_all.sh",
        "scripts/train_temperature_sweep.sh",
        "scripts/evaluate_all_models.py",
    ]
    
    all_pass = True
    for script_path in required_scripts:
        if not os.path.exists(script_path):
            print(f"  ✗ 脚本文件不存在: {script_path}")
            all_pass = False
            continue
        
        # 检查执行权限
        if os.access(script_path, os.X_OK):
            print(f"  ✓ {script_path} (可执行)")
        else:
            print(f"  ⚠ {script_path} (不可执行，可能需要 chmod +x)")
    
    return all_pass


def test_directory_structure():
    """测试目录结构"""
    print("\n测试目录结构...")
    
    required_dirs = [
        "configs",
        "scripts",
        "src",
        "doc",
        "outputs",
    ]
    
    all_pass = True
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ 目录不存在: {dir_path}/")
            all_pass = False
    
    return all_pass


def test_imports():
    """测试关键模块导入"""
    print("\n测试模块导入...")
    
    all_pass = True
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except Exception as e:
        print(f"  ✗ torch 导入失败: {str(e)}")
        all_pass = False
    
    try:
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except Exception as e:
        print(f"  ✗ transformers 导入失败: {str(e)}")
        all_pass = False
    
    try:
        import datasets
        print(f"  ✓ datasets {datasets.__version__}")
    except Exception as e:
        print(f"  ✗ datasets 导入失败: {str(e)}")
        all_pass = False
    
    try:
        import yaml
        print(f"  ✓ pyyaml")
    except Exception as e:
        print(f"  ✗ pyyaml 导入失败: {str(e)}")
        all_pass = False
    
    try:
        import pandas
        print(f"  ✓ pandas {pandas.__version__}")
    except Exception as e:
        print(f"  ✗ pandas 导入失败: {str(e)}")
        all_pass = False
    
    try:
        import matplotlib
        print(f"  ✓ matplotlib {matplotlib.__version__}")
    except Exception as e:
        print(f"  ✗ matplotlib 导入失败: {str(e)}")
        all_pass = False
    
    try:
        import seaborn
        print(f"  ✓ seaborn {seaborn.__version__}")
    except Exception as e:
        print(f"  ✗ seaborn 导入失败: {str(e)}")
        all_pass = False
    
    # 测试自定义模块
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from src.models.teacher import TeacherWhisperModel
        print(f"  ✓ src.models.teacher")
    except Exception as e:
        print(f"  ✗ src.models.teacher 导入失败: {str(e)}")
        all_pass = False
    
    try:
        from src.models.student_asr_model import StudentASRModel
        print(f"  ✓ src.models.student_asr_model")
    except Exception as e:
        print(f"  ✗ src.models.student_asr_model 导入失败: {str(e)}")
        all_pass = False
    
    try:
        from src.utils.metrics import calculate_wer, calculate_cer
        print(f"  ✓ src.utils.metrics")
    except Exception as e:
        print(f"  ✗ src.utils.metrics 导入失败: {str(e)}")
        all_pass = False
    
    return all_pass


def main():
    """主测试函数"""
    print("="*60)
    print("Phase 2 设置验证")
    print("="*60)
    print()
    
    results = {
        '配置文件': test_config_files(),
        '脚本文件': test_scripts(),
        '目录结构': test_directory_structure(),
        '模块导入': test_imports(),
    }
    
    print("\n" + "="*60)
    print("验证结果汇总")
    print("="*60)
    
    all_pass = True
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        all_pass = all_pass and result
    
    print("="*60)
    
    if all_pass:
        print("\n✓ Phase 2 设置验证通过！")
        print("\n下一步:")
        print("  1. 运行数据分析: python scripts/analyze_data_100h.py")
        print("  2. 验证 teacher model: python scripts/test_teacher_small.py")
        print("  3. 开始训练: bash scripts/train_phase2_all.sh")
        return 0
    else:
        print("\n✗ Phase 2 设置验证失败，请检查上述错误。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

