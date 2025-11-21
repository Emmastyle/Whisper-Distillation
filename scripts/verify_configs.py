#!/usr/bin/env python3
"""验证所有 KD 配置文件的正确性"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_yaml(file_path: Path) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def verify_config(config_path: Path) -> Dict[str, Any]:
    """验证单个配置文件"""
    config = load_yaml(config_path)
    
    issues = []
    warnings = []
    
    # 检查必需的顶级键
    required_keys = ['model', 'data', 'training', 'kd']
    for key in required_keys:
        if key not in config:
            issues.append(f"缺少必需的键: {key}")
    
    # 检查模型配置
    if 'model' in config:
        model = config['model']
        if 'teacher_model' not in model:
            issues.append("缺少 teacher_model")
        if 'student_decoder_layers' not in model:
            issues.append("缺少 student_decoder_layers")
        else:
            layers = model['student_decoder_layers']
            if layers not in [2, 3, 4]:
                warnings.append(f"student_decoder_layers={layers} 不在推荐范围 [2,3,4]")
    
    # 检查 KD 配置
    if 'kd' in config:
        kd = config['kd']
        if 'temperature' not in kd:
            issues.append("缺少 temperature")
        else:
            temp = kd['temperature']
            if temp <= 0:
                issues.append(f"temperature={temp} 必须 > 0")
            elif temp < 1:
                warnings.append(f"temperature={temp} < 1 可能导致数值不稳定")
            elif temp > 10:
                warnings.append(f"temperature={temp} > 10 可能过大")
        
        if 'alpha' not in kd:
            issues.append("缺少 alpha")
        else:
            alpha = kd['alpha']
            if not (0 <= alpha <= 1):
                issues.append(f"alpha={alpha} 必须在 [0, 1] 范围内")
    
    # 检查训练配置
    if 'training' in config:
        training = config['training']
        if 'output_dir' not in training:
            issues.append("缺少 output_dir")
        if 'learning_rate' not in training:
            issues.append("缺少 learning_rate")
        else:
            lr = training['learning_rate']
            if lr <= 0:
                issues.append(f"learning_rate={lr} 必须 > 0")
            elif lr > 1e-2:
                warnings.append(f"learning_rate={lr} 可能过大")
    
    # 检查数据配置
    if 'data' in config:
        data = config['data']
        if 'batch_size' not in data:
            issues.append("缺少 batch_size")
        else:
            bs = data['batch_size']
            if bs <= 0:
                issues.append(f"batch_size={bs} 必须 > 0")
    
    return {
        'config': config,
        'issues': issues,
        'warnings': warnings,
    }

def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "configs"
    
    # 查找所有 KD 配置文件
    config_files = sorted(config_dir.glob("small_kd_s*_t*_a01.yaml"))
    
    if not config_files:
        print("❌ 未找到配置文件")
        return 1
    
    print("=" * 80)
    print("KD 配置文件验证")
    print("=" * 80)
    print()
    
    all_valid = True
    results = []
    
    for config_file in config_files:
        print(f"📄 {config_file.name}")
        result = verify_config(config_file)
        results.append((config_file.name, result))
        
        config = result['config']
        issues = result['issues']
        warnings = result['warnings']
        
        # 显示关键参数
        if 'model' in config and 'kd' in config and 'training' in config:
            layers = config['model'].get('student_decoder_layers', 'N/A')
            temp = config['kd'].get('temperature', 'N/A')
            alpha = config['kd'].get('alpha', 'N/A')
            lr = config['training'].get('learning_rate', 'N/A')
            output_dir = config['training'].get('output_dir', 'N/A')
            
            print(f"   Layers: {layers} | Temp: {temp} | Alpha: {alpha} | LR: {lr}")
            print(f"   Output: {output_dir}")
        
        # 显示问题
        if issues:
            all_valid = False
            print(f"   ❌ 错误:")
            for issue in issues:
                print(f"      - {issue}")
        
        # 显示警告
        if warnings:
            print(f"   ⚠️  警告:")
            for warning in warnings:
                print(f"      - {warning}")
        
        if not issues and not warnings:
            print(f"   ✅ 配置正确")
        
        print()
    
    # 汇总
    print("=" * 80)
    print("汇总")
    print("=" * 80)
    print()
    
    print(f"总配置文件数: {len(config_files)}")
    valid_count = sum(1 for _, r in results if not r['issues'])
    print(f"有效配置: {valid_count}/{len(config_files)}")
    
    if not all_valid:
        print()
        print("❌ 部分配置文件存在错误，请修复后再运行训练")
        return 1
    
    print()
    print("=" * 80)
    print("配置矩阵")
    print("=" * 80)
    print()
    
    # 创建矩阵
    matrix = {}
    for config_file in config_files:
        result = verify_config(config_file)
        config = result['config']
        
        if 'model' in config and 'kd' in config:
            layers = config['model'].get('student_decoder_layers')
            temp = config['kd'].get('temperature')
            alpha = config['kd'].get('alpha')
            
            if layers and temp is not None:
                if layers not in matrix:
                    matrix[layers] = {}
                matrix[layers][temp] = {
                    'file': config_file.name,
                    'alpha': alpha,
                }
    
    # 打印矩阵
    print("| 模型 | T=1.0 | T=2.0 |")
    print("|------|-------|-------|")
    
    for layers in sorted(matrix.keys()):
        row = f"| S{layers}   |"
        for temp in [1.0, 2.0]:
            if temp in matrix[layers]:
                alpha = matrix[layers][temp]['alpha']
                row += f" α={alpha} |"
            else:
                row += "   -   |"
        print(row)
    
    print()
    print("=" * 80)
    print("✅ 所有配置文件验证通过！")
    print("=" * 80)
    print()
    print("下一步:")
    print("  bash scripts/train_kd_all_configs.sh")
    print()
    
    return 0

if __name__ == "__main__":
    exit(main())

