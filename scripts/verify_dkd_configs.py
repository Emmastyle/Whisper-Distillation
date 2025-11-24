#!/usr/bin/env python3
"""验证所有 DKD 配置文件"""

import yaml
from pathlib import Path
from typing import Dict, Any

# 颜色代码
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


def print_success(msg: str):
    print(f"{GREEN}✓{NC} {msg}")


def print_error(msg: str):
    print(f"{RED}✗{NC} {msg}")


def print_warning(msg: str):
    print(f"{YELLOW}⚠{NC} {msg}")


def print_info(msg: str):
    print(f"{BLUE}ℹ{NC} {msg}")


def verify_config(config_path: Path) -> tuple[bool, list[str]]:
    """
    验证单个配置文件
    
    Returns:
        (is_valid, errors)
    """
    errors = []
    
    # 检查文件是否存在
    if not config_path.exists():
        errors.append(f"文件不存在: {config_path}")
        return False, errors
    
    # 加载 YAML
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"无法解析 YAML: {e}")
        return False, errors
    
    # 验证必需字段
    required_sections = ['model', 'data', 'training', 'dkd']
    for section in required_sections:
        if section not in config:
            errors.append(f"缺少必需部分: {section}")
    
    # 验证 model 部分
    if 'model' in config:
        model = config['model']
        
        if 'teacher_model' not in model:
            errors.append("model.teacher_model 未定义")
        
        if 'student_decoder_layers' not in model:
            errors.append("model.student_decoder_layers 未定义")
        elif not isinstance(model['student_decoder_layers'], int):
            errors.append("model.student_decoder_layers 必须是整数")
        elif model['student_decoder_layers'] not in [2, 3, 4]:
            errors.append(f"model.student_decoder_layers 应该是 2, 3 或 4，当前: {model['student_decoder_layers']}")
    
    # 验证 dkd 部分
    if 'dkd' in config:
        dkd = config['dkd']
        
        # Temperature
        if 'temperature' not in dkd:
            errors.append("dkd.temperature 未定义")
        elif not isinstance(dkd['temperature'], (int, float)):
            errors.append("dkd.temperature 必须是数字")
        elif dkd['temperature'] <= 0:
            errors.append(f"dkd.temperature 必须 > 0，当前: {dkd['temperature']}")
        elif dkd['temperature'] not in [1.0, 2.0]:
            print_warning(f"  dkd.temperature = {dkd['temperature']} (推荐: 1.0 或 2.0)")
        
        # Alpha (TCKD weight)
        if 'alpha' not in dkd:
            errors.append("dkd.alpha 未定义")
        elif not isinstance(dkd['alpha'], (int, float)):
            errors.append("dkd.alpha 必须是数字")
        elif dkd['alpha'] < 0:
            errors.append(f"dkd.alpha 必须 >= 0，当前: {dkd['alpha']}")
        
        # Beta (NCKD weight)
        if 'beta' not in dkd:
            errors.append("dkd.beta 未定义")
        elif not isinstance(dkd['beta'], (int, float)):
            errors.append("dkd.beta 必须是数字")
        elif dkd['beta'] < 0:
            errors.append(f"dkd.beta 必须 >= 0，当前: {dkd['beta']}")
        
        # Gamma (CE weight)
        if 'gamma' not in dkd:
            errors.append("dkd.gamma 未定义")
        elif not isinstance(dkd['gamma'], (int, float)):
            errors.append("dkd.gamma 必须是数字")
        elif not (0 <= dkd['gamma'] <= 1):
            errors.append(f"dkd.gamma 必须在 [0, 1]，当前: {dkd['gamma']}")
        
        # Top-k (optional, can be None)
        if 'topk' not in dkd:
            errors.append("dkd.topk 未定义")
        elif dkd['topk'] is not None:
            if not isinstance(dkd['topk'], int):
                errors.append("dkd.topk 必须是整数或 null")
            elif dkd['topk'] <= 0:
                errors.append(f"dkd.topk 必须 > 0，当前: {dkd['topk']}")
            elif dkd['topk'] > 1024:
                print_warning(f"  dkd.topk = {dkd['topk']} (通常建议使用 <= 1024)")
    
    # 验证 training 部分
    if 'training' in config:
        training = config['training']
        
        if 'learning_rate' not in training:
            errors.append("training.learning_rate 未定义")
        elif training['learning_rate'] <= 0:
            errors.append(f"training.learning_rate 必须 > 0，当前: {training['learning_rate']}")
        
        if 'output_dir' not in training:
            errors.append("training.output_dir 未定义")
    
    return len(errors) == 0, errors


def main():
    print("=" * 70)
    print("DKD 配置文件验证")
    print("=" * 70)
    print()
    
    # 项目根目录
    project_root = Path(__file__).parent.parent
    configs_dir = project_root / "configs"
    
    # 定义所有配置文件
    config_files = [
        "small_dkd_s2_t1_a08.yaml",
        "small_dkd_s2_t2_a08.yaml",
        "small_dkd_s3_t1_a08.yaml",
        "small_dkd_s3_t2_a08.yaml",
        "small_dkd_s4_t1_a08.yaml",
        "small_dkd_s4_t2_a08.yaml",
    ]
    
    print(f"配置目录: {configs_dir}")
    print(f"配置文件数: {len(config_files)}")
    print()
    
    # 验证每个配置
    results = {}
    all_valid = True
    
    for config_file in config_files:
        config_path = configs_dir / config_file
        print(f"验证: {config_file}")
        
        is_valid, errors = verify_config(config_path)
        results[config_file] = (is_valid, errors)
        
        if is_valid:
            print_success(f"  配置有效")
            
            # 显示关键参数
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"  • 模型: S{config['model']['student_decoder_layers']}")
            print(f"  • Temperature: {config['dkd']['temperature']}")
            print(f"  • Alpha (TCKD): {config['dkd']['alpha']}")
            print(f"  • Beta (NCKD): {config['dkd']['beta']}")
            print(f"  • Gamma (CE): {config['dkd']['gamma']}")
            print(f"  • Top-k (NCKD): {config['dkd'].get('topk', 'all')}")
            print(f"  • Learning Rate: {config['training']['learning_rate']}")
        else:
            print_error(f"  配置无效")
            for error in errors:
                print_error(f"    - {error}")
            all_valid = False
        
        print()
    
    # 总结
    print("=" * 70)
    if all_valid:
        print_success("所有配置文件验证通过！")
    else:
        print_error("部分配置文件验证失败")
    print("=" * 70)
    print()
    
    # 显示配置矩阵
    if all_valid:
        print("配置矩阵:")
        print()
        print("  | 模型 | T=1.0 | T=2.0 |")
        print("  |------|-------|-------|")
        
        for model_size in [2, 3, 4]:
            t1_config = f"small_dkd_s{model_size}_t1_a08.yaml"
            t2_config = f"small_dkd_s{model_size}_t2_a08.yaml"
            
            # 读取配置
            with open(configs_dir / t1_config, 'r') as f:
                t1 = yaml.safe_load(f)
            with open(configs_dir / t2_config, 'r') as f:
                t2 = yaml.safe_load(f)
            
            t1_params = (
                f"α={t1['dkd']['alpha']}, β={t1['dkd']['beta']}, "
                f"γ={t1['dkd']['gamma']}, k={t1['dkd'].get('topk', 'all')}"
            )
            t2_params = (
                f"α={t2['dkd']['alpha']}, β={t2['dkd']['beta']}, "
                f"γ={t2['dkd']['gamma']}, k={t2['dkd'].get('topk', 'all')}"
            )
            
            print(f"  | S{model_size}   | {t1_params} | {t2_params} |")
        
        print()
        print("参数说明:")
        print("  • α (Alpha): TCKD (Target Class KD) 权重")
        print("  • β (Beta): NCKD (Non-target Class KD) 权重")
        print("  • γ (Gamma): CE loss 权重 (DKD 权重 = 1 - γ)")
        print()
        print("Loss 公式:")
        print("  Total Loss = γ * CE + (1 - γ) * DKD")
        print("  DKD = α * TCKD + β * NCKD")
        print()
        
        print_info("准备开始训练:")
        print("  bash scripts/train_dkd_all_configs.sh")
        print()


if __name__ == "__main__":
    main()

