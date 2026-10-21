#!/usr/bin/env python3
"""
分析 outputs 目录下所有模型的参数大小和文件大小
"""

import os
import sys
import torch
from pathlib import Path
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


def count_parameters(state_dict):
    """统计模型参数数量"""
    total_params = 0
    trainable_params = 0
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            num_params = param.numel()
            total_params += num_params
            trainable_params += num_params
    
    return total_params, trainable_params


def get_file_size_mb(file_path):
    """获取文件大小（MB）"""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def analyze_models(outputs_dir):
    """分析所有模型"""
    results = []
    
    outputs_path = Path(outputs_dir)
    
    # 遍历所有子目录
    for model_dir in sorted(outputs_path.iterdir()):
        if not model_dir.is_dir():
            continue
        
        best_model_path = model_dir / "best_model.pt"
        
        if not best_model_path.exists():
            print(f"⚠️  跳过 {model_dir.name}: 没有找到 best_model.pt")
            continue
        
        try:
            # 加载模型 (weights_only=False 因为 checkpoint 包含自定义 config 类)
            checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            
            # 获取模型状态字典
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 统计参数
            total_params, trainable_params = count_parameters(state_dict)
            
            # 获取文件大小
            file_size_mb = get_file_size_mb(best_model_path)
            
            # 获取其他信息
            wer = checkpoint.get('wer', None)
            cer = checkpoint.get('cer', None)
            epoch = checkpoint.get('epoch', None)
            
            results.append({
                'Model Name': model_dir.name,
                'Total Parameters': total_params,
                'Parameters (M)': f"{total_params / 1e6:.2f}",
                'File Size (MB)': f"{file_size_mb:.2f}",
                'WER (%)': f"{wer:.2f}" if wer is not None else "N/A",
                'CER (%)': f"{cer:.2f}" if cer is not None else "N/A",
                'Epoch': epoch if epoch is not None else "N/A",
            })
            
            print(f"✓ {model_dir.name}: {total_params:,} params ({total_params/1e6:.2f}M), {file_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"❌ 加载 {model_dir.name} 失败: {e}")
            continue
    
    return results


def main():
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    
    print("=" * 80)
    print("开始分析模型...")
    print("=" * 80)
    print()
    
    results = analyze_models(outputs_dir)
    
    if not results:
        print("没有找到任何模型文件！")
        return
    
    print()
    print("=" * 80)
    print("分析完成！")
    print("=" * 80)
    print()
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 按模型名称排序
    df = df.sort_values('Model Name')
    
    # 打印表格
    print("模型参数统计表：")
    print()
    print(df.to_string(index=False))
    
    # 保存到 CSV
    csv_path = project_root / "analysis" / "model_sizes.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print()
    print(f"✓ 结果已保存到: {csv_path}")
    
    # 保存到 Markdown
    md_path = project_root / "analysis" / "model_sizes.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 模型参数统计\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"**总计**: {len(results)} 个模型\n")
    print(f"✓ 结果已保存到: {md_path}")
    
    # 统计信息
    print()
    print("=" * 80)
    print("统计摘要：")
    print("=" * 80)
    
    # 转换参数数量为数值
    df['Params_Numeric'] = df['Total Parameters'].astype(float)
    
    print(f"总模型数量: {len(results)}")
    print(f"平均参数量: {df['Params_Numeric'].mean() / 1e6:.2f}M")
    print(f"最小参数量: {df['Params_Numeric'].min() / 1e6:.2f}M ({df.loc[df['Params_Numeric'].idxmin(), 'Model Name']})")
    print(f"最大参数量: {df['Params_Numeric'].max() / 1e6:.2f}M ({df.loc[df['Params_Numeric'].idxmax(), 'Model Name']})")
    
    # 按照学生层数分组统计
    print()
    print("按学生层数分组：")
    for student_layers in ['s2', 's3', 's4']:
        filtered = df[df['Model Name'].str.contains(student_layers)]
        if len(filtered) > 0:
            avg_params = filtered['Params_Numeric'].mean() / 1e6
            print(f"  {student_layers.upper()}: {len(filtered)} 个模型, 平均 {avg_params:.2f}M 参数")


if __name__ == "__main__":
    main()
