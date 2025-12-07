"""Compute approximate FLOPs for teacher / student ASR models.

We use fvcore.nn.FlopCountAnalysis to estimate floating point operations
for a single forward pass (encoder + decoder) with a fixed dummy input
shape. This gives **relative** compute between models; absolute numbers
can be re-scaled by changing the dummy input lengths below.
"""

import sys
import math
from pathlib import Path
from typing import List, Dict, Any

import torch
from fvcore.nn import FlopCountAnalysis


# Ensure project root is on sys.path so `src` can be imported when run from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.teacher import TeacherWhisper
from src.models.student_asr_model import StudentASRModel


def get_dummy_inputs(
    device: torch.device,
    n_mels: int = 80,
    n_frames: int = 3000,
    dec_len: int = 150,
) -> Dict[str, torch.Tensor]:
    """Create dummy Whisper-style inputs.

    - n_mels=80: Whisper 默认 log-Mel 维度
    - n_frames=3000: 近似 10s 音频（和当前实验的 max_duration 对齐）
    - dec_len=150: 近似一句话的 token 长度
    """
    input_features = torch.randn(1, n_mels, n_frames, device=device)
    # 用同一个 token id 占位即可，不影响 FLOPs 统计
    decoder_input_ids = torch.zeros(1, dec_len, dtype=torch.long, device=device)
    return {
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
    }


def compute_flops_for_module(
    module: torch.nn.Module,
    input_features: torch.Tensor,
    decoder_input_ids: torch.Tensor,
) -> int:
    """Return total FLOPs for a forward pass."""
    module.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(module, (input_features, decoder_input_ids)).total()
    return int(flops)


def human_gflops(flops: int) -> float:
    """Convert FLOPs to GFLOPs with 2 decimals."""
    return round(flops / 1e9, 2)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dummy = get_dummy_inputs(device)
    input_features = dummy["input_features"]
    decoder_input_ids = dummy["decoder_input_ids"]

    results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1) Teachers: whisper-small / whisper-large
    # ------------------------------------------------------------------
    print("Loading teachers...")
    teacher_small = TeacherWhisper("openai/whisper-small").to(device)
    teacher_large = TeacherWhisper("openai/whisper-large").to(device)

    print("Computing FLOPs for teachers...")
    flops_teacher_small = compute_flops_for_module(
        teacher_small, input_features, decoder_input_ids
    )
    results.append(
        {
            "Model": "teacher_small",
            "Type": "teacher",
            "Teacher": "openai/whisper-small",
            "Layers": "-",
            "Heads": "-",
            "GFLOPs": human_gflops(flops_teacher_small),
        }
    )

    flops_teacher_large = compute_flops_for_module(
        teacher_large, input_features, decoder_input_ids
    )
    results.append(
        {
            "Model": "teacher_large",
            "Type": "teacher",
            "Teacher": "openai/whisper-large",
            "Layers": "-",
            "Heads": "-",
            "GFLOPs": human_gflops(flops_teacher_large),
        }
    )

    # ------------------------------------------------------------------
    # 2) Students: S2 / S3 / S4 for small-teacher & large-teacher
    #
    # 注意：StudentASRModel 里 d_model 由 teacher.d_model 决定，
    # small-teacher 和 large-teacher 下的学生是两套不同结构。
    # ------------------------------------------------------------------
    student_layer_configs = [
        ("s2", 2),
        ("s3", 3),
        ("s4", 4),
    ]

    print("Computing FLOPs for students (small-teacher)...")
    for tag, n_layers in student_layer_configs:
        # small-teacher 学生：heads=12
        student_small = StudentASRModel(
            teacher=teacher_small,
            student_decoder_layers=n_layers,
            student_decoder_heads=12,
            dropout=0.1,
        ).to(device)
        flops = compute_flops_for_module(
            student_small, input_features, decoder_input_ids
        )
        results.append(
            {
                "Model": f"student_{tag}_smallT",
                "Type": "student",
                "Teacher": "openai/whisper-small",
                "Layers": n_layers,
                "Heads": 12,
                "GFLOPs": human_gflops(flops),
            }
        )

    print("Computing FLOPs for students (large-teacher)...")
    for tag, n_layers in student_layer_configs:
        # large-teacher 学生：heads=16
        student_large = StudentASRModel(
            teacher=teacher_large,
            student_decoder_layers=n_layers,
            student_decoder_heads=16,
            dropout=0.1,
        ).to(device)
        flops = compute_flops_for_module(
            student_large, input_features, decoder_input_ids
        )
        results.append(
            {
                "Model": f"student_{tag}_largeT",
                "Type": "student",
                "Teacher": "openai/whisper-large",
                "Layers": n_layers,
                "Heads": 16,
                "GFLOPs": human_gflops(flops),
            }
        )

    # ------------------------------------------------------------------
    # 打印表格，并保存到 CSV
    # ------------------------------------------------------------------
    headers = ["Model", "Type", "Teacher", "Layers", "Heads", "GFLOPs"]
    col_widths = {
        h: max(len(h), max(len(str(r[h])) for r in results)) for h in headers
    }

    def fmt_row(row: Dict[str, Any]) -> str:
        return " | ".join(
            str(row[h]).rjust(col_widths[h]) if h == "GFLOPs" else str(row[h]).ljust(col_widths[h])
            for h in headers
        )

    print("\nFLOPs table (single forward, batch_size=1, ~10s audio, ~150 tokens):\n")
    print(" | ".join(h.ljust(col_widths[h]) for h in headers))
    print("-+-".join("-" * col_widths[h] for h in headers))
    for row in results:
        print(fmt_row(row))

    # Save CSV
    csv_path = PROJECT_ROOT / "analysis" / "flops_table.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in results:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

    print(f"\nSaved FLOPs results to {csv_path}")


if __name__ == "__main__":
    main()


