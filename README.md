# Whisper Knowledge Distillation (Project Complete)

English README for the finished Whisper knowledge distillation project. Both Phase 1 (tiny teacher) and Phase 2 (small teacher) are fully implemented and exercised; code, configs, scripts, and result artifacts are checked in.

## Current Status
- ✅ Phase 1: whisper-tiny CE & KD baselines on a small subset; documented in `doc/PHASE1_COMPLETION.md`.
- ✅ Phase 2: whisper-small teacher with student decoders (2/3/4 layers), ~9.17 h filtered LibriSpeech data (5,366 clips ≤10s), end-to-end training/evaluation scripts ready. Experiments finished on this subset; full 100 h can be enabled by removing the `max_duration` cap in configs.
- 🚧 Phase 3 (DKD/quantization/pruning) is planned; no implementation required to ship the current project.

## Repository Layout
```
workspace-whisper/
├── configs/              # Data/teacher/student configs (CE, KD, DKD, sweeps)
├── src/
│   ├── data/             # Dataset + DataLoader
│   ├── models/           # Teacher/Student wrappers
│   ├── training/         # Training loops (CE / KD)
│   └── utils/            # Logging, metrics, config helpers
├── scripts/              # One-click training/eval/analysis
├── analysis/             # FLOPs/params/run tables
├── outputs/              # Generated stats & checkpoints
├── results/              # Evaluation tables/plots (generated)
├── doc/                  # Plans, guides, reports
└── requirements.txt      # Dependencies
```

## Environment
- Python 3.10+
- Single GPU (V100/A100/3090 or similar); ≥16 GB VRAM recommended.
- Install deps:
```bash
cd /home/zz359/workspace-whisper
pip install -r requirements.txt
```
- Quick sanity checks:
```bash
python test_setup.py            # Phase 1 smoke test
python test_phase2_setup.py     # Phase 2 file/config/script check
```

## Data & Configs
- Phase 2 uses LibriSpeech `train-clean-100`, filtered to ≤10 s by default (`configs/data_100h.yaml`, `data.max_duration=10.0`), producing ~9.17 h / 5,366 usable clips. Remove or relax `max_duration` to train on the full 100 h.
- Teacher: `configs/teacher_small.yaml` (openai/whisper-small, 244M params).
- Students: `configs/small_ce_s{2,3,4}.yaml` (CE) and `configs/small_kd_s{2,3,4}.yaml` (KD, T=4 by default). Temperature sweep configs live under `configs/temp/`.

## How to Run
### Phase 1 (tiny, minimal subset)
```bash
python -m src.training.train_ce --config configs/ce_tiny_s2.yaml
python -m src.training.train_kd --config configs/kd_tiny_s2.yaml
```

### Phase 2 (small, ~9 h subset)
```bash
# Data profiling (already generated once; re-run if you change filters)
python scripts/analyze_data_100h.py

# CE baselines
bash scripts/train_ce.sh configs/small_ce_s2.yaml
bash scripts/train_ce.sh configs/small_ce_s3.yaml
bash scripts/train_ce.sh configs/small_ce_s4.yaml

# KD baselines (T=4, α=0.5 by default)
bash scripts/train_kd.sh configs/small_kd_s2.yaml
bash scripts/train_kd.sh configs/small_kd_s3.yaml
bash scripts/train_kd.sh configs/small_kd_s4.yaml

# Temperature sweep for S3 (T=2/3/4/5/6/8)
bash scripts/train_temperature_sweep.sh

# Unified evaluation (WER/CER, params, FLOPs, speed)
python scripts/evaluate_all_models.py --num_samples 100
```
Artifacts go to `outputs/` (checkpoints/logs) and `results/` (tables/plots).

## Key Results (Phase 2, filtered 9 h set)
Best runs from `analysis/runs_table.md`:
| Model | Loss | T | WER (%) | CER (%) | Notes |
|-------|------|---|---------|---------|-------|
| S4 CE | CE-only | – | 16.27 | 6.90 | Best overall on subset |
| S3 CE | CE-only | – | 16.54 | 7.36 | Very close to S4 with fewer layers |
| S2 CE | CE-only | – | 22.74 | 11.66 | Smallest student, still reasonable |
| S4 KD | KD | 1.0 | 17.52 | 11.08 | KD underperformed CE on this subset |
| S3 KD | KD | 1.0 | 19.26 | 9.79 | KD slightly worse than CE |
| S2 KD | KD | 4.0 | 24.65 | 12.30 | KD did not beat CE |

Takeaways:
- On the 9 h filtered set, CE baselines outperform KD; KD likely needs more data or retuned α/T.
- Training/validation followed the provided configs; CUDA OOM cases were resolved by restarting with the final configs logged above.
- Teacher baseline metrics were not logged in the repo; re-run `python scripts/test_teacher_small.py --num_samples 100` if you need them.

## Documentation Map
- Quick start and structure: `doc/QUICKSTART.md`, `doc/STRUCTURE.md`
- Phase 1 report: `doc/PHASE1_COMPLETION.md`
- Phase 2 plan/guide/results: `doc/Project Plan Phase 2.md`, `doc/phase2_execution_guide.md`, `doc/phase2_results.md`, `PHASE2_IMPLEMENTATION_COMPLETE.md`, `PHASE2_PROGRESS.md`
- Roadmaps and summaries: `doc/PROJECT_ROADMAP.md`, `doc/PROJECT_SUMMARY.md`
- KD/DKD guides: `doc/KD_EXPERIMENTS_GUIDE.md`, `doc/DKD_EXPERIMENTS_GUIDE.md`, `doc/KD_vs_DKD_COMPARISON.md`

## Next Steps (optional)
- Remove `max_duration` or increase it to leverage the full 100 h and re-train KD/sweep runs.
- Re-tune KD (α, temperature) with more data; consider DKD configs under `configs/large_*` for Phase 3 exploration.
- Export and benchmark speed on realistic inference hardware; update `results/` with new plots/tables.

