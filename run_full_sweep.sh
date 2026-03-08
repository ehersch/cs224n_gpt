#!/bin/bash
# Comprehensive Hyperparameter Sweep for Paraphrase Detection
# Uses GPUs 0, 1, 3 ONLY (GPU 2 has degraded PCIe link - DO NOT USE)
# Runs multiple waves of experiments, each wave parallelizes across 3 GPUs
#
# Target: Beat staff baseline of 0.898 dev accuracy

set -e
cd "$(dirname "$0")"

# Force unbuffered Python output so logs are readable in real-time
export PYTHONUNBUFFERED=1

# Activate virtual environment
source .venv/bin/activate

# Create log/result directories
mkdir -p sweep_logs sweep_results

echo "=========================================="
echo "FULL HYPERPARAMETER SWEEP"
echo "Started: $(date)"
echo "Target: dev_acc > 0.898"
echo "GPUs: 0, 1, 3 (GPU 2 excluded - degraded PCIe)"
echo "=========================================="

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'{torch.cuda.device_count()} GPUs detected')"

###############################################################################
# WAVE 1: Extended training with best learning rates (highest priority)
###############################################################################
echo ""
echo "=== WAVE 1: Extended training (10 epochs) ==="
echo "Started: $(date)"

# GPU 0: Best config extended to 10 epochs (no regularization baseline)
echo "[GPU 0] lr=2e-5, 10 epochs, plain"
CUDA_VISIBLE_DEVICES=0 python hyperparam_sweep.py \
    --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 \
    --run_name w1_lr2e5_10ep \
    > sweep_logs/w1_lr2e5_10ep.log 2>&1 &
P0=$!

# GPU 1: Best LR + scheduler + weight decay (regularized)
echo "[GPU 1] lr=2e-5, 10 epochs, scheduler, wd=0.01"
CUDA_VISIBLE_DEVICES=1 python hyperparam_sweep.py \
    --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 \
    --use_scheduler --weight_decay 0.01 \
    --run_name w1_lr2e5_sched_wd \
    > sweep_logs/w1_lr2e5_sched_wd.log 2>&1 &
P1=$!

# GPU 3: Slower LR + regularization (may peak higher)
echo "[GPU 3] lr=1e-5, 10 epochs, scheduler, wd=0.01"
CUDA_VISIBLE_DEVICES=3 python hyperparam_sweep.py \
    --gpu 0 --lr 1e-5 --epochs 10 --batch_size 8 \
    --use_scheduler --weight_decay 0.01 \
    --run_name w1_lr1e5_sched_wd \
    > sweep_logs/w1_lr1e5_sched_wd.log 2>&1 &
P3=$!

echo "Waiting for Wave 1 (3 experiments)..."
wait $P0 $P1 $P3
echo "Wave 1 complete: $(date)"

###############################################################################
# WAVE 2: Refined learning rates + dropout experiments
###############################################################################
echo ""
echo "=== WAVE 2: Refined LR + Dropout ==="
echo "Started: $(date)"

# GPU 0: LR=1.5e-5 (between the two best), scheduler, wd
echo "[GPU 0] lr=1.5e-5, 10 epochs, scheduler, wd=0.01"
CUDA_VISIBLE_DEVICES=0 python hyperparam_sweep.py \
    --gpu 0 --lr 1.5e-5 --epochs 10 --batch_size 8 \
    --use_scheduler --weight_decay 0.01 \
    --run_name w2_lr1.5e5_sched_wd \
    > sweep_logs/w2_lr1.5e5_sched_wd.log 2>&1 &
P0=$!

# GPU 1: LR=3e-5 (slightly aggressive), scheduler, wd
echo "[GPU 1] lr=3e-5, 10 epochs, scheduler, wd=0.01"
CUDA_VISIBLE_DEVICES=1 python hyperparam_sweep.py \
    --gpu 0 --lr 3e-5 --epochs 10 --batch_size 8 \
    --use_scheduler --weight_decay 0.01 \
    --run_name w2_lr3e5_sched_wd \
    > sweep_logs/w2_lr3e5_sched_wd.log 2>&1 &
P1=$!

# GPU 3: Full regularization (dropout + wd + scheduler)
echo "[GPU 3] lr=2e-5, 10 epochs, dropout=0.1, scheduler, wd=0.01"
CUDA_VISIBLE_DEVICES=3 python hyperparam_sweep.py \
    --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 \
    --clf_dropout 0.1 --use_scheduler --weight_decay 0.01 \
    --run_name w2_lr2e5_drop0.1_sched_wd \
    > sweep_logs/w2_lr2e5_drop0.1_sched_wd.log 2>&1 &
P3=$!

echo "Waiting for Wave 2 (3 experiments)..."
wait $P0 $P1 $P3
echo "Wave 2 complete: $(date)"

###############################################################################
# WAVE 3: More dropout variants + longer training with slow LR
###############################################################################
echo ""
echo "=== WAVE 3: Dropout variants + slow LR extended ==="
echo "Started: $(date)"

# GPU 0: Higher dropout
echo "[GPU 0] lr=2e-5, 10 epochs, dropout=0.2, scheduler, wd=0.01"
CUDA_VISIBLE_DEVICES=0 python hyperparam_sweep.py \
    --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 \
    --clf_dropout 0.2 --use_scheduler --weight_decay 0.01 \
    --run_name w3_lr2e5_drop0.2_sched_wd \
    > sweep_logs/w3_lr2e5_drop0.2_sched_wd.log 2>&1 &
P0=$!

# GPU 1: Plain LR=2e-5 but 15 epochs (longer without regularization)
echo "[GPU 1] lr=2e-5, 15 epochs, plain"
CUDA_VISIBLE_DEVICES=1 python hyperparam_sweep.py \
    --gpu 0 --lr 2e-5 --epochs 15 --batch_size 8 \
    --run_name w3_lr2e5_15ep \
    > sweep_logs/w3_lr2e5_15ep.log 2>&1 &
P1=$!

# GPU 3: Different warmup ratio with best config
echo "[GPU 3] lr=2e-5, 10 epochs, scheduler (warmup=0.1), wd=0.01"
CUDA_VISIBLE_DEVICES=3 python hyperparam_sweep.py \
    --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 \
    --use_scheduler --warmup_ratio 0.1 --weight_decay 0.01 \
    --run_name w3_lr2e5_warm0.1_sched_wd \
    > sweep_logs/w3_lr2e5_warm0.1_sched_wd.log 2>&1 &
P3=$!

echo "Waiting for Wave 3 (3 experiments)..."
wait $P0 $P1 $P3
echo "Wave 3 complete: $(date)"

###############################################################################
# WAVE 4: GPT2-medium experiments (larger model, may get higher accuracy)
###############################################################################
echo ""
echo "=== WAVE 4: GPT2-medium ==="
echo "Started: $(date)"

# GPU 0: gpt2-medium, conservative LR
echo "[GPU 0] gpt2-medium, lr=1e-5, 5 epochs, bs=4, scheduler, wd=0.01"
CUDA_VISIBLE_DEVICES=0 python hyperparam_sweep.py \
    --gpu 0 --lr 1e-5 --epochs 5 --batch_size 4 \
    --model_size gpt2-medium --use_scheduler --weight_decay 0.01 \
    --run_name w4_medium_lr1e5_sched_wd \
    > sweep_logs/w4_medium_lr1e5_sched_wd.log 2>&1 &
P0=$!

# GPU 1: gpt2-medium, slightly higher LR
echo "[GPU 1] gpt2-medium, lr=2e-5, 5 epochs, bs=4, scheduler, wd=0.01"
CUDA_VISIBLE_DEVICES=1 python hyperparam_sweep.py \
    --gpu 0 --lr 2e-5 --epochs 5 --batch_size 4 \
    --model_size gpt2-medium --use_scheduler --weight_decay 0.01 \
    --run_name w4_medium_lr2e5_sched_wd \
    > sweep_logs/w4_medium_lr2e5_sched_wd.log 2>&1 &
P1=$!

# GPU 3: gpt2-medium, very conservative with dropout
echo "[GPU 3] gpt2-medium, lr=5e-6, 5 epochs, bs=4, dropout=0.1, scheduler, wd=0.01"
CUDA_VISIBLE_DEVICES=3 python hyperparam_sweep.py \
    --gpu 0 --lr 5e-6 --epochs 5 --batch_size 4 \
    --model_size gpt2-medium --clf_dropout 0.1 --use_scheduler --weight_decay 0.01 \
    --run_name w4_medium_lr5e6_drop0.1_sched_wd \
    > sweep_logs/w4_medium_lr5e6_drop0.1_sched_wd.log 2>&1 &
P3=$!

echo "Waiting for Wave 4 (3 experiments)..."
wait $P0 $P1 $P3
echo "Wave 4 complete: $(date)"

###############################################################################
# WAVE 5: Best configs with different seeds + gradient accumulation
###############################################################################
echo ""
echo "=== WAVE 5: Seed variation + grad accumulation ==="
echo "Started: $(date)"

# GPU 0: Best config with different seed
echo "[GPU 0] lr=2e-5, 10 epochs, scheduler, wd=0.01, seed=42"
CUDA_VISIBLE_DEVICES=0 python hyperparam_sweep.py \
    --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 \
    --use_scheduler --weight_decay 0.01 --seed 42 \
    --run_name w5_lr2e5_sched_wd_seed42 \
    > sweep_logs/w5_lr2e5_sched_wd_seed42.log 2>&1 &
P0=$!

# GPU 1: Effective batch 16 via gradient accumulation + scaled LR
echo "[GPU 1] lr=3e-5, 10 epochs, bs=8, accum=2 (eff=16), scheduler, wd=0.01"
CUDA_VISIBLE_DEVICES=1 python hyperparam_sweep.py \
    --gpu 0 --lr 3e-5 --epochs 10 --batch_size 8 \
    --grad_accum_steps 2 --use_scheduler --weight_decay 0.01 \
    --run_name w5_lr3e5_accum2_sched_wd \
    > sweep_logs/w5_lr3e5_accum2_sched_wd.log 2>&1 &
P1=$!

# GPU 3: Best config with different seed
echo "[GPU 3] lr=2e-5, 10 epochs, scheduler, wd=0.01, seed=1234"
CUDA_VISIBLE_DEVICES=3 python hyperparam_sweep.py \
    --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 \
    --use_scheduler --weight_decay 0.01 --seed 1234 \
    --run_name w5_lr2e5_sched_wd_seed1234 \
    > sweep_logs/w5_lr2e5_sched_wd_seed1234.log 2>&1 &
P3=$!

echo "Waiting for Wave 5 (3 experiments)..."
wait $P0 $P1 $P3
echo "Wave 5 complete: $(date)"

###############################################################################
# RESULTS SUMMARY
###############################################################################
echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Finished: $(date)"
echo "=========================================="
echo ""

python3 -c "
import json, glob

results = []
for f in sorted(glob.glob('sweep_results/*_results.json')):
    with open(f) as fp:
        d = json.load(fp)
        results.append(d)

results.sort(key=lambda x: x['best_dev_acc'], reverse=True)

print(f\"{'Run Name':<35} {'Dev Acc':<10} {'Dev F1':<10} {'LR':<10} {'BS':<5} {'Ep':<5} {'Model':<15} {'Sched':<6} {'WD':<8} {'Drop':<6}\")
print('-' * 120)
for r in results:
    sched = 'Y' if r.get('use_scheduler', False) else 'N'
    wd = r.get('weight_decay', 0)
    drop = r.get('clf_dropout', 0)
    print(f\"{r['run_name']:<35} {r['best_dev_acc']:<10.4f} {r['best_dev_f1']:<10.4f} {r['lr']:<10} {r['batch_size']:<5} {r['epochs']:<5} {r['model_size']:<15} {sched:<6} {wd:<8} {drop:<6}\")

print(f\"\nBest: {results[0]['run_name']} with dev_acc={results[0]['best_dev_acc']:.4f}\")
baseline = 0.898
best = results[0]['best_dev_acc']
if best > baseline:
    print(f'BEATS STAFF BASELINE ({baseline}) by {best - baseline:.4f}!')
else:
    print(f'Below staff baseline ({baseline}) by {baseline - best:.4f}')
"
