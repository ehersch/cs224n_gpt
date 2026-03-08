#!/bin/bash
# Resume hyperparameter sweep after GPU recovery.
# Run: bash resume_sweep.sh
#
# Prerequisites: GPU driver must be reset first.
#   sudo nvidia-smi --gpu-reset  (or reboot)
#
# This script runs the remaining experiments that couldn't complete
# due to the GPU 2 failure that corrupted the CUDA driver state.

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

echo "=== Checking GPU availability ==="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'{torch.cuda.device_count()} GPUs ready')"

echo ""
echo "=== Phase 1: Extended training with best LR (2e-5), 10 epochs ==="
echo "Running on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python hyperparam_sweep.py --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 --run_name best_lr2e5_10ep > sweep_logs/best_lr2e5_10ep.log 2>&1 &
P1=$!

echo "=== Phase 1b: LR=1e-5, 10 epochs on GPU 1 ==="
CUDA_VISIBLE_DEVICES=1 python hyperparam_sweep.py --gpu 0 --lr 1e-5 --epochs 10 --batch_size 8 --run_name best_lr1e5_10ep > sweep_logs/best_lr1e5_10ep.log 2>&1 &
P2=$!

echo "=== Phase 1c: LR=1.5e-5, 5 epochs on GPU 3 ==="
CUDA_VISIBLE_DEVICES=3 python hyperparam_sweep.py --gpu 0 --lr 1.5e-5 --epochs 5 --batch_size 8 --run_name r2_lr1.5e5 > sweep_logs/r2_lr1.5e5.log 2>&1 &
P3=$!

echo "Waiting for Phase 1..."
wait $P1 $P2 $P3
echo "Phase 1 complete!"

echo ""
echo "=== Phase 2: Weight decay + scheduler experiments ==="
CUDA_VISIBLE_DEVICES=0 python hyperparam_sweep.py --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 --weight_decay 0.01 --run_name wd_0.01_lr2e5 > sweep_logs/wd_0.01_lr2e5.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 python hyperparam_sweep.py --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 --use_scheduler --run_name sched_lr2e5 > sweep_logs/sched_lr2e5.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=3 python hyperparam_sweep.py --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 --weight_decay 0.01 --use_scheduler --run_name wd_sched_lr2e5 > sweep_logs/wd_sched_lr2e5.log 2>&1 &
P3=$!

wait $P1 $P2 $P3
echo "Phase 2 complete!"

echo ""
echo "=== Phase 3: Dropout experiments ==="
CUDA_VISIBLE_DEVICES=0 python hyperparam_sweep.py --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 --clf_dropout 0.1 --run_name drop_0.1_lr2e5 > sweep_logs/drop_0.1_lr2e5.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 python hyperparam_sweep.py --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 --clf_dropout 0.2 --run_name drop_0.2_lr2e5 > sweep_logs/drop_0.2_lr2e5.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=3 python hyperparam_sweep.py --gpu 0 --lr 2e-5 --epochs 10 --batch_size 8 --clf_dropout 0.1 --weight_decay 0.01 --use_scheduler --run_name drop_wd_sched_lr2e5 > sweep_logs/drop_wd_sched_lr2e5.log 2>&1 &
P3=$!

wait $P1 $P2 $P3
echo "Phase 3 complete!"

echo ""
echo "=== Phase 4: gpt2-medium experiments ==="
CUDA_VISIBLE_DEVICES=0 python hyperparam_sweep.py --gpu 0 --lr 1e-5 --epochs 5 --batch_size 4 --model_size gpt2-medium --run_name medium_lr1e5 > sweep_logs/medium_lr1e5.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 python hyperparam_sweep.py --gpu 0 --lr 2e-5 --epochs 5 --batch_size 4 --model_size gpt2-medium --run_name medium_lr2e5 > sweep_logs/medium_lr2e5.log 2>&1 &
P2=$!

wait $P1 $P2
echo "Phase 4 complete!"

echo ""
echo "=== All experiments done! ==="
echo "Results summary:"
python -c "
import json, glob, os
results = []
for f in sorted(glob.glob('sweep_results/*_results.json')):
    with open(f) as fp:
        d = json.load(fp)
        results.append(d)
results.sort(key=lambda x: x['best_dev_acc'], reverse=True)
print(f'{'Run Name':<30} {'Dev Acc':<10} {'Dev F1':<10} {'LR':<10} {'BS':<5} {'Epochs':<7} {'Model':<15}')
print('-'*87)
for r in results:
    print(f'{r[\"run_name\"]:<30} {r[\"best_dev_acc\"]:<10.4f} {r[\"best_dev_f1\"]:<10.4f} {r[\"lr\"]:<10} {r[\"batch_size\"]:<5} {r[\"epochs\"]:<7} {r[\"model_size\"]:<15}')
print(f'\nBest: {results[0][\"run_name\"]} with dev_acc={results[0][\"best_dev_acc\"]:.4f}')
"
