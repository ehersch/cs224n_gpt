"""Check sweep progress by inspecting log files and checkpoints."""
import os
import json
import glob
import re
from datetime import datetime

def parse_log(log_path):
    """Parse a tqdm-style log file to extract epoch and progress info."""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'rb') as f:
        data = f.read()

    # Replace \r with \n to handle tqdm output
    text = data.replace(b'\r', b'\n').decode('utf-8', errors='replace')
    lines = text.split('\n')

    # Find epoch result lines (printed by our training code)
    epoch_results = []
    for line in lines:
        # Match: [GPU0] Epoch 0: loss=0.3704, dev_acc=0.8725, dev_f1=0.8656
        m = re.search(r'Epoch (\d+): loss=([\d.]+), dev_acc=([\d.]+), dev_f1=([\d.]+)', line)
        if m:
            epoch_results.append({
                'epoch': int(m.group(1)),
                'loss': float(m.group(2)),
                'dev_acc': float(m.group(3)),
                'dev_f1': float(m.group(4)),
            })
        # Match: [GPU0] New best: dev_acc=0.8725
        m2 = re.search(r'New best: dev_acc=([\d.]+)', line)
        if m2:
            pass  # captured in epoch_results already

    # Find latest tqdm progress
    current_epoch = None
    current_pct = None
    for line in reversed(lines):
        m = re.search(r'train-(\d+):\s+(\d+)%', line)
        if m:
            current_epoch = int(m.group(1))
            current_pct = int(m.group(2))
            break

    return {
        'epoch_results': epoch_results,
        'current_epoch': current_epoch,
        'current_pct': current_pct,
    }

def check_results():
    """Check all result JSON files."""
    results = []
    for f in sorted(glob.glob('sweep_results/*_results.json')):
        with open(f) as fp:
            d = json.load(fp)
            results.append(d)
    return results

def main():
    print(f"=== Sweep Progress Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    # Check running experiments
    print("--- Active Experiments (from logs) ---")
    for log in sorted(glob.glob('sweep_logs/w*.log')):
        name = os.path.basename(log).replace('.log', '')
        info = parse_log(log)
        if info is None:
            continue

        # Check if there's a .pt checkpoint
        pt_file = f"sweep_results/{name}.pt"
        has_checkpoint = os.path.exists(pt_file)
        pt_size = os.path.getsize(pt_file) if has_checkpoint else 0

        status = ""
        if info['current_epoch'] is not None:
            status = f"Epoch {info['current_epoch']} @ {info['current_pct']}%"

        epoch_str = ""
        if info['epoch_results']:
            best = max(info['epoch_results'], key=lambda x: x['dev_acc'])
            epoch_str = f"best_dev_acc={best['dev_acc']:.4f} (epoch {best['epoch']})"

        print(f"  {name:<40} {status:<20} {epoch_str}")
        if info['epoch_results']:
            for er in info['epoch_results']:
                print(f"    Epoch {er['epoch']}: loss={er['loss']:.4f} dev_acc={er['dev_acc']:.4f} dev_f1={er['dev_f1']:.4f}")

    # Check completed experiments
    print("\n--- Completed Experiments (from results JSON) ---")
    results = check_results()
    if not results:
        print("  No completed experiments yet.")
    else:
        results.sort(key=lambda x: x['best_dev_acc'], reverse=True)
        print(f"  {'Run Name':<35} {'Dev Acc':<10} {'Dev F1':<10} {'LR':<10} {'Epochs':<7}")
        print(f"  {'-'*72}")
        for r in results:
            marker = " ***" if r['best_dev_acc'] > 0.898 else ""
            print(f"  {r['run_name']:<35} {r['best_dev_acc']:<10.4f} {r['best_dev_f1']:<10.4f} {r['lr']:<10} {r['epochs']:<7}{marker}")

        best = results[0]
        print(f"\n  Best overall: {best['run_name']} = {best['best_dev_acc']:.4f}")
        if best['best_dev_acc'] > 0.898:
            print(f"  *** BEATS STAFF BASELINE (0.898) by {best['best_dev_acc'] - 0.898:.4f} ***")
        else:
            print(f"  Below staff baseline (0.898) by {0.898 - best['best_dev_acc']:.4f}")

if __name__ == "__main__":
    main()
