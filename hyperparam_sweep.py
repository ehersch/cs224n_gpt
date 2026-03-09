"""
Hyperparameter sweep for paraphrase detection.
Runs a single experiment with specified hyperparams on a specified GPU.
"""
import argparse
import os
import sys
import json
import random
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data,
)
from evaluation import model_eval_paraphrase
from models.gpt2 import GPT2Model
from optimizer import AdamW

TQDM_DISABLE = False


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
        )
        self.paraphrase_detection_head = nn.Linear(args.d, 2)
        # Optional dropout before classification head
        self.dropout = nn.Dropout(args.clf_dropout) if hasattr(args, 'clf_dropout') and args.clf_dropout > 0 else nn.Identity()

        for param in self.gpt.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        gpt_output = self.gpt(input_ids, attention_mask)
        hidden_states = gpt_output["last_hidden_state"]
        last_hidden_state = hidden_states[:, -1, :]
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.paraphrase_detection_head(last_hidden_state)
        return logits


def add_arguments(args):
    if args.model_size == "gpt2":
        args.d = 768
        args.l = 12
        args.num_heads = 12
    elif args.model_size == "gpt2-medium":
        args.d = 1024
        args.l = 24
        args.num_heads = 16
    elif args.model_size == "gpt2-large":
        args.d = 1280
        args.l = 36
        args.num_heads = 20
    return args


def train_and_eval(args):
    """Train and evaluate, returning best dev accuracy."""
    device = torch.device(f"cuda:{args.gpu}")

    # Enable TF32 for Ampere GPUs (RTX 3090) - faster matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    para_train_data = load_paraphrase_data(args.para_train)
    para_dev_data = load_paraphrase_data(args.para_dev)

    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(
        para_train_data,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=para_train_data.collate_fn,
    )
    para_dev_dataloader = DataLoader(
        para_dev_data,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn,
    )

    args = add_arguments(args)
    model = ParaphraseGPT(args)
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Optional: linear warmup + cosine decay scheduler
    if args.use_scheduler:
        total_steps = len(para_train_dataloader) * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    best_dev_acc = 0
    best_dev_f1 = 0
    epoch_results = []

    # Gradient accumulation
    accum_steps = getattr(args, 'grad_accum_steps', 1)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(para_train_dataloader, desc=f"[GPU{args.gpu}] train-{epoch}", disable=TQDM_DISABLE)):
            b_ids = batch["token_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].flatten().to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                logits = model(b_ids, b_mask)
                loss = F.cross_entropy(logits, labels, reduction="mean")
            loss = loss / accum_steps
            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps
            num_batches += 1

        # Handle remaining gradients
        if num_batches % accum_steps != 0:
            scaler.unscale_(optimizer)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss / num_batches
        with torch.cuda.amp.autocast(enabled=args.fp16):
            dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

        epoch_result = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "dev_acc": round(dev_acc, 4),
            "dev_f1": round(dev_f1, 4),
        }
        epoch_results.append(epoch_result)
        print(f"[GPU{args.gpu}] Epoch {epoch}: loss={train_loss:.4f}, dev_acc={dev_acc:.4f}, dev_f1={dev_f1:.4f}", flush=True)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_dev_f1 = dev_f1
            save_info = {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "args": args,
                "system_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.random.get_rng_state(),
            }
            torch.save(save_info, args.filepath)
            print(f"[GPU{args.gpu}] New best: dev_acc={dev_acc:.4f} (saved to {args.filepath})", flush=True)

    return {
        "best_dev_acc": round(best_dev_acc, 4),
        "best_dev_f1": round(best_dev_f1, 4),
        "epoch_results": epoch_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clf_dropout", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--model_size", type=str, default="gpt2",
                        choices=["gpt2", "gpt2-medium", "gpt2-large"])
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (fp16) training")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="sweep_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.filepath = os.path.join(args.output_dir, f"{args.run_name}.pt")

    seed_everything(args.seed)

    start_time = time.time()
    results = train_and_eval(args)
    elapsed = time.time() - start_time

    # Save results JSON
    summary = {
        "run_name": args.run_name,
        "model_size": args.model_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "clf_dropout": args.clf_dropout,
        "epochs": args.epochs,
        "max_grad_norm": args.max_grad_norm,
        "use_scheduler": args.use_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "grad_accum_steps": args.grad_accum_steps,
        "fp16": args.fp16,
        "seed": args.seed,
        "gpu": args.gpu,
        "best_dev_acc": results["best_dev_acc"],
        "best_dev_f1": results["best_dev_f1"],
        "epoch_results": results["epoch_results"],
        "elapsed_seconds": round(elapsed, 1),
    }

    result_path = os.path.join(args.output_dir, f"{args.run_name}_results.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[GPU{args.gpu}] DONE: {args.run_name} | best_dev_acc={results['best_dev_acc']:.4f} | time={elapsed:.0f}s", flush=True)
    print(f"Results saved to {result_path}", flush=True)


if __name__ == "__main__":
    main()
