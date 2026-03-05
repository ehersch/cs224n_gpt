"""
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.

With LoRA (fewer params, often better for small datasets):
  `python sonnet_generation.py --use_gpu --use_lora`

SFT training on Modal (GPU in the cloud):
  `modal run sonnet_generation.py`           # single run
  `modal run sonnet_generation.py::main_lora_sweep`   # LoRA sweep over ranks 4,8,16,32,64
  `modal run sonnet_generation.py::main_combined`    # train on data/sonnets.txt + synthetic_data/synthetic_sonnets.txt
  Optional: --epochs, --batch-size, --lr, --use-lora, --lora-rank, etc.
  Sweep: --lora-ranks "4,8,16,32,64" (custom list).
"""

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
    SonnetsDataset,
)
from models.gpt2 import GPT2Model
from modules.lora import LoRALinear

from optimizer import AdamW

TQDM_DISABLE = False

# Require dev loss to improve by at least this much to count as improvement (and reset early stopping).
DEV_LOSS_IMPROVEMENT_TOLERANCE = 0.1


def _fake_quantize_ste(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric fake-quantization with straight-through estimator (STE)."""
    if bits <= 0:
        return x
    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax - 1
    max_abs = x.detach().abs().max()
    scale = torch.clamp(max_abs / max(qmax, 1), min=1e-8)
    x_q = torch.clamp(torch.round(x / scale), qmin, qmax) * scale
    return x + (x_q - x).detach()


class QATLinear(nn.Module):
    """nn.Linear wrapper that applies fake quantization to weights/bias on each forward."""

    def __init__(self, base: nn.Linear, bits: int):
        super().__init__()
        self.base = base
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _fake_quantize_ste(self.base.weight, self.bits)
        b = (
            _fake_quantize_ste(self.base.bias, self.bits)
            if self.base.bias is not None
            else None
        )
        return F.linear(x, w, b)


def _apply_qat_weight_fake_quant(module: nn.Module, bits: int):
    """Recursively replace nn.Linear modules with QATLinear wrappers."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QATLinear(child, bits))
        else:
            _apply_qat_weight_fake_quant(child, bits)


def _estimate_model_size_mb(model: nn.Module, weight_bits: int | None = None) -> float:
    """Estimate model size in MB; uses custom bitwidth when provided."""
    total_params = sum(p.numel() for p in model.parameters())
    if weight_bits is None:
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    else:
        total_bytes = total_params * (weight_bits / 8.0)
    return float(total_bytes / (1024**2))


def _load_sonnets_from_file(file_path):
    """Load sonnet strings from .txt/.json files."""
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [str(s).strip() for s in payload if str(s).strip()]
        if isinstance(payload, dict):
            if "sonnets" in payload and isinstance(payload["sonnets"], list):
                return [str(s).strip() for s in payload["sonnets"] if str(s).strip()]
            raise ValueError(
                f"Unsupported JSON schema in {file_path}. Expected list or dict with 'sonnets'."
            )
        raise ValueError(f"Unsupported JSON schema in {file_path}.")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Synthetic format uses ###1###, ###2###, etc.
    if "###" in text[:2000]:
        parts = re.split(r"###\d+###", text)
        sonnets = [s.strip() for s in parts if s.strip()]
    else:
        # Standard format: split by newline-number-newline, skip header
        sonnets = re.split(r"\n\s*\d+\s*\n", text)[1:]
        sonnets = [s.strip() for s in sonnets]
    return sonnets


def load_combined_sonnets(paths):
    """Load and concatenate sonnets from multiple files."""
    all_sonnets = []
    for path in paths:
        sonnets = _load_sonnets_from_file(path)
        all_sonnets.extend(sonnets)
    return all_sonnets


class SonnetsFromListDataset(Dataset):
    """Dataset from a list of sonnet strings (same interface as SonnetsDataset for batching)."""

    def __init__(self, sonnet_list):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sonnets = sonnet_list

    def __len__(self):
        return len(self.sonnets)

    def __getitem__(self, idx):
        return (idx, self.sonnets[idx])

    def collate_fn(self, all_data):
        idx = [example[0] for example in all_data]
        sonnets = [example[1] for example in all_data]
        encoding = self.tokenizer(
            sonnets, return_tensors="pt", padding=True, truncation=True
        )
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])
        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "sent_ids": idx,
        }


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _apply_lora_to_gpt(gpt, rank, alpha):
    """Freeze base GPT and wrap target linear layers with LoRA."""
    for param in gpt.parameters():
        param.requires_grad = False

    for layer in gpt.gpt_layers:
        # Attention: Q, K, V
        layer.self_attention.query = LoRALinear(
            layer.self_attention.query, rank=rank, alpha=alpha
        )
        layer.self_attention.key = LoRALinear(
            layer.self_attention.key, rank=rank, alpha=alpha
        )
        layer.self_attention.value = LoRALinear(
            layer.self_attention.value, rank=rank, alpha=alpha
        )
        # Attention output
        layer.attention_dense = LoRALinear(
            layer.attention_dense, rank=rank, alpha=alpha
        )
        # MLP
        layer.interm_dense = LoRALinear(layer.interm_dense, rank=rank, alpha=alpha)
        layer.out_dense = LoRALinear(layer.out_dense, rank=rank, alpha=alpha)


class SonnetGPT(nn.Module):
    """Your GPT-2 Model designed for sonnet generation."""

    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_lora = getattr(args, "use_lora", False)
        self.qat_weight_bits = int(getattr(args, "qat_weight_bits", 0) or 0)

        if self.use_lora:
            _apply_lora_to_gpt(self.gpt, rank=args.lora_rank, alpha=args.lora_alpha)
        else:
            for param in self.gpt.parameters():
                param.requires_grad = True

        if self.qat_weight_bits > 0:
            _apply_qat_weight_fake_quant(self, self.qat_weight_bits)
            print(
                f"QAT enabled: fake-quantizing linear weights to {self.qat_weight_bits}-bit before/during SFT."
            )

    def forward(self, input_ids, attention_mask):
        """
        This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
        not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
        not just the distribution over next tokens for the last token!
        """
        ### YOUR CODE HERE
        gpt_output = self.gpt(input_ids, attention_mask)
        last_hidden_state = gpt_output["last_hidden_state"]
        logits = self.gpt.hidden_state_to_token(last_hidden_state)
        return logits

    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    # @torch.no_grad()
    # def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    #   """
    #   Generates an original sonnet using top-p sampling and softmax temperature.

    #   TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    #   In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    #   there are many.
    #   """
    #   token_ids = encoding.to(self.get_device())
    #   attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    #   for _ in range(max_length):
    #     # Forward pass to get logits
    #     logits_sequence = self.forward(token_ids, attention_mask)
    #     logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

    #     # Convert logits to probabilities
    #     probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

    #     # Top-p (nucleus) sampling
    #     sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    #     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    #     top_p_mask = cumulative_probs <= top_p
    #     top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
    #     top_p_mask[..., 0] = True  # Always include the highest probability token
    #     filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
    #     filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

    #     # Sample from filtered distribution
    #     sampled_index = torch.multinomial(filtered_probs, 1)
    #     sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

    #     # Stop if end-of-sequence token is reached
    #     if sampled_token.item() == self.tokenizer.eos_token_id:
    #       break

    #     # Append sampled token
    #     token_ids = torch.cat([token_ids, sampled_token], dim=1)
    #     attention_mask = torch.cat(
    #       [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
    #     )

    #   generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    #   return token_ids, generated_output

    @torch.no_grad()
    def generate(self, encoding, temperature=0.8, top_p=0.9, top_k=40, max_length=128):
        """
        Generates a sonnet using optimized Top-K and Top-P (Nucleus) sampling.
        """
        model_device = self.get_device()
        token_ids = encoding.to(model_device)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        current_ids = token_ids

        for _ in range(max_length):
            attention_mask = torch.ones_like(current_ids).to(model_device)
            logits = self.forward(current_ids, attention_mask)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Top-P (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        generated_output = self.tokenizer.decode(
            current_ids[0], skip_special_tokens=True
        )
        return current_ids, generated_output


def save_model(model, optimizer, args, filepath):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train(args):
    """Train GPT-2 for sonnet generation with padding mask, grad clip, LR schedule, early stopping."""
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(
        sonnet_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=sonnet_dataset.collate_fn,
        num_workers=args.num_workers,
    )

    args = add_arguments(args)
    model = SonnetGPT(args)
    model = model.to(device)

    fp32_size_mb = _estimate_model_size_mb(model, weight_bits=None)
    quant_bits = int(getattr(args, "qat_weight_bits", 0) or 0)
    quant_size_mb = (
        _estimate_model_size_mb(model, weight_bits=quant_bits)
        if quant_bits > 0
        else fp32_size_mb
    )
    print(
        f"Model size estimate: fp32={fp32_size_mb:.2f} MB, "
        + (
            f"quantized({quant_bits}-bit)={quant_size_mb:.2f} MB"
            if quant_bits > 0
            else "quantized=disabled"
        )
    )

    pretrain_dev_loss = compute_dev_ce_loss(
        model, device, args.dev_gold_path, batch_size=args.batch_size
    )
    print(
        f"Pre-SFT dev CE loss on {'quantized' if quant_bits > 0 else 'base'} pretrained model: {pretrain_dev_loss:.3f}"
    )

    lr = args.lr
    if args.use_lora:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=lr, weight_decay=args.weight_decay)
        print(f"LoRA enabled: training {sum(p.numel() for p in params):,} parameters")
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    pad_id = model.tokenizer.pad_token_id
    best_dev_loss = float("inf")
    best_epoch_dev = -1
    total_steps = 0
    total_examples = 0
    train_start_time = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(
            sonnet_dataloader, desc=f"train-{epoch}", disable=TQDM_DISABLE
        ):
            b_ids, b_mask = batch["token_ids"], batch["attention_mask"]
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")
            labels = b_ids[:, 1:].contiguous().flatten()
            loss = F.cross_entropy(
                logits, labels, reduction="mean", ignore_index=pad_id
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters() if not args.use_lora else params,
                max_norm=args.grad_clip_max_norm,
            )
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            total_steps += 1
            total_examples += b_ids.size(0)

        train_loss = train_loss / num_batches
        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}.")

        save_model(model, optimizer, args, f"{epoch}_{args.filepath}")

        if epoch % 5 == 0 and epoch != 0:
            model.eval()
            dev_loss = compute_dev_ce_loss(
                model, device, args.dev_gold_path, batch_size=args.batch_size
            )
            print(f"  Dev CE loss (epoch {epoch}): {dev_loss:.3f}")
            if dev_loss < best_dev_loss - DEV_LOSS_IMPROVEMENT_TOLERANCE:
                best_dev_loss = dev_loss
                best_epoch_dev = epoch
                save_model(model, optimizer, args, f"{args.epochs-1}_{args.filepath}")
                print(
                    f"  New best dev CE loss {best_dev_loss:.3f}, saved as final checkpoint."
                )
            model.train()

        if (
            best_epoch_dev >= 0
            and epoch - best_epoch_dev >= args.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch} (no dev loss improvement for {args.early_stopping_patience} epochs)."
            )
            break

    elapsed_sec = time.perf_counter() - train_start_time
    ex_per_sec = (total_examples / elapsed_sec) if elapsed_sec > 0 else 0.0
    steps_per_sec = (total_steps / elapsed_sec) if elapsed_sec > 0 else 0.0
    print(
        f"SFT speed: {elapsed_sec:.2f}s total, {ex_per_sec:.2f} examples/s, {steps_per_sec:.2f} steps/s."
    )

    metrics_out = getattr(args, "quant_metrics_out", "")
    if metrics_out:
        metrics = {
            "qat_weight_bits": quant_bits,
            "pretrain_dev_ce_loss": float(pretrain_dev_loss),
            "train_elapsed_sec": float(elapsed_sec),
            "train_examples_per_sec": float(ex_per_sec),
            "train_steps_per_sec": float(steps_per_sec),
            "estimated_model_size_mb_fp32": float(fp32_size_mb),
            "estimated_model_size_mb_quantized": float(quant_size_mb),
            "best_dev_ce_loss": (
                float(best_dev_loss) if best_dev_loss < float("inf") else None
            ),
            "best_dev_epoch": int(best_epoch_dev),
        }
        os.makedirs(os.path.dirname(metrics_out) or ".", exist_ok=True)
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved quantization/SFT metrics to {metrics_out}")


@torch.no_grad()
def generate_submission_sonnets(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    saved = torch.load(
        f"{args.epochs-1}_{args.filepath}",
        weights_only=False,
        map_location=device,
    )

    model = SonnetGPT(saved["args"])
    model.load_state_dict(saved["model"])
    model = model.to(device)
    model.eval()

    # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
        sonnet_id = batch[0]
        encoding = model.tokenizer(
            batch[1], return_tensors="pt", padding=False, truncation=True
        ).to(device)
        output = model.generate(
            encoding["input_ids"], temperature=args.temperature, top_p=args.top_p
        )[0][0]
        decoded_output = model.tokenizer.decode(output, skip_special_tokens=True)
        full_sonnet = f"{decoded_output}\n\n"
        generated_sonnets.append((sonnet_id, full_sonnet))

        # print(f'{decoded_output}\n\n')

    with open(args.sonnet_out, "w+") as f:
        f.write(f"--Generated Sonnets-- \n\n")
        for sonnet in generated_sonnets:
            f.write(f"\n{sonnet[0]}\n")
            f.write(sonnet[1])


@torch.no_grad()
def compute_dev_ce_loss(model, device, dev_gold_path, batch_size=8):
    """Compute mean cross-entropy loss on the true dev sonnets (e.g. data/TRUE_sonnets_held_out_dev.txt)."""
    dev_dataset = SonnetsDataset(dev_gold_path)
    dev_loader = DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=dev_dataset.collate_fn,
    )
    pad_id = model.tokenizer.pad_token_id
    total_loss = 0.0
    num_batches = 0
    for batch in dev_loader:
        b_ids = batch["token_ids"].to(device)
        b_mask = batch["attention_mask"].to(device)
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(logits, labels, reduction="mean", ignore_index=pad_id)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches if num_batches else float("inf")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument(
        "--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt"
    )
    parser.add_argument(
        "--sonnet_out", type=str, default="predictions/generated_sonnets_dev.txt"
    )
    parser.add_argument(
        "--dev_held_out_path", type=str, default="data/sonnets_held_out_dev.txt"
    )
    parser.add_argument(
        "--dev_gold_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt"
    )

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action="store_true")

    # Generation parameters.
    parser.add_argument(
        "--temperature", type=float, help="softmax temperature.", default=1.2
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="Cumulative probability distribution for nucleus sampling.",
        default=0.9,
    )

    parser.add_argument(
        "--batch_size", help="The training batch size.", type=int, default=8
    )
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument(
        "--model_size",
        type=str,
        help="The model size as specified on hugging face.",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        default="gpt2-xl",
    )

    # Training improvements
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="AdamW weight decay"
    )
    parser.add_argument(
        "--grad_clip_max_norm",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Stop after N epochs without dev CHRF improvement",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader num_workers (0 to disable)",
    )

    # LoRA (Low-Rank Adaptation) - fewer trainable params, often better for small datasets
    parser.add_argument(
        "--use_lora", action="store_true", help="Use LoRA instead of full fine-tuning"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=8, help="LoRA rank (typical: 4, 8, 16)"
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=16.0, help="LoRA scaling (2*rank)"
    )
    parser.add_argument(
        "--qat_weight_bits",
        type=int,
        default=0,
        choices=[0, 4, 8],
        help="Enable quantization-aware SFT by fake-quantizing linear weights before training (0 disables).",
    )
    parser.add_argument(
        "--quant_metrics_out",
        type=str,
        default="predictions/quant_sft_metrics.json",
        help="Where to save model-size/speed/eval metrics for quantized SFT.",
    )

    args = parser.parse_args()
    return args


def add_arguments(args):
    """Add arguments that are deterministic on model size."""
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
    else:
        raise Exception(f"{args.model_size} is not supported.")
    return args


# ---------------------------------------------------------------------------
# Modal SFT training: run with `modal run sonnet_generation.py`
# ---------------------------------------------------------------------------
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    MODAL_WORKSPACE = "/workspace"
    MODAL_CHECKPOINTS = "/checkpoints"

    sonnet_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch",
            "transformers",
            "einops",
            "tqdm",
            "requests",
            "filelock",
            "importlib-metadata",
            "sacrebleu",
            "scikit-learn",
        )
        .workdir(MODAL_WORKSPACE)
        .add_local_dir(".", remote_path=MODAL_WORKSPACE, ignore=[".git", ".git/*"])
    )

    volume = modal.Volume.from_name("sonnet-checkpoints", create_if_missing=True)
    volumes = {MODAL_CHECKPOINTS: volume}

    app = modal.App("sonnet-lora-alpha-sweep")

    @app.function(
        image=sonnet_image,
        gpu="L4",
        timeout=3600 * 2,  # 2 hours
        volumes=volumes,
    )
    def train_sft(
        sonnet_path: str = "data/sonnets.txt",
        held_out_sonnet_path: str = "data/sonnets_held_out_dev.txt",
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 1e-5,
        model_size: str = "gpt2",
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        seed: int = 11711,
        temperature: float = 1.2,
        top_p: float = 0.9,
        weight_decay: float = 0.01,
        grad_clip_max_norm: float = 1.0,
        early_stopping_patience: int = 5,
        num_workers: int = 2,
        dev_held_out_path: str = "data/sonnets_held_out_dev.txt",
        dev_gold_path: str = "data/TRUE_sonnets_held_out_dev.txt",
        checkpoint_subdir: str = "",
        run_name: str = "",
        init_checkpoint: str = "",
    ):
        """Run SFT training on Modal. Dev CHRF every 5 epochs; early stop on dev CHRF."""
        sys.path.insert(0, MODAL_WORKSPACE)
        seed_everything(seed)

        args = argparse.Namespace(
            sonnet_path=os.path.join(MODAL_WORKSPACE, sonnet_path),
            held_out_sonnet_path=os.path.join(MODAL_WORKSPACE, held_out_sonnet_path),
            sonnet_out="predictions/generated_sonnets_dev.txt",
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_size=model_size,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            temperature=temperature,
            top_p=top_p,
            weight_decay=weight_decay,
            grad_clip_max_norm=grad_clip_max_norm,
            early_stopping_patience=early_stopping_patience,
            num_workers=num_workers,
            dev_held_out_path=os.path.join(MODAL_WORKSPACE, dev_held_out_path),
            dev_gold_path=os.path.join(MODAL_WORKSPACE, dev_gold_path),
        )
        filename = (
            f"{epochs}-{lr}-sonnet-lora{lora_rank}-alpha{lora_alpha}.pt"
            if use_lora
            else f"{epochs}-{lr}-sonnet.pt"
        )
        args.filepath = f"{run_name}-{filename}" if run_name else filename
        args.use_gpu = True

        checkpoint_dir = (
            os.path.join(MODAL_CHECKPOINTS, checkpoint_subdir)
            if checkpoint_subdir
            else MODAL_CHECKPOINTS
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        device = torch.device("cuda")
        sonnet_dataset = SonnetsDataset(args.sonnet_path)
        sonnet_dataloader = DataLoader(
            sonnet_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sonnet_dataset.collate_fn,
            num_workers=num_workers,
        )

        args = add_arguments(args)
        model = SonnetGPT(args)
        model = model.to(device)
        if init_checkpoint:
            init_path = os.path.join(MODAL_CHECKPOINTS, init_checkpoint)
            print(f"Loading init checkpoint from {init_checkpoint}")
            init_saved = torch.load(init_path, weights_only=False, map_location=device)
            model.load_state_dict(init_saved["model"], strict=False)

        if args.use_lora:
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
            print(
                f"LoRA enabled: training {sum(p.numel() for p in params):,} parameters"
            )
        else:
            optimizer = AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )

        pad_id = model.tokenizer.pad_token_id
        best_dev_loss = float("inf")
        best_epoch_dev = -1

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            for batch in tqdm(sonnet_dataloader, desc=f"train-{epoch}", disable=False):
                b_ids = batch["token_ids"].to(device)
                b_mask = batch["attention_mask"].to(device)
                optimizer.zero_grad()
                logits = model(b_ids, b_mask)
                logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")
                labels = b_ids[:, 1:].contiguous().flatten()
                loss = F.cross_entropy(
                    logits, labels, reduction="mean", ignore_index=pad_id
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    params if args.use_lora else model.parameters(),
                    max_norm=args.grad_clip_max_norm,
                )
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches
            print(f"Epoch {epoch}: train loss :: {train_loss:.3f}")

            if epoch % 5 == 0 and epoch != 0:
                model.eval()
                dev_loss = compute_dev_ce_loss(
                    model, device, args.dev_gold_path, batch_size=args.batch_size
                )
                print(f"  Dev CE loss (epoch {epoch}): {dev_loss:.3f}")
                if dev_loss < best_dev_loss - DEV_LOSS_IMPROVEMENT_TOLERANCE:
                    best_dev_loss = dev_loss
                    best_epoch_dev = epoch
                    best_name = f"best_{args.filepath}"
                    best_path = os.path.join(checkpoint_dir, best_name)
                    save_model(model, optimizer, args, best_path)
                    print(
                        f"  New best dev CE loss {best_dev_loss:.3f}, saved as only checkpoint."
                    )
                model.train()

            if (
                best_epoch_dev >= 0
                and epoch - best_epoch_dev >= args.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch} (no dev loss improvement for {args.early_stopping_patience} epochs)."
                )
                break
        # If we never ran dev (e.g. < 5 epochs), save current model as the checkpoint
        if best_epoch_dev < 0:
            best_name = f"best_{args.filepath}"
            best_path = os.path.join(checkpoint_dir, best_name)
            save_model(model, optimizer, args, best_path)
            print(f"Saved checkpoint (no dev run): {best_name}")
        volume.commit()

        final_name = (
            f"{checkpoint_subdir}/best_{args.filepath}"
            if checkpoint_subdir
            else f"best_{args.filepath}"
        )
        return {"final_checkpoint": final_name}

    @app.function(
        image=sonnet_image,
        gpu="L4:4",
        timeout=3600 * 2,
        volumes=volumes,
    )
    def train_sft_combined(
        sonnet_path: str = "data/sonnets.txt",
        synthetic_sonnet_path: str = "synthetic_data/synthetic_sonnets.txt",
        held_out_sonnet_path: str = "data/sonnets_held_out_dev.txt",
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 1e-5,
        model_size: str = "gpt2",
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        seed: int = 11711,
        temperature: float = 1.2,
        top_p: float = 0.9,
        weight_decay: float = 0.01,
        grad_clip_max_norm: float = 1.0,
        early_stopping_patience: int = 5,
        num_workers: int = 2,
        dev_held_out_path: str = "data/sonnets_held_out_dev.txt",
        dev_gold_path: str = "data/TRUE_sonnets_held_out_dev.txt",
        checkpoint_subdir: str = "",
        run_name: str = "",
        init_checkpoint: str = "",
    ):
        """Train on both data/sonnets.txt and synthetic_data/synthetic_sonnets.txt. Supports use_lora; generates dev txt at end via main_combined."""
        sys.path.insert(0, MODAL_WORKSPACE)
        seed_everything(seed)

        path1 = os.path.join(MODAL_WORKSPACE, sonnet_path)
        path2 = os.path.join(MODAL_WORKSPACE, synthetic_sonnet_path)
        combined_sonnets = load_combined_sonnets([path1, path2])
        print(
            f"Combined dataset: {len(combined_sonnets)} sonnets from {sonnet_path} + {synthetic_sonnet_path}"
        )

        args = argparse.Namespace(
            sonnet_path=path1,
            held_out_sonnet_path=os.path.join(MODAL_WORKSPACE, held_out_sonnet_path),
            sonnet_out="predictions/generated_sonnets_dev.txt",
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_size=model_size,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            temperature=temperature,
            top_p=top_p,
            weight_decay=weight_decay,
            grad_clip_max_norm=grad_clip_max_norm,
            early_stopping_patience=early_stopping_patience,
            num_workers=num_workers,
            dev_held_out_path=os.path.join(MODAL_WORKSPACE, dev_held_out_path),
            dev_gold_path=os.path.join(MODAL_WORKSPACE, dev_gold_path),
        )
        filename = (
            f"{epochs}-{lr}-sonnet-combined-lora{lora_rank}-alpha{lora_alpha}.pt"
            if use_lora
            else f"{epochs}-{lr}-sonnet-combined.pt"
        )
        args.filepath = f"{run_name}-{filename}" if run_name else filename
        args.use_gpu = True

        checkpoint_dir = (
            os.path.join(MODAL_CHECKPOINTS, checkpoint_subdir)
            if checkpoint_subdir
            else MODAL_CHECKPOINTS
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        device = torch.device("cuda")
        sonnet_dataset = SonnetsFromListDataset(combined_sonnets)
        sonnet_dataloader = DataLoader(
            sonnet_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sonnet_dataset.collate_fn,
            num_workers=num_workers,
        )

        args = add_arguments(args)
        model = SonnetGPT(args)
        model = model.to(device)
        if init_checkpoint:
            init_path = os.path.join(MODAL_CHECKPOINTS, init_checkpoint)
            print(f"Loading init checkpoint from {init_checkpoint}")
            init_saved = torch.load(init_path, weights_only=False, map_location=device)
            model.load_state_dict(init_saved["model"], strict=False)

        if args.use_lora:
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
            print(
                f"LoRA enabled: training {sum(p.numel() for p in params):,} parameters"
            )
        else:
            optimizer = AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        params = (
            [p for p in model.parameters() if p.requires_grad]
            if args.use_lora
            else model.parameters()
        )

        pad_id = model.tokenizer.pad_token_id
        best_dev_loss = float("inf")
        best_epoch_dev = -1

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            for batch in tqdm(sonnet_dataloader, desc=f"train-{epoch}", disable=False):
                b_ids = batch["token_ids"].to(device)
                b_mask = batch["attention_mask"].to(device)
                optimizer.zero_grad()
                logits = model(b_ids, b_mask)
                logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")
                labels = b_ids[:, 1:].contiguous().flatten()
                loss = F.cross_entropy(
                    logits, labels, reduction="mean", ignore_index=pad_id
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    params,
                    max_norm=args.grad_clip_max_norm,
                )
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches
            print(f"Epoch {epoch}: train loss :: {train_loss:.3f}")

            if epoch % 5 == 0 and epoch != 0:
                model.eval()
                dev_loss = compute_dev_ce_loss(
                    model, device, args.dev_gold_path, batch_size=args.batch_size
                )
                print(f"  Dev CE loss (epoch {epoch}): {dev_loss:.3f}")
                if dev_loss < best_dev_loss - DEV_LOSS_IMPROVEMENT_TOLERANCE:
                    best_dev_loss = dev_loss
                    best_epoch_dev = epoch
                    best_name = f"best_{args.filepath}"
                    best_path = os.path.join(checkpoint_dir, best_name)
                    save_model(model, optimizer, args, best_path)
                    print(
                        f"  New best dev CE loss {best_dev_loss:.3f}, saved as only checkpoint."
                    )
                model.train()

            if (
                best_epoch_dev >= 0
                and epoch - best_epoch_dev >= args.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch} (no dev loss improvement for {args.early_stopping_patience} epochs)."
                )
                break

        if best_epoch_dev < 0:
            best_name = f"best_{args.filepath}"
            best_path = os.path.join(checkpoint_dir, best_name)
            save_model(model, optimizer, args, best_path)
            print(f"Saved checkpoint (no dev run): {best_name}")
        volume.commit()

        final_name = (
            f"{checkpoint_subdir}/best_{args.filepath}"
            if checkpoint_subdir
            else f"best_{args.filepath}"
        )
        return {"final_checkpoint": final_name}

    @app.function(image=sonnet_image, volumes=volumes)
    def download_checkpoint(filename: str) -> bytes:
        """Read checkpoint from Volume (must run remotely)."""
        volume.reload()
        path = os.path.join(MODAL_CHECKPOINTS, filename)
        with open(path, "rb") as f:
            return f.read()

    @app.function(image=sonnet_image, gpu="L4", volumes=volumes, timeout=600)
    def run_generation_and_get_submission(
        checkpoint_filename: str,
        held_out_path: str,
        temperature: float = 1.2,
        top_p: float = 0.9,
    ) -> str:
        """Load checkpoint from Volume, run generation, return submission file content."""
        sys.path.insert(0, MODAL_WORKSPACE)
        volume.reload()
        checkpoint_path = os.path.join(MODAL_CHECKPOINTS, checkpoint_filename)
        saved = torch.load(checkpoint_path, weights_only=False, map_location="cuda")
        args = add_arguments(saved["args"])
        model = SonnetGPT(args)
        model.load_state_dict(saved["model"])
        model = model.to("cuda")
        model.eval()

        held_out_full = os.path.join(MODAL_WORKSPACE, held_out_path)
        dataset = SonnetsDataset(held_out_full)
        generated_sonnets = []
        for batch in dataset:
            sonnet_id = batch[0]
            encoding = model.tokenizer(
                batch[1], return_tensors="pt", padding=False, truncation=True
            ).to("cuda")
            output = model.generate(
                encoding["input_ids"], temperature=temperature, top_p=top_p
            )[0][0]
            decoded = model.tokenizer.decode(output, skip_special_tokens=True)
            generated_sonnets.append((sonnet_id, f"{decoded}\n\n"))

        lines = ["--Generated Sonnets-- \n\n"]
        for sonnet in generated_sonnets:
            lines.append(f"\n{sonnet[0]}\n")
            lines.append(sonnet[1])
        return "".join(lines)

    @app.local_entrypoint()
    def main(
        sonnet_path: str = "data/sonnets.txt",
        held_out_sonnet_path: str = "data/sonnets_held_out_dev.txt",
        sonnet_out: str = "predictions/generated_sonnets_dev.txt",
        epochs: int = 25,
        batch_size: int = 32,
        lr: float = 5e-5,
        model_size: str = "gpt2",
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        temperature: float = 1.1,
        top_p: float = 0.9,
    ):
        """Run SFT training on Modal. Checkpoints are saved to the Volume (no local download)."""
        print("Running SFT training on Modal...")
        result = train_sft.remote(
            sonnet_path=sonnet_path,
            held_out_sonnet_path=held_out_sonnet_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_size=model_size,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            temperature=temperature,
            top_p=top_p,
        )
        final_checkpoint = result["final_checkpoint"]
        print(
            f"Training complete. Final checkpoint: {final_checkpoint} (in Volume 'sonnet-checkpoints')."
        )
        print("Running generation and downloading submission file...")
        submission_txt = run_generation_and_get_submission.remote(
            checkpoint_filename=final_checkpoint,
            held_out_path=held_out_sonnet_path,
            temperature=temperature,
            top_p=top_p,
        )
        os.makedirs(os.path.dirname(sonnet_out) or ".", exist_ok=True)
        with open(sonnet_out, "w") as f:
            f.write(submission_txt)
        print(f"Submission file saved to {sonnet_out}")

    LORA_RANKS = [4, 8, 16, 32, 64, 128]
    # LORA_RANKS = [64]

    @app.local_entrypoint()
    def main_lora_sweep(
        sonnet_path: str = "data/sonnets.txt",
        held_out_sonnet_path: str = "data/sonnets_held_out_dev.txt",
        sonnet_out: str = "predictions/generated_sonnets_dev.txt",
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        model_size: str = "gpt2",
        lora_alpha: float = 16.0,
        temperature: float = 1.2,
        top_p: float = 0.9,
        lora_ranks: str = "32,64,128",
    ):
        """Run SFT with LoRA sequentially for multiple ranks (default: 4,8,16,32,64)."""
        ranks = [int(r.strip()) for r in lora_ranks.split(",")]
        for i, rank in enumerate(ranks):
            print(f"\n{'='*60}")
            print(f"Experiment {i+1}/{len(ranks)}: LoRA rank = {rank}")
            print(f"{'='*60}\n")
            result = train_sft.remote(
                sonnet_path=sonnet_path,
                held_out_sonnet_path=held_out_sonnet_path,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                model_size=model_size,
                use_lora=True,
                lora_rank=rank,
                lora_alpha=lora_alpha,
                temperature=temperature,
                top_p=top_p,
                weight_decay=0,
            )
            final_checkpoint = result["final_checkpoint"]
            print(
                f"LoRA rank {rank} done. Running generation and downloading submission file..."
            )
            base, ext = os.path.splitext(sonnet_out)
            run_sonnet_out = f"{base}_lora{rank}{ext}"
            submission_txt = run_generation_and_get_submission.remote(
                checkpoint_filename=final_checkpoint,
                held_out_path=held_out_sonnet_path,
                temperature=temperature,
                top_p=top_p,
            )
            os.makedirs(os.path.dirname(run_sonnet_out) or ".", exist_ok=True)
            with open(run_sonnet_out, "w") as f:
                f.write(submission_txt)
            print(f"Submission file saved to {run_sonnet_out}")

        print(f"\nDone. Ran {len(ranks)} experiments for LoRA ranks: {ranks}")

    @app.local_entrypoint()
    def main_lora_alpha_sweep(
        sonnet_path: str = "data/sonnets.txt",
        held_out_sonnet_path: str = "data/sonnets_held_out_dev.txt",
        sonnet_out: str = "predictions/generated_sonnets_dev.txt",
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
        model_size: str = "gpt2",
        lora_rank: int = 8,
        temperature: float = 1.2,
        top_p: float = 0.9,
        lora_alphas: str = "8,16,32",
    ):
        """Run SFT with LoRA sequentially for multiple alphas (rank fixed at 8). Default alphas: 8,16,32."""
        alphas = [int(a.strip()) for a in lora_alphas.split(",")]
        for i, alpha in enumerate(alphas):
            print(f"\n{'='*60}")
            print(
                f"Experiment {i+1}/{len(alphas)}: LoRA alpha = {alpha} (rank = {lora_rank})"
            )
            print(f"{'='*60}\n")
            result = train_sft.remote(
                sonnet_path=sonnet_path,
                held_out_sonnet_path=held_out_sonnet_path,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                model_size=model_size,
                use_lora=True,
                lora_rank=lora_rank,
                lora_alpha=alpha,
                temperature=temperature,
                top_p=top_p,
            )
            final_checkpoint = result["final_checkpoint"]
            print(
                f"LoRA alpha {alpha} done. Running generation and downloading submission file..."
            )
            base, ext = os.path.splitext(sonnet_out)
            run_sonnet_out = f"{base}_alpha{alpha}{ext}"
            submission_txt = run_generation_and_get_submission.remote(
                checkpoint_filename=final_checkpoint,
                held_out_path=held_out_sonnet_path,
                temperature=temperature,
                top_p=top_p,
            )
            os.makedirs(os.path.dirname(run_sonnet_out) or ".", exist_ok=True)
            with open(run_sonnet_out, "w") as f:
                f.write(submission_txt)
            print(f"Submission file saved to {run_sonnet_out}")

        print(
            f"\nDone. Ran {len(alphas)} experiments for LoRA alphas: {alphas} (rank={lora_rank})"
        )

    @app.local_entrypoint()
    def main_combined(
        sonnet_path: str = "data/sonnets.txt",
        synthetic_sonnet_path: str = "synthetic_data/synthetic_sonnets.txt",
        held_out_sonnet_path: str = "data/sonnets_held_out_dev.txt",
        sonnet_out: str = "predictions/generated_sonnets_dev.txt",
        epochs: int = 25,
        batch_size: int = 32,
        lr: float = 1e-3,
        model_size: str = "gpt2",
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        temperature: float = 1.2,
        top_p: float = 0.9,
        weight_decay: float = 0.0,
        grad_clip_max_norm: float = 1.0,
        early_stopping_patience: int = 5,
        checkpoint_subdir: str = "",
        run_name: str = "",
        init_checkpoint: str = "",
    ):
        """Train on both data/sonnets.txt and synthetic_data/synthetic_sonnets.txt on Modal; optional LoRA; write dev predictions to sonnet_out."""
        print("Running combined SFT (sonnets + synthetic) on Modal...")
        result = train_sft_combined.remote(
            sonnet_path=sonnet_path,
            synthetic_sonnet_path=synthetic_sonnet_path,
            held_out_sonnet_path=held_out_sonnet_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            model_size=model_size,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            temperature=temperature,
            top_p=top_p,
            weight_decay=weight_decay,
            grad_clip_max_norm=grad_clip_max_norm,
            early_stopping_patience=early_stopping_patience,
            checkpoint_subdir=checkpoint_subdir,
            run_name=run_name,
            init_checkpoint=init_checkpoint,
        )
        final_checkpoint = result["final_checkpoint"]
        print(
            f"Training complete. Final checkpoint: {final_checkpoint} (in Volume 'sonnet-checkpoints')."
        )
        print("Running generation and downloading submission file...")
        submission_txt = run_generation_and_get_submission.remote(
            checkpoint_filename=final_checkpoint,
            held_out_path=held_out_sonnet_path,
            temperature=temperature,
            top_p=top_p,
        )
        os.makedirs(os.path.dirname(sonnet_out) or ".", exist_ok=True)
        with open(sonnet_out, "w") as f:
            f.write(submission_txt)
        print(f"Submission file saved to {sonnet_out}")

    @app.local_entrypoint()
    def generate_from_checkpoint(
        checkpoint_filename: str,
        held_out_sonnet_path: str = "data/sonnets_held_out.txt",
        sonnet_out: str = "predictions/generated_sonnets_test.txt",
        temperature: float = 1.2,
        top_p: float = 0.9,
    ):
        """Generate submission text from an existing checkpoint in Modal Volume (no training)."""
        print(f"Running generation from checkpoint {checkpoint_filename} on Modal...")
        submission_txt = run_generation_and_get_submission.remote(
            checkpoint_filename=checkpoint_filename,
            held_out_path=held_out_sonnet_path,
            temperature=temperature,
            top_p=top_p,
        )
        os.makedirs(os.path.dirname(sonnet_out) or ".", exist_ok=True)
        with open(sonnet_out, "w") as f:
            f.write(submission_txt)
        print(f"Submission file saved to {sonnet_out}")


if __name__ == "__main__":
    # With Modal installed, "modal run" should only run inference. Use "train" subcommand to train.
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        args = get_args()
        qat_suffix = f"-qat{args.qat_weight_bits}" if args.qat_weight_bits > 0 else ""
        args.filepath = f"{args.epochs}-{args.lr}-sonnet{qat_suffix}.pt"  # Save path.
        seed_everything(args.seed)  # Fix the seed for reproducibility.
        train(args)
        generate_submission_sonnets(args)
    elif modal is not None:
        pass
    else:
        args = get_args()
        qat_suffix = f"-qat{args.qat_weight_bits}" if args.qat_weight_bits > 0 else ""
        args.filepath = f"{args.epochs}-{args.lr}-sonnet{qat_suffix}.pt"  # Save path.
        seed_everything(args.seed)  # Fix the seed for reproducibility.
        train(args)
        generate_submission_sonnets(args)
