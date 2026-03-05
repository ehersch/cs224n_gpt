import argparse
import copy
import os
import tempfile
import sys
import time

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader

from datasets import SonnetsDataset
from sonnet_generation import SonnetGPT, add_arguments

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from torchao.quantization import quantize_, Float8WeightOnlyConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)


def estimate_flops_per_token_gpt2(model) -> float:
    """Approximate FLOPs/token for decoder-only GPT."""
    # SonnetGPT stores the base model at model.gpt with config-like args already baked in.
    # Use dimensions from the loaded checkpoint args where available.
    # Formula: per-layer ~ (12*d^2 + 2*d*seq_len) for seq_len contribution separately.
    # Here we report the seq_len-independent base term and let caller add seq_len term.
    d = model.gpt.config.hidden_size if hasattr(model.gpt, "config") else None
    l = model.gpt.config.num_hidden_layers if hasattr(model.gpt, "config") else None
    if d is None or l is None:
        # Fallback for project custom GPT2Model
        d = model.gpt.embed_layer.weight.shape[1]
        l = len(model.gpt.gpt_layers)
    return float(l * 12 * d * d)


def estimate_flops_per_token_with_seqlen(model, seq_len: int) -> float:
    d = model.gpt.config.hidden_size if hasattr(model.gpt, "config") else None
    l = model.gpt.config.num_hidden_layers if hasattr(model.gpt, "config") else None
    if d is None or l is None:
        d = model.gpt.embed_layer.weight.shape[1]
        l = len(model.gpt.gpt_layers)
    return float(l * ((12 * d * d) + (2 * seq_len * d)))


def serialized_model_size_mb(model) -> float:
    """Measure serialized state_dict size to reflect quantized packing too."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(model.state_dict(), tmp_path)
        num_bytes = os.path.getsize(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return num_bytes / (1024**2)


@torch.no_grad()
def eval_dev_metrics(
    model, dev_gold_path: str, batch_size: int = 8, device: str = "cpu"
):
    model = model.to(device)
    model.eval()

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
    correct = 0
    total = 0

    for batch in dev_loader:
        b_ids = batch["token_ids"].to(device)
        b_mask = batch["attention_mask"].to(device)
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")
        labels = b_ids[:, 1:].contiguous().flatten()

        loss = F.cross_entropy(logits, labels, reduction="mean", ignore_index=pad_id)
        total_loss += loss.item()
        num_batches += 1

        preds = torch.argmax(logits, dim=-1)
        valid = labels != pad_id
        correct += (preds[valid] == labels[valid]).sum().item()
        total += valid.sum().item()

    ce = total_loss / num_batches if num_batches else float("inf")
    token_acc = (correct / total) if total else 0.0
    ppl = float(torch.exp(torch.tensor(ce)).item()) if ce < 50 else float("inf")
    return {"dev_ce": ce, "token_acc": token_acc, "ppl": ppl}


@torch.no_grad()
def eval_dev_chrf(
    model,
    dev_held_out_path: str,
    dev_gold_path: str,
    device: str = "cpu",
    temperature: float = 1.2,
    top_p: float = 0.9,
):
    from sacrebleu.metrics import CHRF

    model = model.to(device)
    model.eval()
    held_out_dataset = SonnetsDataset(dev_held_out_path)
    gold_dataset = SonnetsDataset(dev_gold_path)

    generated_sonnets = []
    total_gen_time = 0.0
    total_samples = 0
    total_new_tokens = 0
    for batch in held_out_dataset:
        encoding = model.tokenizer(
            batch[1], return_tensors="pt", padding=False, truncation=True
        ).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = model.generate(
            encoding["input_ids"], temperature=temperature, top_p=top_p
        )[0][0]
        if device == "cuda":
            torch.cuda.synchronize()
        total_gen_time += time.perf_counter() - t0
        decoded_output = model.tokenizer.decode(output, skip_special_tokens=True)
        generated_sonnets.append(f"{decoded_output}\n\n")
        total_samples += 1
        new_tokens = max(0, int(output.shape[0] - encoding["input_ids"].shape[1]))
        total_new_tokens += new_tokens

    true_sonnets = [x[1] for x in gold_dataset]
    max_len = min(len(true_sonnets), len(generated_sonnets))
    true_sonnets = true_sonnets[:max_len]
    generated_sonnets = generated_sonnets[:max_len]
    chrf = float(CHRF().corpus_score(generated_sonnets, [true_sonnets]).score)
    avg_sec_per_sample = (
        total_gen_time / total_samples if total_samples else float("inf")
    )
    toks_per_sec = total_new_tokens / total_gen_time if total_gen_time > 0 else 0.0
    return {
        "dev_chrf": chrf,
        "avg_gen_sec_per_sample": avg_sec_per_sample,
        "gen_tokens_per_sec": toks_per_sec,
    }


def load_checkpoint_model(checkpoint_path: str):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = add_arguments(saved["args"])
    model = SonnetGPT(args)
    model.load_state_dict(saved["model"], strict=False)
    model.eval()
    return model


def make_variant(model, variant: str):
    variant = variant.lower().strip()
    m = copy.deepcopy(model).cpu().eval()

    if variant == "fp32":  # 4 bytes
        return m.float(), "fp32"

    if variant == "fp16":  # 2 bytes
        return m.to(dtype=torch.float16), "fp16"

    if variant == "fp64":  # 8 bytes
        return m.to(dtype=torch.float64), "fp64"

    if variant == "bf16":
        return m.to(dtype=torch.bfloat16), "bf16"

    if variant == "fp8":
        quantize_(m, Float8WeightOnlyConfig())
        return m, "fp8"

    if variant == "int8":
        # IMPORTANT: bitsandbytes quantization must happen on load, not after.
        # If SonnetGPT is a custom nn.Module, bnb won't quantize it automatically.
        # You need it to be a HF model (AutoModelForCausalLM / GPT2LMHeadModel).
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

        hf_model = AutoModelForCausalLM.from_pretrained(
            "gpt2",  # <-- or your HF checkpoint dir if you saved it as HF
            quantization_config=quant_config,
            device_map="auto",
        ).eval()

        # If your checkpoint is *not* a HF checkpoint, you cannot directly load
        # SonnetGPT weights into this without a conversion step.
        return hf_model, "int8_bnb"

    raise ValueError(f"Unknown variant: {variant}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dev_gold_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument(
        "--variants",
        type=str,
        default="fp32,bf16,int8",
        help="Comma-separated: fp32,bf16,int8",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Download it locally first."
        )

    base = load_checkpoint_model(args.checkpoint)
    flops_per_token = estimate_flops_per_token_with_seqlen(base, seq_len=args.seq_len)

    print(f"checkpoint: {args.checkpoint}")
    print(f"dev_gold_path: {args.dev_gold_path}")
    print(f"seq_len for FLOPs estimate: {args.seq_len}")
    print("\nvariant,dev_ce,token_acc,ppl,size_mb,approx_flops_per_token")

    for v in [x.strip() for x in args.variants.split(",") if x.strip()]:
        try:
            model_v, label = make_variant(base, v)
            eval_device = args.device
            if label == "int8_dynamic" and args.device == "cuda":
                # torch dynamic int8 quantization runs on CPU kernels.
                eval_device = "cpu"
            metrics = eval_dev_metrics(
                model_v,
                dev_gold_path=args.dev_gold_path,
                batch_size=args.batch_size,
                device=eval_device,
            )
            size_mb = serialized_model_size_mb(model_v)
            print(
                f"{label},{metrics['dev_ce']:.6f},{metrics['token_acc']:.6f},"
                f"{metrics['ppl']:.6f},{size_mb:.2f},{flops_per_token:.0f}"
            )
        except Exception as e:
            print(f"{v},ERROR,{e}")


def _run_benchmark(
    checkpoint: str,
    dev_held_out_path: str,
    dev_gold_path: str,
    batch_size: int,
    seq_len: int,
    variants: str,
    device: str,
    temperature: float,
    top_p: float,
):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    base = load_checkpoint_model(checkpoint)
    flops_per_token = estimate_flops_per_token_with_seqlen(base, seq_len=seq_len)

    lines = [
        f"checkpoint: {checkpoint}",
        f"dev_held_out_path: {dev_held_out_path}",
        f"dev_gold_path: {dev_gold_path}",
        f"seq_len for FLOPs estimate: {seq_len}",
        "",
        "variant,device,dev_chrf,dev_ce,token_acc,ppl,avg_gen_sec_per_sample,gen_tokens_per_sec,size_mb,approx_flops_per_token",
    ]
    for v in [x.strip() for x in variants.split(",") if x.strip()]:
        try:
            model_v, label = make_variant(base, v)
            eval_device = device
            if label == "int8_dynamic" and device == "cuda":
                eval_device = "cpu"
            metrics = eval_dev_metrics(
                model_v,
                dev_gold_path=dev_gold_path,
                batch_size=batch_size,
                device=eval_device,
            )
            gen_eval = eval_dev_chrf(
                model_v,
                dev_held_out_path=dev_held_out_path,
                dev_gold_path=dev_gold_path,
                device=eval_device,
                temperature=temperature,
                top_p=top_p,
            )
            size_mb = serialized_model_size_mb(model_v)
            lines.append(
                f"{label},{eval_device},{gen_eval['dev_chrf']:.6f},{metrics['dev_ce']:.6f},{metrics['token_acc']:.6f},"
                f"{metrics['ppl']:.6f},{gen_eval['avg_gen_sec_per_sample']:.6f},{gen_eval['gen_tokens_per_sec']:.3f},"
                f"{size_mb:.2f},{flops_per_token:.0f}"
            )
        except Exception as e:
            lines.append(f"{v},ERROR,{e}")
    return "\n".join(lines) + "\n"


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    MODAL_WORKSPACE = "/workspace"
    MODAL_CHECKPOINTS = "/checkpoints"

    quant_image = (
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
            "numpy",
            "pandas",
            "torchao",
        )
        .workdir(MODAL_WORKSPACE)
        .add_local_dir(".", remote_path=MODAL_WORKSPACE, ignore=[".git", ".git/*"])
    )

    volume = modal.Volume.from_name("sonnet-checkpoints", create_if_missing=True)
    app = modal.App("sonnet-quantization-benchmark")

    @app.function(
        image=quant_image,
        gpu="L4:4",
        timeout=3600 * 2,
        volumes={MODAL_CHECKPOINTS: volume},
    )
    def run_quant_benchmark_modal(
        checkpoint_in_volume: str,
        dev_held_out_path: str = "data/sonnets_held_out_dev.txt",
        dev_gold_path: str = "data/TRUE_sonnets_held_out_dev.txt",
        batch_size: int = 8,
        seq_len: int = 128,
        variants: str = "fp32,fp16,fp64,bf16,fp8,int8",
        device: str = "cuda",
        temperature: float = 1.2,
        top_p: float = 0.9,
    ) -> str:
        sys.path.insert(0, MODAL_WORKSPACE)
        checkpoint_path = os.path.join(MODAL_CHECKPOINTS, checkpoint_in_volume)
        dev_held_out_full = os.path.join(MODAL_WORKSPACE, dev_held_out_path)
        dev_gold_full = os.path.join(MODAL_WORKSPACE, dev_gold_path)
        return _run_benchmark(
            checkpoint=checkpoint_path,
            dev_held_out_path=dev_held_out_full,
            dev_gold_path=dev_gold_full,
            batch_size=batch_size,
            seq_len=seq_len,
            variants=variants,
            device=device,
            temperature=temperature,
            top_p=top_p,
        )

    @app.local_entrypoint()
    def main_modal(
        checkpoint_in_volume: str,
        dev_held_out_path: str = "data/sonnets_held_out_dev.txt",
        dev_gold_path: str = "data/TRUE_sonnets_held_out_dev.txt",
        batch_size: int = 8,
        seq_len: int = 128,
        variants: str = "fp32,bf16,int8",
        device: str = "cuda",
        temperature: float = 1.2,
        top_p: float = 0.9,
        out: str = "predictions/quantization/quant_report.csv",
    ):
        report = run_quant_benchmark_modal.remote(
            checkpoint_in_volume=checkpoint_in_volume,
            dev_held_out_path=dev_held_out_path,
            dev_gold_path=dev_gold_path,
            batch_size=batch_size,
            seq_len=seq_len,
            variants=variants,
            device=device,
            temperature=temperature,
            top_p=top_p,
        )
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        print(f"Saved report to {out}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "modal":
        # Keep local CLI behavior explicit when desired.
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main()
    else:
        main()
